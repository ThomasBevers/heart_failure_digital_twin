"""
EchoNet-Dynamic inference for the heart-failure digital twin.

This file uses an EchoNet model to predict EF from an echo video
and a project-trained model to estimate EDV and ESV.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Keep the vendored Python package with the backend, but keep generated models
# at the project root.  The original Streamlit project and its README both use
# ``project_root/outputs``; using ``backend/outputs`` here made a correctly
# downloaded checkpoint invisible to the API.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
_VENDOR_DIR = BACKEND_ROOT / "vendor" / "echonet_dynamic"

if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from .imaging import ImagingError


# Default locations of the trained models.
DEFAULT_EF_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "echonet_weights" / "r2plus1d_18_32_2_pretrained.pt"
)
DEFAULT_VOLUME_ESTIMATOR = (
    PROJECT_ROOT / "outputs" / "echonet_volume_estimator.joblib"
)


# Mean and standard deviation used to normalize the video before EF prediction.
_CLIP_MEAN = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)
_CLIP_STD = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)


# Features used by the project's EDV/ESV prediction model.
VOLUME_FEATURE_NAMES = (
    "frame_count",
    "fps",
    "width",
    "height",
    "mean_intensity",
    "intensity_std",
    "temporal_change",
)


def _resolve_path(
    explicit: Optional[Path],
    env_var: str,
    default: Path,
) -> Path:
    """Choose a model path from an argument, environment variable, or default."""
    if explicit is not None:
        return Path(explicit)

    from_env = os.environ.get(env_var)
    return Path(from_env) if from_env else default


def infer_ef_from_video(
    video_path: Path,
    *,
    ef_weights: Optional[Path] = None,
) -> float:
    """Use the EchoNet model to predict ejection fraction (EF) from a video."""

    # Find the EF model and check that it exists.
    checkpoint = _resolve_path(
        ef_weights,
        "ECHONET_EF_CHECKPOINT",
        DEFAULT_EF_CHECKPOINT,
    )

    if not checkpoint.exists():
        raise ImagingError(
            f"EF checkpoint not found at {checkpoint}. "
            "Run `python scripts/download_echonet_weights.py`, or set the "
            "ECHONET_EF_CHECKPOINT environment variable to your own checkpoint."
        )

    # Load the libraries needed for EchoNet inference.
    try:
        import torch
        import torchvision
        import echonet  # noqa: F401

    except Exception as error:
        raise ImagingError(
            f"EF inference requires a working torch + torchvision install: {error}"
        ) from error

    video_path = Path(video_path)

    if not video_path.exists():
        raise ImagingError(f"Video file does not exist: {video_path}")

    # Load the echo video using EchoNet's video loader.
    try:
        clip = echonet.utils.loadvideo(
            str(video_path)
        ).astype(np.float32)

    except Exception as error:
        raise ImagingError(
            f"Could not read video for EF inference: {error}"
        ) from error

    # Select up to 32 frames and normalize their pixel values.
    c, f, h, w = clip.shape
    frame_count = min(f, 32)
    indices = np.linspace(0, f - 1, frame_count).astype(int)

    clip = clip[:, indices, :, :] / 255.0
    clip = (
        clip - _CLIP_MEAN.reshape(3, 1, 1, 1)
    ) / _CLIP_STD.reshape(3, 1, 1, 1)

    # Use the GPU if available; otherwise use the CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the R(2+1)D-18 model with one output: EF.
    model = torchvision.models.video.r2plus1d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    # Load the trained EchoNet weights.
    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as error:
        raise ImagingError(
            f"Could not read checkpoint {checkpoint}: {error}"
        ) from error

    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    try:
        model.load_state_dict(state_dict)

    except RuntimeError as error:
        raise ImagingError(
            f"Checkpoint is incompatible with the expected EF architecture: {error}"
        ) from error

    # Run the model and return the predicted EF.
    model.to(device).eval()

    with torch.no_grad():
        tensor = torch.from_numpy(
            clip[None]
        ).float().to(device)

        ef = float(
            model(tensor).view(-1)[0].item()
        )

    return ef


def video_features(video_path: Path) -> np.ndarray:
    """Extract simple video features used by the EDV/ESV model."""

    # OpenCV is used to read and analyse the video.
    try:
        import cv2

    except ImportError as error:
        raise ImagingError(
            "OpenCV is required for volume-estimator inference."
        ) from error

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ImagingError(
            f"Cannot open video for volume inference: {video_path}"
        )

    frames = []

    # Read basic video information.
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Read up to 32 frames and convert them to small grayscale images.
    try:
        while len(frames) < 32:
            ok, frame = capture.read()

            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(
                cv2.resize(
                    gray,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
            )

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT) or len(frames)
        )

    finally:
        capture.release()

    if not frames:
        raise ImagingError(
            f"Video contains no readable frames: {video_path}"
        )

    # Calculate brightness and frame-to-frame movement features.
    values = np.asarray(
        frames,
        dtype=np.float32,
    ) / 255.0

    temporal_change = (
        float(np.abs(np.diff(values, axis=0)).mean())
        if len(values) > 1
        else 0.0
    )

    return np.asarray(
        [
            frame_count,
            fps,
            width,
            height,
            float(values.mean()),
            float(values.std()),
            temporal_change,
        ],
        dtype=np.float32,
    )


def infer_volumes_from_segmentation(
    video_path: Path,
    *,
    volume_weights: Optional[Path] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate EDV and ESV from an echo video using the project's
    trained volume regression models.
    """

    # Find the trained EDV/ESV model.
    artifact_path = _resolve_path(
        volume_weights,
        "ECHONET_VOLUME_ESTIMATOR",
        DEFAULT_VOLUME_ESTIMATOR,
    )

    # If no model exists, return no volume result instead of a fake value.
    if not artifact_path.exists():
        return None, None

    # Load the trained scikit-learn models.
    try:
        import joblib

    except ImportError as error:
        raise ImagingError(
            "joblib is required to load the volume estimator."
        ) from error

    artifact = joblib.load(artifact_path)

    # Make sure the saved model expects the same features we calculate here.
    if tuple(
        artifact.get("feature_names", ())
    ) != VOLUME_FEATURE_NAMES:

        raise ImagingError(
            f"Volume estimator artifact at {artifact_path} uses incompatible features."
        )

    # Extract features from the video and use them for prediction.
    features = video_features(
        Path(video_path)
    ).reshape(1, -1)

    edv = max(
        0.0,
        float(
            artifact["edv_model"].predict(features)[0]
        ),
    )

    esv = min(
        edv,
        max(
            0.0,
            float(
                artifact["esv_model"].predict(features)[0]
            ),
        ),
    )

    return edv, esv
