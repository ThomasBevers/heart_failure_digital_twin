"""EchoNet-Dynamic inference for the app's optional "EchoNet" volume source.

Self-contained: this does not import a separate `echo` package (this project
doesn't have one). It uses the vendored `echonet` package's own video loader
(`echonet.utils.loadvideo`) plus:

  - EF: the official r2plus1d_18 EF checkpoint, loaded directly with
    torchvision -- no vendor function for "just predict EF on this video"
    exists upstream, so that part is implemented here.
  - EDV/ESV: the project's own trained `VolumeEstimator` artifact (a plain
    dict of two scikit-learn regressors saved with joblib) -- again, no
    upstream function does this; it's this project's own research code.

Both artifacts already exist in outputs/ once you've run
`scripts/download_echonet_weights.py` (EF) and `scripts/train_volume_estimator.py`
(EDV/ESV). A missing file raises a clear ImagingError instead of a silent
None or a fabricated placeholder number.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # src/heart_twin/echonet_integration.py -> project root
_VENDOR_DIR = ROOT / "vendor" / "echonet_dynamic"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from .imaging import ImagingError

DEFAULT_EF_CHECKPOINT = ROOT / "outputs" / "echonet_weights" / "r2plus1d_18_32_2_pretrained.pt"
DEFAULT_VOLUME_ESTIMATOR = ROOT / "outputs" / "echonet_volume_estimator.joblib"

# Standard Kinetics video-model normalization (torchvision's own r2plus1d_18
# training stats). The original EchoNet paper instead computes mean/std from
# its own training set at runtime (echonet.utils.get_mean_and_std); doing
# that here would require the full ~10,000-video dataset on disk just to run
# inference on one new video, which isn't practical for this app. Using the
# fixed constants is a documented approximation, not a fabricated result.
_CLIP_MEAN = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)
_CLIP_STD = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)

VOLUME_FEATURE_NAMES = (
    "frame_count",
    "fps",
    "width",
    "height",
    "mean_intensity",
    "intensity_std",
    "temporal_change",
)


def _resolve_path(explicit: Optional[Path], env_var: str, default: Path) -> Path:
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(env_var)
    return Path(from_env) if from_env else default


def infer_ef_from_video(
    video_path: Path,
    *,
    ef_weights: Optional[Path] = None,
) -> float:
    """Predict EF (%) using the official EchoNet-Dynamic EF checkpoint.

    Checks (in order): an explicit `ef_weights` argument, the
    `ECHONET_EF_CHECKPOINT` environment variable, then
    outputs/echonet_weights/r2plus1d_18_32_2_pretrained.pt.
    """
    checkpoint = _resolve_path(ef_weights, "ECHONET_EF_CHECKPOINT", DEFAULT_EF_CHECKPOINT)
    if not checkpoint.exists():
        raise ImagingError(
            f"EF checkpoint not found at {checkpoint}. "
            "Run `python scripts/download_echonet_weights.py`, or set the "
            "ECHONET_EF_CHECKPOINT environment variable to your own checkpoint."
        )
    try:
        import torch
        import torchvision
        import echonet  # noqa: F401  (vendored package; needed for loadvideo)
    except Exception as error:
        # Broad on purpose: a broken/partial torch install can raise OSError
        # (missing native libs), not just ImportError, and either way the
        # user needs the same actionable message, not a raw traceback.
        raise ImagingError(f"EF inference requires a working torch + torchvision install: {error}") from error

    video_path = Path(video_path)
    if not video_path.exists():
        raise ImagingError(f"Video file does not exist: {video_path}")

    try:
        clip = echonet.utils.loadvideo(str(video_path)).astype(np.float32)  # (C, F, H, W), 0-255
    except Exception as error:
        raise ImagingError(f"Could not read video for EF inference: {error}") from error

    c, f, h, w = clip.shape
    frame_count = min(f, 32)
    indices = np.linspace(0, f - 1, frame_count).astype(int)
    clip = clip[:, indices, :, :] / 255.0
    clip = (clip - _CLIP_MEAN.reshape(3, 1, 1, 1)) / _CLIP_STD.reshape(3, 1, 1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.video.r2plus1d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ImagingError(f"Could not read checkpoint {checkpoint}: {error}") from error
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ImagingError(f"Checkpoint is incompatible with the expected EF architecture: {error}") from error

    model.to(device).eval()
    with torch.no_grad():
        tensor = torch.from_numpy(clip[None]).float().to(device)
        ef = float(model(tensor).view(-1)[0].item())
    return ef


def video_features(video_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise ImagingError("OpenCV is required for volume-estimator inference.") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ImagingError(f"Cannot open video for volume inference: {video_path}")
    frames = []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    try:
        while len(frames) < 32:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or len(frames))
    finally:
        capture.release()
    if not frames:
        raise ImagingError(f"Video contains no readable frames: {video_path}")

    values = np.asarray(frames, dtype=np.float32) / 255.0
    temporal_change = float(np.abs(np.diff(values, axis=0)).mean()) if len(values) > 1 else 0.0
    return np.asarray(
        [frame_count, fps, width, height, float(values.mean()), float(values.std()), temporal_change],
        dtype=np.float32,
    )


def infer_volumes_from_segmentation(
    video_path: Path,
    *,
    volume_weights: Optional[Path] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate EDV/ESV (mL) using the project's trained volume estimator.

    This is a video-feature regression proxy (see scripts/train_volume_estimator.py)
    -- NOT a true LV-segmentation-based volume. Returns (None, None), never a
    fabricated placeholder, when no estimator has been trained/configured.

    Checks (in order): an explicit `volume_weights` argument, the
    `ECHONET_VOLUME_ESTIMATOR` environment variable, then
    outputs/echonet_volume_estimator.joblib.
    """
    artifact_path = _resolve_path(volume_weights, "ECHONET_VOLUME_ESTIMATOR", DEFAULT_VOLUME_ESTIMATOR)
    if not artifact_path.exists():
        return None, None

    try:
        import joblib
    except ImportError as error:
        raise ImagingError("joblib is required to load the volume estimator.") from error

    artifact = joblib.load(artifact_path)
    if tuple(artifact.get("feature_names", ())) != VOLUME_FEATURE_NAMES:
        raise ImagingError(f"Volume estimator artifact at {artifact_path} uses incompatible features.")

    features = video_features(Path(video_path)).reshape(1, -1)
    edv = max(0.0, float(artifact["edv_model"].predict(features)[0]))
    esv = min(edv, max(0.0, float(artifact["esv_model"].predict(features)[0])))
    return edv, esv
