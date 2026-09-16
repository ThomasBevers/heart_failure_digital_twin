"""Non-destructive echocardiography input inspection and conversion.

The module prepares local research inputs only. It deliberately does not
claim de-identification or make an EF prediction: both need a separately
validated, governed imaging workflow.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class ImagingError(RuntimeError):
    """Raised when an echocardiography file cannot be prepared safely."""


@dataclass(frozen=True)
class EchoMetadata:
    study_id: str
    source_path: str
    input_format: str
    frame_count: int
    frame_rate_fps: float
    frame_height: int
    frame_width: int
    pixel_spacing_mm: Optional[tuple[float, float]]
    view: str
    provenance: str


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImagingError("OpenCV is required for echocardiography preprocessing.") from error
    return cv2


def _video_metadata(path: Path) -> EchoMetadata:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ImagingError(f"Cannot open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0 or frame_rate <= 0:
        raise ImagingError(f"Video has no usable dimensions, frames, or frame rate: {path}")
    return EchoMetadata(
        study_id=path.stem,
        source_path=str(path),
        input_format=path.suffix.lower().lstrip("."),
        frame_count=frame_count,
        frame_rate_fps=frame_rate,
        frame_height=height,
        frame_width=width,
        pixel_spacing_mm=None,
        view="Unknown",
        provenance="Video file metadata only; image view and quality require review.",
    )


def inspect_video(path: Path, *, view: str = "Unknown") -> EchoMetadata:
    """Inspect a video without modifying it."""
    metadata = _video_metadata(Path(path))
    return EchoMetadata(**{**asdict(metadata), "view": view})


def _dicom_frame_rate(dataset) -> float:
    for attribute in ("RecommendedDisplayFrameRate", "CineRate"):
        value = getattr(dataset, attribute, None)
        if value and float(value) > 0:
            return float(value)
    for attribute in ("FrameTime", "ActualFrameDuration"):
        value = getattr(dataset, attribute, None)
        if value and float(value) > 0:
            return 1000.0 / float(value)
    frame_times = getattr(dataset, "FrameTimeVector", None)
    if frame_times:
        valid = [float(value) for value in frame_times if float(value) > 0]
        if valid:
            return 1000.0 / float(np.median(valid))
    return 0.0


def _as_frame_sequence(pixel_array: np.ndarray, samples_per_pixel: int, frame_count: int) -> np.ndarray:
    """Normalise common DICOM pixel layouts to (frames, height, width, channels?)."""
    pixels = np.asarray(pixel_array)
    if pixels.ndim == 2:
        return pixels[None, ...]
    if pixels.ndim == 3:
        if samples_per_pixel > 1 and pixels.shape[-1] in (3, 4):
            return pixels[None, ...]
        return pixels
    if pixels.ndim == 4:
        if pixels.shape[-1] in (1, 3, 4):
            return pixels
        if pixels.shape[1] in (1, 3, 4):
            return np.moveaxis(pixels, 1, -1)
    raise ImagingError(
        f"Unsupported DICOM pixel array shape {pixels.shape}; expected monochrome or RGB cine frames."
    )


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - low) * 255.0 / (high - low)).clip(0, 255).astype(np.uint8)


def convert_dicom_to_avi(
    dicom_path: Path,
    output_path: Path,
    *,
    view: str = "Unknown",
    crop_top_fraction: float = 0.0,
    fallback_frame_rate_fps: float = 25.0,
) -> EchoMetadata:
    """Convert a cine DICOM to AVI while preserving the original DICOM file.

    ``crop_top_fraction`` can remove a known text band from a research copy,
    but it is not a de-identification guarantee.
    """
    if not 0.0 <= crop_top_fraction < 0.5:
        raise ImagingError("crop_top_fraction must be between 0.0 and 0.5.")
    if fallback_frame_rate_fps <= 0:
        raise ImagingError("fallback_frame_rate_fps must be positive.")
    try:
        import pydicom
    except ImportError as error:
        raise ImagingError("pydicom is required to convert DICOM studies.") from error
    path = Path(dicom_path)
    if not path.is_file():
        raise ImagingError(f"DICOM file does not exist: {path}")
    dataset = pydicom.dcmread(str(path), force=False)
    if not hasattr(dataset, "PixelData"):
        raise ImagingError(f"DICOM file contains no pixel data: {path}")
    try:
        frames = _as_frame_sequence(
            dataset.pixel_array,
            int(getattr(dataset, "SamplesPerPixel", 1) or 1),
            int(getattr(dataset, "NumberOfFrames", 1) or 1),
        )
    except Exception as error:
        if isinstance(error, ImagingError):
            raise
        raise ImagingError("DICOM pixel data could not be decoded by the installed codecs.") from error

    cv2 = _cv2()
    frame_rate = _dicom_frame_rate(dataset)
    used_fallback = frame_rate <= 0
    frame_rate = fallback_frame_rate_fps if used_fallback else frame_rate
    processed: list[np.ndarray] = []
    for frame in frames:
        image = _to_uint8(frame)
        if image.ndim == 3:
            if image.shape[-1] == 1:
                image = image[..., 0]
            elif image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                raise ImagingError(f"Unsupported colour frame shape {image.shape}.")
        if image.ndim != 2:
            raise ImagingError(f"Unsupported frame shape {image.shape}.")
        if crop_top_fraction:
            image = image[round(image.shape[0] * crop_top_fraction):, :]
        processed.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if not processed:
        raise ImagingError("DICOM study contains no image frames.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = processed[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), frame_rate, (width, height))
    if not writer.isOpened():
        raise ImagingError(f"Cannot create AVI output: {output}")
    try:
        for frame in processed:
            writer.write(frame)
    finally:
        writer.release()
    spacing = getattr(dataset, "PixelSpacing", None)
    pixel_spacing = tuple(float(item) for item in spacing[:2]) if spacing is not None and len(spacing) >= 2 else None
    return EchoMetadata(
        study_id=output.stem,
        source_path=str(path),
        input_format="dcm",
        frame_count=len(processed),
        frame_rate_fps=frame_rate,
        frame_height=height,
        frame_width=width,
        pixel_spacing_mm=pixel_spacing,
        view=view,
        provenance=(
            "DICOM pixel data converted to a grayscale AVI derivative; source DICOM retained."
            + (" Frame-rate fallback of 25 FPS used because timing was absent." if used_fallback else "")
            + (" Top crop applied; this is not a de-identification guarantee." if crop_top_fraction else "")
        ),
    )


def preview_frame_png(video_path: Path) -> bytes:
    """Return a centre-frame PNG preview without relying on browser video codecs."""
    cv2 = _cv2()
    metadata = _video_metadata(Path(video_path))
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, metadata.frame_count // 2)
        success, frame = capture.read()
    finally:
        capture.release()
    if not success:
        raise ImagingError("Video contained no readable frames.")
    encoded, image = cv2.imencode(".png", frame)
    if not encoded:
        raise ImagingError("Could not encode an image preview.")
    return bytes(image)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_echonet_dataset(dataset_dir: Path, sample_size: int = 25) -> dict[str, object]:
    """Check EchoNet metadata consistency without running model inference."""
    dataset_dir = Path(dataset_dir)
    file_list = dataset_dir / "FileList.csv"
    traces = dataset_dir / "VolumeTracings.csv"
    videos = dataset_dir / "Videos"
    for required in (file_list, traces, videos):
        if not required.exists():
            raise ImagingError(f"Required EchoNet input is missing: {required}")
    file_rows = _read_csv(file_list)
    trace_rows = _read_csv(traces)
    expected = {f"{Path(row['FileName']).stem}.avi" for row in file_rows}
    available = {path.name for path in videos.glob("*.avi")}
    traced = {Path(row["FileName"]).name for row in trace_rows}
    sampled = [_video_metadata(videos / name) for name in sorted(available)[:max(0, sample_size)]]
    return {
        "dataset_dir": str(dataset_dir),
        "file_list_rows": len(file_rows),
        "trace_rows": len(trace_rows),
        "video_files": len(available),
        "missing_videos": sorted(expected - available),
        "videos_with_traces": len(expected & traced),
        "sampled_videos": [asdict(item) for item in sampled],
        "usable": bool(file_rows) and not (expected - available),
        "note": "Metadata validation only. EchoNet labels are research labels, not predictions for new studies.",
    }


def write_echonet_manifest(dataset_dir: Path, output_path: Path, sample_size: int = 25) -> Path:
    """Write a compact, reproducible EchoNet input manifest."""
    report = validate_echonet_dataset(dataset_dir, sample_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
