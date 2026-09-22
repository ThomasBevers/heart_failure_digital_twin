"""Echo video/DICOM technical inspection and EchoNet inference endpoints.

Deliberately two separate endpoints, matching how the rest of this project
treats inference as something that must be explicitly opted into:
/inspect never runs EF/volume inference (metadata + preview only), and
/infer is the only endpoint that does.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.heart_twin.echonet_integration import infer_ef_from_video, infer_volumes_from_segmentation
from backend.heart_twin.imaging import ImagingError, convert_dicom_to_avi, inspect_video, preview_frame_png

from ..schemas import EchoInferResponse, EchoInspectResponse, EchoMetadataOut

router = APIRouter(prefix="/api/echo", tags=["echo"])

ALLOWED_SUFFIXES = {".dcm", ".dicom", ".avi", ".mp4", ".mov"}
DICOM_SUFFIXES = {".dcm", ".dicom"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB, matching the upload limit shown in the UI


async def _write_upload(file: UploadFile, tmpdir: Path) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    source = tmpdir / f"upload{suffix}"
    source.write_bytes(content)
    return source


def _preview_base64(video_path: Path) -> str:
    return base64.b64encode(preview_frame_png(video_path)).decode("ascii")


@router.post(
    "/inspect",
    response_model=EchoInspectResponse,
    summary="Inspect a video/DICOM's technical metadata and preview (no inference)",
    description=(
        "Upload only research-approved, de-identified files. Reports technical video metadata "
        "(frame count, frame rate, dimensions, pixel spacing if available from DICOM) and a "
        "centre-frame preview image. Never runs EF or volume inference -- use /api/echo/infer for that."
    ),
)
async def inspect_echo(file: UploadFile = File(...), view: str = Form("Unknown")) -> EchoInspectResponse:
    with tempfile.TemporaryDirectory(prefix="heart_twin_upload_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        try:
            source = await _write_upload(file, tmpdir)
            suffix = source.suffix.lower()
            if suffix in DICOM_SUFFIXES:
                converted = tmpdir / "converted_echo.avi"
                metadata = convert_dicom_to_avi(source, converted, view=view)
                model_input = converted
            else:
                model_input = source
                metadata = inspect_video(model_input, view=view)
            preview_b64 = _preview_base64(model_input)
        except ImagingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return EchoInspectResponse(
            metadata=EchoMetadataOut(
                study_id=metadata.study_id,
                filename=file.filename or "unknown",
                input_format=metadata.input_format,
                frame_count=metadata.frame_count,
                frame_rate_fps=metadata.frame_rate_fps,
                frame_height=metadata.frame_height,
                frame_width=metadata.frame_width,
                pixel_spacing_mm=metadata.pixel_spacing_mm,
                view=metadata.view,
                provenance=metadata.provenance,
            ),
            preview_png_base64=preview_b64,
        )


@router.post(
    "/infer",
    response_model=EchoInferResponse,
    summary="Run real EchoNet EF (and, if configured, EDV/ESV) inference",
    description=(
        "Runs the official EchoNet-Dynamic EF checkpoint on the uploaded video/DICOM, plus the "
        "project's own trained volume estimator if one has been configured. Returns a clear 422 "
        "(not a silent None or a fabricated number) if the EF checkpoint is missing -- see "
        "scripts/download_echonet_weights.py."
    ),
)
async def infer_echo(file: UploadFile = File(...), view: str = Form("A4C")) -> EchoInferResponse:
    with tempfile.TemporaryDirectory(prefix="heart_twin_upload_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        try:
            source = await _write_upload(file, tmpdir)
            suffix = source.suffix.lower()
            if suffix in DICOM_SUFFIXES:
                converted = tmpdir / "converted_echo.avi"
                convert_dicom_to_avi(source, converted, view=view)
                model_input = converted
            else:
                model_input = source
            ef = infer_ef_from_video(model_input)
            edv, esv = infer_volumes_from_segmentation(model_input)
            preview_b64 = _preview_base64(model_input)
        except ImagingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return EchoInferResponse(ef_percent=ef, edv_ml=edv, esv_ml=esv, preview_png_base64=preview_b64)
