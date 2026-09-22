"""Patient validation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.heart_twin.patient import PatientInputError

from ..conversions import patient_input_to_profile, volumes_to_out
from ..schemas import PatientInput, VentricularVolumesOut

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post(
    "/validate",
    response_model=VentricularVolumesOut,
    summary="Validate patient measurements and resolve EDV/ESV",
    description=(
        "Runs the same validation heart_twin.PatientProfile applies (range checks, "
        "systolic > diastolic, EDV/ESV consistency with LVEF, ...) and returns the "
        "resolved EDV/ESV volumes without running the full simulation. Useful for a "
        "frontend to validate a form and preview the resolved volume source before "
        "committing to a /api/simulate call."
    ),
)
def validate_patient(payload: PatientInput) -> VentricularVolumesOut:
    try:
        profile = patient_input_to_profile(payload)
        volumes = profile.ventricular_volumes()
    except PatientInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return volumes_to_out(volumes)
