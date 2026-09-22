"""Baseline + what-if scenario simulation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.heart_twin.patient import PatientInputError
from backend.heart_twin.twin import ScenarioControls, run_twin

from ..conversions import patient_input_to_profile, twin_state_to_out, volumes_to_out
from ..schemas import SimulateRequest, SimulateResponse

router = APIRouter(prefix="/api", tags=["simulate"])


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="Calibrate a patient baseline and run a what-if scenario",
    description=(
        "Calibrates the closed-loop cardiovascular model so its baseline reproduces the "
        "patient's entered LVEF, then applies the given what-if perturbations (contractility, "
        "preload, afterload, heart rate) on top of that fitted baseline. Both baseline and "
        "scenario traces are returned, each trimmed to their final 2 cardiac cycles."
    ),
)
def simulate(payload: SimulateRequest) -> SimulateResponse:
    try:
        profile = patient_input_to_profile(payload.patient)
        controls = ScenarioControls(**payload.controls.model_dump())
        comparison = run_twin(profile, controls)
    except (PatientInputError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    scenario_heart_rate = profile.heart_rate_bpm + controls.heart_rate_delta_bpm
    return SimulateResponse(
        volume_input=volumes_to_out(comparison.volume_input),
        baseline=twin_state_to_out(comparison.baseline, profile.heart_rate_bpm),
        scenario=twin_state_to_out(comparison.scenario, scenario_heart_rate),
    )
