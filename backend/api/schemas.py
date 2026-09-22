"""Pydantic request/response models.

These mirror heart_twin's domain dataclasses (PatientProfile, ScenarioControls,
VentricularVolumes, TwinSummary, SimulationTrace) field-for-field, but are
kept as a separate layer on purpose: the domain package has no FastAPI/Pydantic
dependency, and these models add API-level concerns (request validation,
JSON-friendly shapes, docs) without the domain logic ever importing a web
framework.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class PatientInput(BaseModel):
    """Clinician-entered measurements for one digital-twin session.

    Field ranges mirror heart_twin.patient.PatientProfile's own validation,
    so obviously-invalid requests fail fast with a structured 422 before
    ever reaching the simulator. PatientProfile's own checks (e.g. EDV/ESV
    consistency with LVEF) still run as the authoritative source of truth.
    """

    identifier: str = Field(..., min_length=1, description="Patient/session identifier.")
    heart_rate_bpm: float = Field(..., gt=0, le=240)
    systolic_bp_mmhg: float = Field(..., gt=0, le=300)
    diastolic_bp_mmhg: float = Field(..., gt=0, le=200)
    lvef_percent: float = Field(..., gt=0, le=100)
    lv_diameter_cm: Optional[float] = Field(None, ge=1, le=10, description="LV end-diastolic diameter (cm).")
    edv_ml: Optional[float] = Field(None, ge=1, le=1000)
    esv_ml: Optional[float] = Field(None, ge=0, le=1000)
    age_band: str = "Not provided"
    sex: str = "Not provided"
    nyha_class: str = "Not provided"

    @model_validator(mode="after")
    def _check_pressures(self) -> "PatientInput":
        if self.systolic_bp_mmhg <= self.diastolic_bp_mmhg:
            raise ValueError("systolic_bp_mmhg must be greater than diastolic_bp_mmhg.")
        if (self.edv_ml is None) != (self.esv_ml is None):
            raise ValueError("Provide both edv_ml and esv_ml, or neither.")
        return self


class ScenarioControlsInput(BaseModel):
    """Dimensionless what-if perturbations applied to the calibrated baseline."""

    contractility_percent: float = Field(0.0, ge=-50, le=50)
    preload_percent: float = Field(0.0, ge=-40, le=40)
    afterload_percent: float = Field(0.0, ge=-40, le=40)
    heart_rate_delta_bpm: float = Field(0.0, ge=-30, le=30)


class SimulateRequest(BaseModel):
    patient: PatientInput
    controls: ScenarioControlsInput = Field(default_factory=ScenarioControlsInput)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class VentricularVolumesOut(BaseModel):
    edv_ml: float
    esv_ml: float
    stroke_volume_ml: float
    ejection_fraction_percent: float
    source: str = Field(..., description="'reported', 'teichholz', or 'reference'.")
    note: str


class TwinSummaryOut(BaseModel):
    ejection_fraction_percent: float
    end_diastolic_volume_ml: float
    end_systolic_volume_ml: float
    stroke_volume_ml: float
    cardiac_output_l_min: float
    mean_arterial_pressure_mmhg: float
    emax_mmhg_per_ml: float
    systemic_resistance_mmhg_s_per_ml: float
    arterial_compliance_ml_per_mmhg: float


class SimulationTraceOut(BaseModel):
    """Trimmed to the final ``cycles`` cardiac cycles (default 2) to keep the payload small."""

    time_s: List[float]
    ventricular_volume_ml: List[float]
    ventricular_pressure_mmhg: List[float]
    arterial_pressure_mmhg: List[float]
    venous_pressure_mmhg: List[float]
    aortic_flow_ml_s: List[float]
    mitral_flow_ml_s: List[float]


class TwinStateOut(BaseModel):
    summary: TwinSummaryOut
    trace: SimulationTraceOut


class SimulateResponse(BaseModel):
    volume_input: VentricularVolumesOut
    baseline: TwinStateOut
    scenario: TwinStateOut


class EchoMetadataOut(BaseModel):
    study_id: str
    filename: str = Field(..., description="Original uploaded filename (server paths are never returned).")
    input_format: str
    frame_count: int
    frame_rate_fps: float
    frame_height: int
    frame_width: int
    pixel_spacing_mm: Optional[Tuple[float, float]]
    view: str
    provenance: str


class EchoInspectResponse(BaseModel):
    metadata: EchoMetadataOut
    preview_png_base64: str = Field(..., description="Centre-frame preview, base64-encoded PNG.")


class EchoInferResponse(BaseModel):
    ef_percent: float
    edv_ml: Optional[float] = Field(None, description="None if no volume estimator is configured.")
    esv_ml: Optional[float] = Field(None, description="None if no volume estimator is configured.")
    preview_png_base64: str


class HealthResponse(BaseModel):
    status: str
    echonet_dynamic_available: bool
    ef_checkpoint_available: bool
    volume_estimator_available: bool
