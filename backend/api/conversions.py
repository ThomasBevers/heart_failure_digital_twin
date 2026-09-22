"""Conversions between heart_twin domain objects and API (Pydantic) schemas."""

from __future__ import annotations

from dataclasses import asdict

from backend.heart_twin.patient import PatientProfile, VentricularVolumes
from backend.heart_twin.physiology import SimulationTrace
from backend.heart_twin.twin import TwinState

from .schemas import (
    PatientInput,
    SimulationTraceOut,
    TwinStateOut,
    TwinSummaryOut,
    VentricularVolumesOut,
)


def patient_input_to_profile(payload: PatientInput) -> PatientProfile:
    """Construct the domain PatientProfile; its own __post_init__ remains the
    authoritative validation (the Pydantic model only fails fast on shape)."""
    return PatientProfile(**payload.model_dump())


def volumes_to_out(volumes: VentricularVolumes) -> VentricularVolumesOut:
    return VentricularVolumesOut(
        edv_ml=volumes.edv_ml,
        esv_ml=volumes.esv_ml,
        stroke_volume_ml=volumes.stroke_volume_ml,
        ejection_fraction_percent=volumes.ejection_fraction_percent,
        source=volumes.source,
        note=volumes.note,
    )


def trim_trace(trace: SimulationTrace, heart_rate_bpm: float, cycles: int = 2) -> SimulationTraceOut:
    """Keep only the final ``cycles`` cardiac cycles of a simulated trace.

    Full traces run 12 simulated seconds at dt=0.002s (6001 points); a
    frontend chart only ever needs the settled final cycles, and trimming
    here keeps the JSON response small regardless of how long the caller's
    simulation window is.
    """
    heart_rate_bpm = max(1.0, heart_rate_bpm)
    window_s = cycles * 60.0 / heart_rate_bpm
    start_time = trace.time_s[-1] - window_s
    keep = [index for index, t in enumerate(trace.time_s) if t >= start_time]
    if not keep:
        keep = list(range(len(trace.time_s)))
    return SimulationTraceOut(
        time_s=[trace.time_s[i] for i in keep],
        ventricular_volume_ml=[trace.ventricular_volume_ml[i] for i in keep],
        ventricular_pressure_mmhg=[trace.ventricular_pressure_mmhg[i] for i in keep],
        arterial_pressure_mmhg=[trace.arterial_pressure_mmhg[i] for i in keep],
        venous_pressure_mmhg=[trace.venous_pressure_mmhg[i] for i in keep],
        aortic_flow_ml_s=[trace.aortic_flow_ml_s[i] for i in keep],
        mitral_flow_ml_s=[trace.mitral_flow_ml_s[i] for i in keep],
    )


def twin_state_to_out(state: TwinState, heart_rate_bpm: float, cycles: int = 2) -> TwinStateOut:
    return TwinStateOut(
        summary=TwinSummaryOut(**asdict(state.summary)),
        trace=trim_trace(state.trace, heart_rate_bpm, cycles=cycles),
    )
