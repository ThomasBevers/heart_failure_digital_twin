"""Patient-specific initialisation and what-if runs for the research twin."""

from __future__ import annotations

from dataclasses import dataclass

from .patient import PatientProfile, VentricularVolumes
from .physiology import CardiovascularParameters, ClosedLoopCardiovascularModel, SimulationTrace, cycle_summary


@dataclass(frozen=True)
class ScenarioControls:
    """Dimensionless perturbations applied to a calibrated baseline."""

    contractility_percent: float = 0.0
    preload_percent: float = 0.0
    afterload_percent: float = 0.0
    heart_rate_delta_bpm: float = 0.0

    def __post_init__(self) -> None:
        if not -50.0 <= self.contractility_percent <= 50.0:
            raise ValueError("Contractility change must be between -50% and +50%.")
        if not -40.0 <= self.preload_percent <= 40.0:
            raise ValueError("Preload change must be between -40% and +40%.")
        if not -40.0 <= self.afterload_percent <= 40.0:
            raise ValueError("Afterload change must be between -40% and +40%.")
        if not -30.0 <= self.heart_rate_delta_bpm <= 30.0:
            raise ValueError("Heart-rate change must be between -30 and +30 bpm.")


@dataclass(frozen=True)
class TwinSummary:
    ejection_fraction_percent: float
    end_diastolic_volume_ml: float
    end_systolic_volume_ml: float
    stroke_volume_ml: float
    cardiac_output_l_min: float
    mean_arterial_pressure_mmhg: float
    emax_mmhg_per_ml: float
    systemic_resistance_mmhg_s_per_ml: float
    arterial_compliance_ml_per_mmhg: float


@dataclass(frozen=True)
class TwinState:
    summary: TwinSummary
    trace: SimulationTrace


@dataclass(frozen=True)
class TwinComparison:
    volume_input: VentricularVolumes
    baseline: TwinState
    scenario: TwinState


def _make_parameters(
    patient: PatientProfile,
    volumes: VentricularVolumes,
    *,
    emax: float,
    controls: ScenarioControls,
) -> CardiovascularParameters:
    heart_rate = max(30.0, patient.heart_rate_bpm + controls.heart_rate_delta_bpm)
    target_flow_ml_s = max(1.0, volumes.stroke_volume_ml * heart_rate / 60.0)
    baseline_resistance = patient.mean_arterial_pressure_mmhg / target_flow_ml_s
    pulse_pressure = patient.systolic_bp_mmhg - patient.diastolic_bp_mmhg
    arterial_compliance = min(4.0, max(0.5, volumes.stroke_volume_ml / pulse_pressure))
    return CardiovascularParameters(
        emax_mmhg_per_ml=emax * (1.0 + controls.contractility_percent / 100.0),
        systemic_resistance_mmhg_s_per_ml=baseline_resistance * (1.0 + controls.afterload_percent / 100.0),
        arterial_compliance_ml_per_mmhg=arterial_compliance,
        heart_rate_bpm=heart_rate,
    )


def _run(
    patient: PatientProfile,
    volumes: VentricularVolumes,
    parameters: CardiovascularParameters,
    controls: ScenarioControls,
) -> TwinState:
    preload_factor = 1.0 + controls.preload_percent / 100.0
    trace = ClosedLoopCardiovascularModel(parameters).simulate(
        initial_lv_volume_ml=volumes.edv_ml * preload_factor,
        initial_arterial_pressure_mmhg=patient.mean_arterial_pressure_mmhg,
        initial_venous_pressure_mmhg=8.0 * preload_factor,
    )
    edv, esv, map_mmhg, ef = cycle_summary(trace, parameters.heart_rate_bpm)
    stroke_volume = edv - esv
    return TwinState(
        TwinSummary(
            ejection_fraction_percent=ef,
            end_diastolic_volume_ml=edv,
            end_systolic_volume_ml=esv,
            stroke_volume_ml=stroke_volume,
            cardiac_output_l_min=stroke_volume * parameters.heart_rate_bpm / 1000.0,
            mean_arterial_pressure_mmhg=map_mmhg,
            emax_mmhg_per_ml=parameters.emax_mmhg_per_ml,
            systemic_resistance_mmhg_s_per_ml=parameters.systemic_resistance_mmhg_s_per_ml,
            arterial_compliance_ml_per_mmhg=parameters.arterial_compliance_ml_per_mmhg,
        ),
        trace,
    )


def _calibrate_emax(patient: PatientProfile, volumes: VentricularVolumes) -> float:
    """Find Emax that reproduces the entered EF at the baseline operating point."""
    target_ef = patient.lvef_percent
    low, high = 0.10, 5.00
    neutral = ScenarioControls()

    for _ in range(16):
        candidate = (low + high) / 2.0
        parameters = _make_parameters(
            patient,
            volumes,
            emax=candidate,
            controls=neutral,
        )

        try:
            predicted_ef = _run(
                patient,
                volumes,
                parameters,
                neutral,
            ).summary.ejection_fraction_percent
        except ValueError as error:
            raise ValueError(
                f"Patient baseline cannot be simulated: {error}"
            ) from error

        if predicted_ef < target_ef:
            low = candidate
        else:
            high = candidate

    return (low + high) / 2.0


def run_twin(patient: PatientProfile, controls: ScenarioControls = ScenarioControls()) -> TwinComparison:
    """Calibrate a patient baseline, then simulate a controlled perturbation."""
    volumes = patient.ventricular_volumes()
    emax = _calibrate_emax(patient, volumes)
    neutral = ScenarioControls()
    baseline_parameters = _make_parameters(patient, volumes, emax=emax, controls=neutral)
    scenario_parameters = _make_parameters(patient, volumes, emax=emax, controls=controls)
    return TwinComparison(
        volume_input=volumes,
        baseline=_run(patient, volumes, baseline_parameters, neutral),
        scenario=_run(patient, volumes, scenario_parameters, controls),
    )
