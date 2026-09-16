"""Validated patient inputs and ventricular-volume estimates.

This module deliberately keeps the clinician-entered measurements separate
from derived values. A derived Teichholz volume is useful for a research
simulation, but must never be presented as a measured ventricular volume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal, Optional


class PatientInputError(ValueError):
    """Raised when submitted patient measurements are inconsistent."""


VolumeSource = Literal["reported", "teichholz", "reference"]


@dataclass(frozen=True)
class VentricularVolumes:
    """A paired EDV/ESV estimate used to initialize the simulator."""

    edv_ml: float
    esv_ml: float
    source: VolumeSource
    note: str

    @property
    def stroke_volume_ml(self) -> float:
        return self.edv_ml - self.esv_ml

    @property
    def ejection_fraction_percent(self) -> float:
        return 100.0 * self.stroke_volume_ml / self.edv_ml


@dataclass(frozen=True)
class PatientProfile:
    """Minimal, validated state for one research digital twin session."""

    identifier: str
    heart_rate_bpm: float
    systolic_bp_mmhg: float
    diastolic_bp_mmhg: float
    lvef_percent: float
    lv_diameter_cm: Optional[float] = None
    edv_ml: Optional[float] = None
    esv_ml: Optional[float] = None
    age_band: str = "Not provided"
    sex: str = "Not provided"
    nyha_class: str = "Not provided"
    bnp_pg_ml: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise PatientInputError("Patient ID is required.")
        self._require_range("Heart rate", self.heart_rate_bpm, 20.0, 240.0)
        self._require_range("Systolic blood pressure", self.systolic_bp_mmhg, 40.0, 300.0)
        self._require_range("Diastolic blood pressure", self.diastolic_bp_mmhg, 20.0, 200.0)
        if self.systolic_bp_mmhg <= self.diastolic_bp_mmhg:
            raise PatientInputError("Systolic blood pressure must be greater than diastolic blood pressure.")
        self._require_range("LVEF", self.lvef_percent, 1.0, 100.0)
        if self.lv_diameter_cm is not None:
            self._require_range("LV diameter", self.lv_diameter_cm, 1.0, 10.0)
        if (self.edv_ml is None) != (self.esv_ml is None):
            raise PatientInputError("Provide both EDV and ESV, or leave both blank.")
        if self.edv_ml is not None and self.esv_ml is not None:
            self._require_range("EDV", self.edv_ml, 1.0, 1_000.0)
            self._require_range("ESV", self.esv_ml, 0.0, 1_000.0)
            if self.esv_ml >= self.edv_ml:
                raise PatientInputError("ESV must be lower than EDV.")
            derived_ef = 100.0 * (self.edv_ml - self.esv_ml) / self.edv_ml
            if abs(derived_ef - self.lvef_percent) > 10.0:
                raise PatientInputError(
                    f"EDV and ESV imply an LVEF of {derived_ef:.1f}%, which differs from the entered LVEF by more than 10 points."
                )
        if self.bnp_pg_ml is not None:
            self._require_range("BNP", self.bnp_pg_ml, 0.0, 100_000.0)
        if self.nyha_class not in {"Not provided", "I", "II", "III", "IV"}:
            raise PatientInputError("NYHA class must be I, II, III, IV, or Not provided.")

    @staticmethod
    def _require_range(label: str, value: float, minimum: float, maximum: float) -> None:
        if not isfinite(float(value)) or not minimum <= float(value) <= maximum:
            raise PatientInputError(f"{label} must be between {minimum:g} and {maximum:g}.")

    @property
    def mean_arterial_pressure_mmhg(self) -> float:
        return (self.systolic_bp_mmhg + 2.0 * self.diastolic_bp_mmhg) / 3.0

    def ventricular_volumes(self) -> VentricularVolumes:
        """Resolve a volume pair without hiding which method produced it."""
        if self.edv_ml is not None and self.esv_ml is not None:
            return VentricularVolumes(self.edv_ml, self.esv_ml, "reported", "Clinician-entered EDV and ESV.")
        if self.lv_diameter_cm is not None:
            edv = teichholz_volume_ml(self.lv_diameter_cm)
            return VentricularVolumes(
                edv,
                edv * (1.0 - self.lvef_percent / 100.0),
                "teichholz",
                "EDV estimated from LV end-diastolic diameter using the Teichholz formula.",
            )
        edv = 110.0 if self.sex == "Female" else 140.0 if self.sex == "Male" else 125.0
        return VentricularVolumes(
            edv,
            edv * (1.0 - self.lvef_percent / 100.0),
            "reference",
            "Reference EDV used because direct volumes and LV diameter were not supplied.",
        )

    def with_imaging_measurements(
        self,
        *,
        lvef_percent: float,
        edv_ml: Optional[float] = None,
        esv_ml: Optional[float] = None,
    ) -> "PatientProfile":
        """Return a new profile with externally supplied imaging measurements."""
        if (edv_ml is None) != (esv_ml is None):
            raise PatientInputError("Imaging must provide both EDV and ESV, or neither.")
        return replace(
            self,
            lvef_percent=lvef_percent,
            edv_ml=edv_ml if edv_ml is not None else self.edv_ml,
            esv_ml=esv_ml if esv_ml is not None else self.esv_ml,
        )


def teichholz_volume_ml(lv_diameter_cm: float) -> float:
    """Estimate EDV (mL) from LV internal diameter (cm)."""
    if not 1.0 <= float(lv_diameter_cm) <= 10.0:
        raise PatientInputError("LV diameter must be between 1 and 10 cm.")
    return 7.0 * float(lv_diameter_cm) ** 3 / (2.4 + float(lv_diameter_cm))
