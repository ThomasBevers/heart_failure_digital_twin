"""A closed-loop, zero-dimensional cardiovascular model for research use.

The model contains a time-varying-elastance left ventricle, idealised mitral
and aortic valves, and arterial and venous compliance. Unlike the previous
one-way model, blood volume is conserved and the ventricle refills during
diastole. It is not a validated clinical decision model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CardiovascularParameters:
    """Parameters in mmHg, mL, seconds, and mmHg·s/mL."""

    emax_mmhg_per_ml: float
    emin_mmhg_per_ml: float = 0.06
    unstressed_lv_volume_ml: float = 10.0
    arterial_compliance_ml_per_mmhg: float = 1.5
    venous_compliance_ml_per_mmhg: float = 100.0
    systemic_resistance_mmhg_s_per_ml: float = 1.2
    aortic_valve_resistance_mmhg_s_per_ml: float = 0.01
    mitral_valve_resistance_mmhg_s_per_ml: float = 0.01
    heart_rate_bpm: float = 75.0

    def __post_init__(self) -> None:
        positive = (
            self.emax_mmhg_per_ml,
            self.emin_mmhg_per_ml,
            self.arterial_compliance_ml_per_mmhg,
            self.venous_compliance_ml_per_mmhg,
            self.systemic_resistance_mmhg_s_per_ml,
            self.aortic_valve_resistance_mmhg_s_per_ml,
            self.mitral_valve_resistance_mmhg_s_per_ml,
            self.heart_rate_bpm,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("All cardiovascular parameters must be positive.")
        if self.emax_mmhg_per_ml < self.emin_mmhg_per_ml:
            raise ValueError("Emax must be greater than or equal to Emin.")

    @property
    def cycle_length_s(self) -> float:
        return 60.0 / self.heart_rate_bpm


@dataclass(frozen=True)
class SimulationTrace:
    time_s: tuple[float, ...]
    ventricular_volume_ml: tuple[float, ...]
    ventricular_pressure_mmhg: tuple[float, ...]
    arterial_pressure_mmhg: tuple[float, ...]
    venous_pressure_mmhg: tuple[float, ...]
    aortic_flow_ml_s: tuple[float, ...]
    mitral_flow_ml_s: tuple[float, ...]


class ClosedLoopCardiovascularModel:
    """Closed-loop LV elastance/Windkessel model with forward-Euler integration."""

    def __init__(self, parameters: CardiovascularParameters) -> None:
        self.parameters = parameters

    def _activation(self, time_s: float) -> float:
        """Smooth, unit-height systolic activation over 35% of a cycle."""
        phase = (time_s % self.parameters.cycle_length_s) / self.parameters.cycle_length_s
        if phase >= 0.35:
            return 0.0
        return math.sin(math.pi * phase / 0.35) ** 2

    def simulate(
        self,
        *,
        initial_lv_volume_ml: float,
        initial_arterial_pressure_mmhg: float,
        initial_venous_pressure_mmhg: float = 8.0,
        duration_s: float = 12.0,
        dt_s: float = 0.002,
    ) -> SimulationTrace:
        """Run a volume-conserving simulation and return the complete trace."""
        p = self.parameters
        if initial_lv_volume_ml <= p.unstressed_lv_volume_ml:
            raise ValueError("Initial LV volume must exceed the unstressed LV volume.")
        if initial_arterial_pressure_mmhg <= 0 or initial_venous_pressure_mmhg <= 0:
            raise ValueError("Initial arterial and venous pressures must be positive.")
        if duration_s <= 0 or dt_s <= 0:
            raise ValueError("duration_s and dt_s must be positive.")

        lv_volume = float(initial_lv_volume_ml)
        arterial_volume = initial_arterial_pressure_mmhg * p.arterial_compliance_ml_per_mmhg
        venous_volume = initial_venous_pressure_mmhg * p.venous_compliance_ml_per_mmhg
        steps = int(math.ceil(duration_s / dt_s))
        values = {name: [] for name in (
            "time", "lv_volume", "lv_pressure", "arterial_pressure", "venous_pressure", "aortic_flow", "mitral_flow"
        )}

        for step in range(steps + 1):
            now = min(step * dt_s, duration_s)
            elastance = p.emin_mmhg_per_ml + (p.emax_mmhg_per_ml - p.emin_mmhg_per_ml) * self._activation(now)
            lv_pressure = elastance * max(0.0, lv_volume - p.unstressed_lv_volume_ml)
            arterial_pressure = arterial_volume / p.arterial_compliance_ml_per_mmhg
            venous_pressure = venous_volume / p.venous_compliance_ml_per_mmhg
            aortic_flow = max(0.0, (lv_pressure - arterial_pressure) / p.aortic_valve_resistance_mmhg_s_per_ml)
            mitral_flow = max(0.0, (venous_pressure - lv_pressure) / p.mitral_valve_resistance_mmhg_s_per_ml)
            systemic_flow = max(0.0, (arterial_pressure - venous_pressure) / p.systemic_resistance_mmhg_s_per_ml)

            values["time"].append(now)
            values["lv_volume"].append(lv_volume)
            values["lv_pressure"].append(lv_pressure)
            values["arterial_pressure"].append(arterial_pressure)
            values["venous_pressure"].append(venous_pressure)
            values["aortic_flow"].append(aortic_flow)
            values["mitral_flow"].append(mitral_flow)
            if step == steps:
                break

            lv_volume += (mitral_flow - aortic_flow) * dt_s
            arterial_volume += (aortic_flow - systemic_flow) * dt_s
            venous_volume += (systemic_flow - mitral_flow) * dt_s
            lv_volume = max(p.unstressed_lv_volume_ml + 0.01, lv_volume)

        return SimulationTrace(
            tuple(values["time"]),
            tuple(values["lv_volume"]),
            tuple(values["lv_pressure"]),
            tuple(values["arterial_pressure"]),
            tuple(values["venous_pressure"]),
            tuple(values["aortic_flow"]),
            tuple(values["mitral_flow"]),
        )


def cycle_summary(trace: SimulationTrace, heart_rate_bpm: float) -> tuple[float, float, float, float]:
    """Return EDV, ESV, mean arterial pressure, and EF from the final cycle."""
    cycle_length = 60.0 / heart_rate_bpm
    start_time = max(trace.time_s[0], trace.time_s[-1] - cycle_length)
    start = next((index for index, value in enumerate(trace.time_s) if value >= start_time), 0)
    volumes = trace.ventricular_volume_ml[start:]
    pressures = trace.arterial_pressure_mmhg[start:]
    edv = max(volumes)
    esv = min(volumes)
    mean_pressure = sum(pressures) / len(pressures)
    ef = 100.0 * (edv - esv) / edv
    return edv, esv, mean_pressure, ef


def total_circulating_volume(trace: SimulationTrace, parameters: CardiovascularParameters) -> Sequence[float]:
    """Expose the conserved model volume for tests and numerical diagnostics."""
    return tuple(
        lv + arterial * parameters.arterial_compliance_ml_per_mmhg + venous * parameters.venous_compliance_ml_per_mmhg
        for lv, arterial, venous in zip(
            trace.ventricular_volume_ml,
            trace.arterial_pressure_mmhg,
            trace.venous_pressure_mmhg,
        )
    )
