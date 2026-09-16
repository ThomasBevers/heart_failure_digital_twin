"""Reproducible preparation of the Zigong cohort for offline research.

The raw PhysioNet files are not modified. This module does not create a
clinical risk score: outcome labels need a preregistered held-out validation
protocol before such a model belongs in the application.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .patient import teichholz_volume_ml


MISSING_VALUES = {"", "na", "n/a", "nan", "null", "none"}


class CohortDataError(RuntimeError):
    """Raised when a required raw cohort input cannot be read."""


@dataclass(frozen=True)
class CohortSummary:
    records: int
    unique_patients: int
    records_with_lvef: int
    records_with_lv_diameter: int


def _clean_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return None if value.casefold() in MISSING_VALUES else value


def _as_number(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def load_zigong_cohort(data_dir: Path) -> list[dict[str, object]]:
    """Load one cleaned record per inpatient admission from ``dat.csv``."""
    path = Path(data_dir) / "dat.csv"
    if not path.is_file():
        raise CohortDataError(f"Expected Zigong cohort file was not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records: list[dict[str, object]] = []
        for row in reader:
            clean = {
                key.strip().lstrip("\ufeff"): _clean_value(value)
                for key, value in row.items()
                if key and key.strip()
            }
            identifier = clean.get("inpatient.number")
            if not identifier:
                continue
            lvef = _as_number(clean.get("LVEF"))
            diameter = _as_number(clean.get("left.ventricular.end.diastolic.diameter.LV"))
            edv = teichholz_volume_ml(diameter) if diameter is not None and 1.0 <= diameter <= 10.0 else None
            clean["estimated_edv_ml"] = edv
            clean["estimated_esv_ml"] = edv * (1.0 - lvef / 100.0) if edv is not None and lvef is not None and 0 <= lvef <= 100 else None
            records.append(clean)
    if not records:
        raise CohortDataError(f"No patient records were found in {path}")
    return records


def summarize_cohort(records: Sequence[dict[str, object]]) -> CohortSummary:
    identifiers = {str(record["inpatient.number"]) for record in records if record.get("inpatient.number")}
    return CohortSummary(
        records=len(records),
        unique_patients=len(identifiers),
        records_with_lvef=sum(_as_number(record.get("LVEF")) is not None for record in records),
        records_with_lv_diameter=sum(
            _as_number(record.get("left.ventricular.end.diastolic.diameter.LV")) is not None
            for record in records
        ),
    )


def write_cohort_csv(records: Sequence[dict[str, object]], output_path: Path) -> None:
    """Write a clean, derived research artifact without modifying raw data."""
    if not records:
        raise ValueError("Cannot write an empty cohort.")
    fields = list(dict.fromkeys(field for record in records for field in record))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
