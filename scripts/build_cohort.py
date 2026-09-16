"""Build a clean Zigong cohort artifact without modifying raw PhysioNet files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_twin.cohort import load_zigong_cohort, summarize_cohort, write_cohort_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Zigong cohort for offline research.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "zigong")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "zigong_cohort_clean.csv")
    args = parser.parse_args()
    records = load_zigong_cohort(args.data_dir)
    write_cohort_csv(records, args.output)
    summary = summarize_cohort(records)
    print(f"Wrote {summary.records} records ({summary.unique_patients} unique patients) to {args.output}")
    print(f"LVEF present: {summary.records_with_lvef}; LV diameter present: {summary.records_with_lv_diameter}")


if __name__ == "__main__":
    main()
