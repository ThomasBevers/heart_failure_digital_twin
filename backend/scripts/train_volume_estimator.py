"""Train the EDV/ESV volume estimator used by heart_twin.echonet_integration.

Self-contained: uses the same video_features(...) the app calls at inference
time, so training and inference can never silently drift apart. Split sizes
are always configurable, so a small subset can be used for a quick local run.

Usage:
    python scripts/train_volume_estimator.py
    python scripts/train_volume_estimator.py --n-train 50 --n-val 20 --n-test 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.heart_twin.echonet_integration import VOLUME_FEATURE_NAMES, video_features  # noqa: E402


def _rows(dataset_dir: Path, split: str, *, n_patients: Optional[int], seed: int) -> list[dict]:
    with (dataset_dir / "FileList.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["Split"].upper() == split.upper()]
    if n_patients is not None and n_patients < len(rows):
        rows = random.Random(seed).sample(rows, n_patients)
    return rows


def _extract(dataset_dir: Path, rows: list[dict]):
    import numpy as np

    features, edv, esv = [], [], []
    for row in rows:
        video = dataset_dir / "Videos" / f"{row['FileName']}.avi"
        if not video.exists():
            continue
        features.append(video_features(video))
        edv.append(float(row["EDV"]))
        esv.append(float(row["ESV"]))
    return np.asarray(features), np.asarray(edv), np.asarray(esv)


def _evaluate(model_edv, model_esv, features, actual_edv, actual_esv) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    predicted_edv = model_edv.predict(features)
    predicted_esv = model_esv.predict(features)
    return {
        "records": float(len(actual_edv)),
        "edv_mae_ml": float(mean_absolute_error(actual_edv, predicted_edv)),
        "edv_rmse_ml": float(mean_squared_error(actual_edv, predicted_edv) ** 0.5),
        "edv_r2": float(r2_score(actual_edv, predicted_edv)) if len(actual_edv) > 1 else float("nan"),
        "esv_mae_ml": float(mean_absolute_error(actual_esv, predicted_esv)),
        "esv_rmse_ml": float(mean_squared_error(actual_esv, predicted_esv) ** 0.5),
        "esv_r2": float(r2_score(actual_esv, predicted_esv)) if len(actual_esv) > 1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "EchoNet-Dynamic")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "echonet_volume_estimator.joblib")
    parser.add_argument("--n-train", type=int, default=None, help="TRAIN patients to use (default: all).")
    parser.add_argument("--n-val", type=int, default=None, help="VAL patients to evaluate on (default: all).")
    parser.add_argument("--n-test", type=int, default=None, help="TEST patients to evaluate on (default: all).")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    train_rows = _rows(args.dataset_dir, "TRAIN", n_patients=args.n_train, seed=args.seed)
    train_features, train_edv, train_esv = _extract(args.dataset_dir, train_rows)
    if len(train_features) < 10:
        raise SystemExit(
            f"At least 10 readable TRAIN videos are required, found {len(train_features)}. "
            "Increase --n-train or check --dataset-dir."
        )

    model_edv = ExtraTreesRegressor(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1).fit(train_features, train_edv)
    model_esv = ExtraTreesRegressor(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1).fit(train_features, train_esv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_names": VOLUME_FEATURE_NAMES, "edv_model": model_edv, "esv_model": model_esv}, args.output)

    report = {"output": str(args.output), "train_records": len(train_features)}
    if not args.skip_val:
        val_rows = _rows(args.dataset_dir, "VAL", n_patients=args.n_val, seed=args.seed)
        val_features, val_edv, val_esv = _extract(args.dataset_dir, val_rows)
        if len(val_features):
            report["validation"] = _evaluate(model_edv, model_esv, val_features, val_edv, val_esv)
    if not args.skip_test:
        test_rows = _rows(args.dataset_dir, "TEST", n_patients=args.n_test, seed=args.seed)
        test_features, test_edv, test_esv = _extract(args.dataset_dir, test_rows)
        if len(test_features):
            report["test"] = _evaluate(model_edv, model_esv, test_features, test_edv, test_esv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
