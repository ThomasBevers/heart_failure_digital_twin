"""API tests. Run with: pytest tests/test_api.py

conftest.py (project root) puts backend/ and backend/vendor/echonet_dynamic/
on sys.path, so `import api.main` and `import heart_twin` both resolve here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validate_patient_ok():
    response = client.post(
        "/api/patients/validate",
        json={
            "identifier": "HF-001",
            "heart_rate_bpm": 82,
            "systolic_bp_mmhg": 118,
            "diastolic_bp_mmhg": 74,
            "lvef_percent": 32,
            "lv_diameter_cm": 6.1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "teichholz"
    assert abs(body["ejection_fraction_percent"] - 32.0) < 0.01


def test_validate_patient_rejects_bad_pressures():
    response = client.post(
        "/api/patients/validate",
        json={
            "identifier": "HF-002",
            "heart_rate_bpm": 82,
            "systolic_bp_mmhg": 70,
            "diastolic_bp_mmhg": 90,
            "lvef_percent": 32,
        },
    )
    assert response.status_code == 422


def test_simulate_baseline_matches_entered_lvef():
    response = client.post(
        "/api/simulate",
        json={
            "patient": {
                "identifier": "HF-003",
                "heart_rate_bpm": 82,
                "systolic_bp_mmhg": 118,
                "diastolic_bp_mmhg": 74,
                "lvef_percent": 32,
                "lv_diameter_cm": 6.1,
            }
        },
    )
    assert response.status_code == 200
    baseline_ef = response.json()["baseline"]["summary"]["ejection_fraction_percent"]
    assert abs(baseline_ef - 32.0) < 0.5


def test_simulate_reports_unreachable_targets_instead_of_silently_diverging():
    """Regression test for the calibration-convergence bug: a chamber too small
    for the target EF must raise a clear error, not return a mismatched baseline."""
    response = client.post(
        "/api/simulate",
        json={
            "patient": {
                "identifier": "HF-edge",
                "heart_rate_bpm": 80,
                "systolic_bp_mmhg": 120,
                "diastolic_bp_mmhg": 70,
                "lvef_percent": 95,
                "lv_diameter_cm": 1.0,
            }
        },
    )
    assert response.status_code == 422
    assert "cannot be simulated" in response.json()["detail"]


def test_simulate_ignores_inconsistent_independent_esv():
    """Regression test: when EDV and ESV imply a different EF than the entered
    LVEF (e.g. from two independent models), the baseline must still match
    the entered LVEF, not the independently-implied one."""
    response = client.post(
        "/api/simulate",
        json={
            "patient": {
                "identifier": "HF-004",
                "heart_rate_bpm": 80,
                "systolic_bp_mmhg": 120,
                "diastolic_bp_mmhg": 70,
                "lvef_percent": 66.0,
                "edv_ml": 180.0,
                "esv_ml": 74.0,  # implies ~58.9% -- must NOT become the baseline target
            }
        },
    )
    assert response.status_code == 200
    baseline_ef = response.json()["baseline"]["summary"]["ejection_fraction_percent"]
    assert abs(baseline_ef - 66.0) < 0.5


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 64))
    for i in range(20):
        writer.write(np.full((64, 64, 3), (i * 10) % 255, dtype=np.uint8))
    writer.release()
    return video_path


def test_echo_inspect(synthetic_video: Path):
    with open(synthetic_video, "rb") as handle:
        response = client.post("/api/echo/inspect", files={"file": ("clip.avi", handle, "video/avi")}, data={"view": "A4C"})
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["frame_count"] == 20
    assert len(body["preview_png_base64"]) > 0


def test_echo_inspect_rejects_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with open(bad_file, "rb") as handle:
        response = client.post("/api/echo/inspect", files={"file": ("notes.txt", handle, "text/plain")})
    assert response.status_code == 422


def test_echo_infer_reports_missing_checkpoint_cleanly(synthetic_video: Path):
    """Without a downloaded checkpoint, this must be a clear 422, never a 500 crash."""
    with open(synthetic_video, "rb") as handle:
        response = client.post("/api/echo/infer", files={"file": ("clip.avi", handle, "video/avi")})
    assert response.status_code in (200, 422)
    if response.status_code == 422:
        assert "checkpoint" in response.json()["detail"].lower()
