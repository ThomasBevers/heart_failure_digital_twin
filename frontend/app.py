"""Streamlit interface for the heart-failure digital-twin research prototype."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "echonet_dynamic"))

from backend.echonet_integration import infer_ef_from_video, infer_volumes_from_segmentation  # noqa: E402
from backend.imaging import ImagingError, convert_dicom_to_avi, inspect_video, preview_frame_png  # noqa: E402
from backend.patient import PatientInputError, PatientProfile  # noqa: E402
from backend.twin import ScenarioControls, TwinComparison, run_twin  # noqa: E402


st.set_page_config(page_title="Heart failure digital twin", page_icon=":material/favorite:", layout="wide")
st.session_state.setdefault("patient_profile", None)
st.session_state.setdefault("echo_metadata", None)
st.session_state.setdefault("echo_preview", None)
st.session_state.setdefault("echonet_extracted", False)
st.session_state.setdefault("volume_source", "Manual entry")
st.session_state.setdefault("use_echo_values", False)


def _clear_twin() -> None:
    st.session_state.patient_profile = None
    st.session_state.echo_metadata = None
    st.session_state.echo_preview = None
    st.session_state.echonet_extracted = False
    st.session_state.volume_source = "Manual entry"
    st.session_state.use_echo_values = False


def _required(value: float | None, label: str) -> float:
    if value is None:
        raise PatientInputError(f"{label} is required.")
    return float(value)


@st.cache_data(max_entries=32, show_spinner=False)
def calculate_twin(patient: PatientProfile, controls: ScenarioControls) -> TwinComparison:
    """Cache the deterministic simulation for an unchanged patient scenario."""
    return run_twin(patient, controls)


def _chart_frame(comparison: TwinComparison, heart_rate_bpm: float) -> pd.DataFrame:
    trace = comparison.scenario.trace
    start_time = trace.time_s[-1] - 2.0 * 60.0 / heart_rate_bpm
    rows = [
        {
            "Time (s)": time,
            "LV pressure (mmHg)": lv_pressure,
            "Arterial pressure (mmHg)": arterial_pressure,
            "LV volume (mL)": lv_volume,
        }
        for time, lv_pressure, arterial_pressure, lv_volume in zip(
            trace.time_s,
            trace.ventricular_pressure_mmhg,
            trace.arterial_pressure_mmhg,
            trace.ventricular_volume_ml,
        )
        if time >= start_time
    ]
    return pd.DataFrame(rows)


def _run_echonet_extraction(uploaded_file, view: str) -> tuple[dict[str, object], bytes]:
    """Run real EF/volume inference directly on the uploaded video/DICOM.

    No frame-extraction pipeline, no fabricated fallback numbers: a missing
    checkpoint or unreadable video surfaces as `echonet_error`, and EF/EDV/ESV
    stay None rather than showing a plausible-looking but meaningless value.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="heart_twin_upload_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        source = tmpdir_path / f"upload{suffix}"
        source.write_bytes(uploaded_file.getvalue())
        try:
            if suffix in {".dcm", ".dicom"}:
                model_input = tmpdir_path / "converted_echo.avi"
                convert_dicom_to_avi(source, model_input, view=view)
            else:
                model_input = source
            ef = infer_ef_from_video(model_input)
            edv, esv = infer_volumes_from_segmentation(model_input)
            preview = preview_frame_png(model_input)
        except ImagingError as error:
            return (
                {
                    "echo_ef_percent": None,
                    "echo_edv_ml": None,
                    "echo_esv_ml": None,
                    "echonet_error": str(error),
                },
                b"",
            )
        return {"echo_ef_percent": ef, "echo_edv_ml": edv, "echo_esv_ml": esv}, preview


def _inspect_upload(uploaded_file, view: str) -> tuple[dict[str, object], bytes]:
    """Technical metadata + preview only. Deliberately runs no EF/volume inference."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="heart_twin_upload_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        source = tmpdir_path / f"upload{suffix}"
        source.write_bytes(uploaded_file.getvalue())
        if suffix in {".dcm", ".dicom"}:
            model_input = tmpdir_path / "converted_echo.avi"
            metadata = convert_dicom_to_avi(source, model_input, view=view)
        else:
            model_input = source
            metadata = inspect_video(model_input, view=view)
        preview = preview_frame_png(model_input)
        return asdict(metadata), preview


st.title("Heart failure digital twin")
st.caption("Patient-specific, closed-loop cardiovascular simulation for research and education.")
st.warning(
    "Research prototype only. This model is not clinically validated and must not be used for diagnosis, treatment, or patient management.",
    icon=":material/science:",
)

# -------------------------
# Sidebar: volume source selector (outside the form)
# -------------------------
with st.sidebar:
    st.header("Volume source")
    st.caption("Choose whether to enter EF/EDV/ESV manually or extract them from an uploaded A4C video.")

    source_choice = st.radio(
        "Source of EF/EDV/ESV",
        ["Manual entry", "EchoNet (from uploaded video)"],
        index=0 if st.session_state.volume_source == "Manual entry" else 1,
        key="volume_source_radio",
    )
    st.session_state.volume_source = source_choice

    if source_choice == "EchoNet (from uploaded video)":
        st.info("EchoNet mode: upload an A4C echo video and click Run EchoNet extraction.")
        uploaded_echo_for_echonet = st.file_uploader(
            "Upload A4C echo video for EchoNet",
            type=["dcm", "dicom", "avi", "mp4", "mov"],
            key="echo_file_echonet",
        )

        if st.button("Run EchoNet extraction", key="run_echonet_extraction"):
            if uploaded_echo_for_echonet is None:
                st.error("Please upload a video before running EchoNet extraction.")
            else:
                with st.spinner("Running EchoNet inference..."):
                    metadata_dict, preview_bytes = _run_echonet_extraction(uploaded_echo_for_echonet, "A4C")
                st.session_state.echo_metadata = metadata_dict
                st.session_state.echo_preview = preview_bytes
                st.session_state.echonet_extracted = True
                if metadata_dict.get("echo_ef_percent") is None:
                    st.warning(f"EchoNet could not produce a result: {metadata_dict.get('echonet_error', 'unknown error')}")
                else:
                    st.success("EchoNet extraction completed and values stored.")

        if st.session_state.echo_metadata:
            em = st.session_state.echo_metadata
            st.markdown("**EchoNet extracted values**")
            ef, edv, esv = em.get("echo_ef_percent"), em.get("echo_edv_ml"), em.get("echo_esv_ml")
            st.write(f"- **LVEF:** {ef:.1f}%" if ef is not None else "- **LVEF:** not available")
            st.write(f"- **EDV:** {edv:.1f} mL" if edv is not None else "- **EDV:** not available")
            st.write(f"- **ESV:** {esv:.1f} mL" if esv is not None else "- **ESV:** not available")

            if st.session_state.echo_preview:
                st.image(st.session_state.echo_preview, caption="Echo preview", use_container_width=True)

            with st.expander("Extraction diagnostics"):
                st.write(em)

            st.checkbox("Use these EchoNet values for the twin", value=False, key="use_echo_values")

# -------------------------
# Sidebar: patient form
# -------------------------
with st.sidebar:
    with st.form("patient_measurements", clear_on_submit=False):
        identifier = st.text_input("Patient ID", key="patient_id")
        age_band = st.selectbox("Age band", ["Not provided", "≤49", "50–59", "60–69", "70–79", "≥80"], key="age_band")
        sex = st.selectbox("Sex", ["Not provided", "Female", "Male"], key="sex")
        heart_rate = st.number_input("Heart rate (bpm)", min_value=20.0, max_value=240.0, value=None, key="heart_rate")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=40.0, max_value=300.0, value=None, key="systolic_bp")
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=20.0, max_value=200.0, value=None, key="diastolic_bp")

        if st.session_state.volume_source == "Manual entry":
            lvef = st.number_input("LVEF (%)", min_value=1.0, max_value=100.0, value=None, key="lvef")
            edv = st.number_input("EDV (mL, optional)", min_value=1.0, max_value=1_000.0, value=None, key="edv")
            esv = st.number_input("ESV (mL, optional)", min_value=0.0, max_value=1_000.0, value=None, key="esv")
        else:
            lvef = edv = esv = None
            st.caption("LVEF/EDV/ESV will come from the EchoNet extraction above.")

        lv_diameter = st.number_input("LV end-diastolic diameter (cm, optional)", min_value=1.0, max_value=10.0, value=None, key="lv_diameter")
        nyha = st.selectbox("NYHA class", ["Not provided", "I", "II", "III", "IV"], key="nyha")
        submitted = st.form_submit_button("Create or update twin", type="primary", icon=":material/person_add:")

    if submitted:
        if st.session_state.volume_source == "EchoNet (from uploaded video)":
            if not st.session_state.echonet_extracted:
                st.error("Run EchoNet extraction first.")
                st.stop()
            if not st.session_state.use_echo_values:
                st.error("Check 'Use these EchoNet values for the twin' to confirm using extracted values.")
                st.stop()
            echo_meta = st.session_state.echo_metadata or {}
            final_lvef = echo_meta.get("echo_ef_percent")
            final_edv = echo_meta.get("echo_edv_ml")
            final_esv = echo_meta.get("echo_esv_ml")
            if final_lvef is None:
                st.error("EchoNet EF is not available. Check the diagnostics above or switch to Manual entry.")
                st.stop()
        else:
            final_lvef, final_edv, final_esv = lvef, edv, esv

        try:
            st.session_state.patient_profile = PatientProfile(
                identifier=identifier,
                heart_rate_bpm=_required(heart_rate, "Heart rate"),
                systolic_bp_mmhg=_required(systolic_bp, "Systolic BP"),
                diastolic_bp_mmhg=_required(diastolic_bp, "Diastolic BP"),
                lvef_percent=_required(final_lvef, "LVEF"),
                edv_ml=float(final_edv) if final_edv is not None else None,
                esv_ml=float(final_esv) if final_esv is not None else None,
                lv_diameter_cm=float(lv_diameter) if lv_diameter is not None else None,
                age_band=age_band,
                sex=sex,
                nyha_class=nyha,
            )
        except PatientInputError as error:
            st.error(str(error))
        else:
            st.success("Twin baseline updated.")
    st.button("Clear current twin", on_click=_clear_twin, icon=":material/delete_sweep:")

patient = st.session_state.patient_profile
if patient is None:
    st.info("Create a patient twin from the sidebar to begin a simulation.", icon=":material/arrow_back:")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("What-if scenario")
    st.caption("These perturbations affect the virtual model only; they do not modify source measurements.")
    contractility = st.slider("Contractility change", -50, 50, 0, 5, format="%d%%", key="contractility")
    preload = st.slider("Preload change", -40, 40, 0, 5, format="%d%%", key="preload")
    afterload = st.slider("Afterload change", -40, 40, 0, 5, format="%d%%", key="afterload")
    heart_rate_delta = st.slider("Heart-rate change", -30, 30, 0, 5, format="%+d bpm", key="heart_rate_delta")

controls = ScenarioControls(contractility, preload, afterload, heart_rate_delta)
with st.spinner("Calibrating baseline physiology and running the scenario..."):
    comparison = calculate_twin(patient, controls)
baseline = comparison.baseline.summary
scenario = comparison.scenario.summary

st.subheader(f"Patient {patient.identifier}")
with st.container(horizontal=True):
    st.metric("Entered LVEF", f"{patient.lvef_percent:.1f}%", border=True)
    st.metric("Heart rate", f"{patient.heart_rate_bpm:.0f} bpm", border=True)
    st.metric("Blood pressure", f"{patient.systolic_bp_mmhg:.0f}/{patient.diastolic_bp_mmhg:.0f} mmHg", border=True)
    st.metric("NYHA class", patient.nyha_class, border=True)
    st.metric("Volume input", comparison.volume_input.source.capitalize(), border=True)
st.caption(comparison.volume_input.note)

st.subheader("Virtual cardiovascular state")
with st.container(horizontal=True):
    st.metric("Ejection fraction", f"{scenario.ejection_fraction_percent:.1f}%", f"{scenario.ejection_fraction_percent - baseline.ejection_fraction_percent:+.1f} pp", border=True)
    st.metric("Stroke volume", f"{scenario.stroke_volume_ml:.1f} mL", f"{scenario.stroke_volume_ml - baseline.stroke_volume_ml:+.1f} mL", border=True)
    st.metric("Cardiac output", f"{scenario.cardiac_output_l_min:.2f} L/min", f"{scenario.cardiac_output_l_min - baseline.cardiac_output_l_min:+.2f} L/min", border=True)
    st.metric("Mean arterial pressure", f"{scenario.mean_arterial_pressure_mmhg:.1f} mmHg", f"{scenario.mean_arterial_pressure_mmhg - baseline.mean_arterial_pressure_mmhg:+.1f} mmHg", border=True)

chart_data = _chart_frame(comparison, patient.heart_rate_bpm + heart_rate_delta)
pressure_chart, volume_chart = st.columns(2)
with pressure_chart:
    with st.container(border=True):
        st.markdown("**Pressure over the final two cycles**")
        st.line_chart(chart_data, x="Time (s)", y=["LV pressure (mmHg)", "Arterial pressure (mmHg)"])
with volume_chart:
    with st.container(border=True):
        st.markdown("**LV volume over the final two cycles**")
        st.line_chart(chart_data, x="Time (s)", y="LV volume (mL)")

st.subheader("Baseline and scenario")
st.dataframe(
    pd.DataFrame(
        {
            "Metric": ["Ejection fraction (%)", "EDV (mL)", "ESV (mL)", "Stroke volume (mL)", "Cardiac output (L/min)", "MAP (mmHg)", "Emax (mmHg/mL)"],
            "Baseline": [baseline.ejection_fraction_percent, baseline.end_diastolic_volume_ml, baseline.end_systolic_volume_ml, baseline.stroke_volume_ml, baseline.cardiac_output_l_min, baseline.mean_arterial_pressure_mmhg, baseline.emax_mmhg_per_ml],
            "Scenario": [scenario.ejection_fraction_percent, scenario.end_diastolic_volume_ml, scenario.end_systolic_volume_ml, scenario.stroke_volume_ml, scenario.cardiac_output_l_min, scenario.mean_arterial_pressure_mmhg, scenario.emax_mmhg_per_ml],
        }
    ),
    hide_index=True,
)

with st.expander("Model scope and interpretation"):
    st.write(
        "The baseline is calibrated (by solving for the contractility, Emax, that reproduces the "
        "entered LVEF) to the entered heart rate, blood pressure, and LVEF. The virtual heart is a "
        "closed-loop model: a time-varying-elastance left ventricle, idealised forward-only mitral "
        "and aortic valves, and arterial + venous compliance, so blood volume is conserved across "
        "cycles. What-if sliders perturb this fitted baseline; they do not re-fit it from new data."
    )


st.subheader("Echocardiography technical check")
st.caption("Upload only research-approved, de-identified files. This inspector reports technical metadata and a preview only; it does not run EF/volume inference.")
with st.form("echo_input", border=True):
    echo_view = st.selectbox("Confirmed view", ["A4C", "Other", "Unknown"], key="echo_view")
    uploaded_echo = st.file_uploader("Echo study (technical inspect)", type=["dcm", "dicom", "avi", "mp4", "mov"], key="echo_file")
    inspect_echo = st.form_submit_button("Inspect study", icon=":material/visibility:")

if inspect_echo:
    if uploaded_echo is None:
        st.error("Choose an echocardiography file before inspecting it.")
    else:
        try:
            metadata_dict, preview_bytes = _inspect_upload(uploaded_echo, echo_view)
        except ImagingError as error:
            st.error(f"Could not inspect the echo study: {error}")
        else:
            st.session_state.echo_metadata = metadata_dict
            st.session_state.echo_preview = preview_bytes
            st.success("Echo study inspected (technical only; no EF/volume inference was run).")

if st.session_state.echo_metadata is not None:
    metadata = st.session_state.echo_metadata
    details, preview = st.columns([1, 1])
    with details:
        st.dataframe(pd.DataFrame([metadata]), hide_index=True)
        st.caption(str(metadata.get("provenance", "")))
    with preview:
        if st.session_state.echo_preview:
            st.image(st.session_state.echo_preview, caption="Centre-frame technical preview", use_container_width=True)
        else:
            st.info("No preview available.")
