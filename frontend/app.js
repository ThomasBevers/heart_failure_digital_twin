/* CardioTwin research workspace — framework-free frontend for the supplied FastAPI API. */

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const defaultApiBase = window.location.protocol === "file:"
  ? "http://localhost:8000/api"
  : `${window.location.origin}/api`;

const seedRecords = [
  {
    id: "HF-042", ageBand: "60–69", sex: "Female", nyha: "II", heartRate: 76, systolicBp: 126,
    diastolicBp: 78, lvef: 38, edv: 162, esv: 100, lvDiameter: 5.4, source: "Manual entry", edited: "Research example"
  },
  {
    id: "HF-117", ageBand: "70–79", sex: "Male", nyha: "III", heartRate: 68, systolicBp: 118,
    diastolicBp: 70, lvef: 31, edv: 188, esv: 130, lvDiameter: 6.1, source: "Manual entry", edited: "Research example"
  },
  {
    id: "HF-208", ageBand: "50–59", sex: "Female", nyha: "II", heartRate: 82, systolicBp: 132,
    diastolicBp: 84, lvef: 45, edv: 142, esv: 78, lvDiameter: 5.0, source: "Manual entry", edited: "Research example"
  }
];

const state = {
  apiBase: localStorage.getItem("cardioTwinApiBase") || defaultApiBase,
  source: "manual",
  echo: { file: null, result: null, confirmed: false },
  selectedId: null,
  lastResult: null,
  records: loadRecords(),
  isApiOnline: false,
  health: null
};

function loadRecords() {
  try {
    const saved = JSON.parse(sessionStorage.getItem("cardioTwinRecords"));
    return Array.isArray(saved) && saved.length ? saved : seedRecords;
  } catch {
    return seedRecords;
  }
}

function persistRecords() {
  sessionStorage.setItem("cardioTwinRecords", JSON.stringify(state.records));
}

function numberValue(id, { optional = false } = {}) {
  const value = $(id).value.trim();
  if (optional && !value) return null;
  return Number(value);
}

function controls() {
  return {
    contractility_percent: Number($("#contractility").value),
    preload_percent: Number($("#preload").value),
    afterload_percent: Number($("#afterload").value),
    heart_rate_delta_bpm: Number($("#heartRateDelta").value)
  };
}

function messageFromError(error) {
  if (Array.isArray(error?.detail)) return error.detail.map((item) => item.msg || "Invalid value.").join(" ");
  return error?.detail || error?.message || "The request could not be completed.";
}

async function apiRequest(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(`${state.apiBase}${path}`, { ...options, headers });
  } catch {
    throw new Error("Could not reach the simulation service. Check its address and that it is running.");
  }
  let data = null;
  try { data = await response.json(); } catch { /* an empty body is still a response */ }
  if (!response.ok) throw new Error(messageFromError(data) || `Request failed (${response.status}).`);
  return data;
}

function updateApiUi(status, caption) {
  const dot = $("#apiDot");
  dot.className = `status-dot ${status}`;
  $("#apiStatus").textContent = status === "online" ? "Simulation service online" : status === "offline" ? "Simulation service offline" : "Checking API";
  $("#apiCaption").textContent = caption;
  state.isApiOnline = status === "online";
}

function isEchoNetAvailable() {
  return state.isApiOnline && state.health?.ef_checkpoint_available === true;
}

async function checkConnection({ quiet = false } = {}) {
  updateApiUi("checking", "Looking for the simulation service.");
  try {
    const health = await apiRequest("/health");
    state.health = health;
    const echoReady = health.ef_checkpoint_available ? "EchoNet ready" : "EchoNet weights unavailable";
    updateApiUi("online", `${echoReady} · API v1.0`);
    renderEchoResult();
    if (!quiet) setSettingsFeedback(`Connected. ${echoReady}.`, false);
    return true;
  } catch (error) {
    state.health = null;
    updateApiUi("offline", "Set an API URL, then test again.");
    if (!quiet) setSettingsFeedback(error.message, true);
    return false;
  }
}

function setSettingsFeedback(text, isError) {
  const feedback = $("#settingsFeedback");
  feedback.textContent = text;
  feedback.style.color = isError ? "#b14a40" : "#087d75";
}

function showToast(text, type = "success") {
  const toast = $("#toast");
  toast.textContent = text;
  toast.className = `toast ${type} show`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.className = "toast"; }, 4200);
}

function showFormError(text) {
  const error = $("#formError");
  error.textContent = text;
  error.hidden = !text;
}

function switchView(view) {
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => {
    const active = section.id === `${view}View`;
    section.hidden = !active;
    section.classList.toggle("active-view", active);
  });
  $(".sidebar").classList.remove("open");
  if (view === "patients") renderPatientList();
  if (view === "workspace") window.scrollTo({ top: 0, behavior: "smooth" });
}

function markDraft() {
  if (state.lastResult) clearRunDisplay();
  $("#draftState").className = "state-badge draft";
  $("#draftState").textContent = "NOT SAVED";
  $("#patientFormTitle").textContent = state.selectedId ? `${state.selectedId} · edited draft` : "New patient draft";
  $("#crumbPatient").textContent = state.selectedId ? `${state.selectedId} · draft` : "New patient draft";
  renderPatientOverview();
}

function setFormSaved(id) {
  $("#draftState").className = "state-badge saved";
  $("#draftState").textContent = "SAVED";
  $("#patientFormTitle").textContent = id;
  $("#crumbPatient").textContent = id;
  renderPatientOverview();
}

function preparePatientEditor() {
  const dialog = $("#patientDialog");
  const panel = $(".patient-panel");
  if (!dialog.contains(panel)) dialog.append(panel);
}

function openPatientEditor() {
  preparePatientEditor();
  const dialog = $("#patientDialog");
  if (!dialog.open) dialog.showModal();
}

function closePatientEditor() {
  const dialog = $("#patientDialog");
  if (dialog.open) dialog.close();
}

function renderPatientOverview() {
  const identifier = $("#identifier")?.value.trim() || "";
  const hasPatient = Boolean(identifier || state.selectedId);
  $("#patientOverviewEmpty").hidden = hasPatient;
  $("#patientOverviewDetails").hidden = !hasPatient;
  if (!hasPatient) {
    $("#overviewTitle").textContent = "No patient selected";
    $("#overviewState").className = "state-badge waiting";
    $("#overviewState").textContent = "AWAITING RECORD";
    return;
  }
  const lvef = state.source === "echonet" ? state.echo.result?.ef_percent : $("#lvef").value;
  const source = state.source === "echonet" ? "EchoNet extraction" : "Manual entry";
  $("#overviewTitle").textContent = identifier || state.selectedId;
  $("#overviewMeta").textContent = `${$("#ageBand").value} · ${$("#sex").value}`;
  $("#overviewEf").textContent = lvef ? `${formatNumber(lvef, 1)}%` : "—";
  $("#overviewHr").textContent = $("#heartRate").value ? `${$("#heartRate").value} bpm` : "—";
  const systolic = $("#systolicBp").value, diastolic = $("#diastolicBp").value;
  $("#overviewBp").textContent = systolic && diastolic ? `${systolic}/${diastolic}` : "—";
  $("#overviewNyha").textContent = $("#nyha").value === "Not provided" ? "—" : $("#nyha").value;
  $("#overviewSource").textContent = source;
  const saved = Boolean(state.selectedId && $("#draftState").textContent === "SAVED");
  $("#overviewState").className = `state-badge ${saved ? "saved" : "draft"}`;
  $("#overviewState").textContent = saved ? "READY TO RUN" : "DRAFT";
}

function changeSource(nextSource, { preserveValues = true } = {}) {
  state.source = nextSource;
  const manual = nextSource === "manual";
  $$(".source-tab").forEach((button) => {
    const active = button.dataset.source === nextSource;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $("#manualFields").hidden = !manual;
  $("#echoNetFields").hidden = manual;
  $("#lvef").required = manual;
  if (!preserveValues && !manual) {
    state.echo = { file: null, result: null, confirmed: false };
    renderEchoResult();
  }
  updateSourceSummary();
}

function updateSourceSummary() {
  const label = state.source === "manual"
    ? "Manual-entry measurements are the active source."
    : state.health && !state.health.ef_checkpoint_available
      ? "EchoNet cannot run yet: the required EF model checkpoint is not installed on the server."
    : state.echo.result && state.echo.confirmed
      ? "Confirmed EchoNet values are the active source."
      : "EchoNet values must be extracted and explicitly confirmed before they can run the twin.";
  $("#sourceSummary").textContent = label;
}

function renderEchoResult() {
  const resultEl = $("#echoResult");
  const run = $("#runEchoNet");
  const useExisting = $("#echoUseValues");
  if (useExisting) useExisting.closest(".use-echo-row")?.remove();
  run.disabled = !state.echo.file || !isEchoNetAvailable();
  if (!state.echo.result) {
    resultEl.hidden = true;
    if (state.health && !state.health.ef_checkpoint_available) {
      $("#echoRunState").textContent = "EchoNet is unavailable: install the server EF checkpoint, then reconnect.";
    }
    updateSourceSummary();
    return;
  }
  const result = state.echo.result;
  const preview = result.preview_png_base64 ? `<img alt="Echo centre frame preview" src="data:image/png;base64,${result.preview_png_base64}" />` : "";
  const edv = result.edv_ml == null ? "not available" : `${formatNumber(result.edv_ml, 1)} mL`;
  const esv = result.esv_ml == null ? "not available" : `${formatNumber(result.esv_ml, 1)} mL`;
  resultEl.innerHTML = `${preview}<div><h4>Extraction completed</h4><div class="echo-values"><span class="echo-value"><b>LVEF</b> ${formatNumber(result.ef_percent, 1)}%</span><span class="echo-value"><b>EDV</b> ${edv}</span><span class="echo-value"><b>ESV</b> ${esv}</span></div><label class="use-echo-row"><input id="echoUseValues" type="checkbox" ${state.echo.confirmed ? "checked" : ""} /> I confirm these values should be used for this twin</label></div>`;
  resultEl.hidden = false;
  $("#echoUseValues").addEventListener("change", (event) => {
    state.echo.confirmed = event.target.checked;
    markDraft();
    updateSourceSummary();
  });
  updateSourceSummary();
}

function setEchoFile(file) {
  state.echo.file = file || null;
  state.echo.result = null;
  state.echo.confirmed = false;
  $("#uploadLabel").textContent = file ? file.name : "Drop an echo study here";
  $("#uploadHint").textContent = file ? `${formatFileSize(file.size)} · Ready for explicit inference` : "DICOM, AVI, MP4 or MOV · max 200 MB";
  renderEchoResult();
  markDraft();
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function runEchoNet() {
  if (!state.echo.file) return;
  if (!state.isApiOnline && !(await checkConnection({ quiet: true }))) {
    showToast("The simulation service is not connected.", "error");
    return;
  }
  if (!isEchoNetAvailable()) {
    showFormError("EchoNet cannot run because its EF checkpoint is not installed on the server. Install it, then test the connection again.");
    return;
  }
  const button = $("#runEchoNet");
  button.disabled = true;
  button.innerHTML = "Running inference…";
  $("#echoRunState").textContent = "Uploading the de-identified study and running EchoNet.";
  try {
    const body = new FormData();
    body.append("file", state.echo.file);
    body.append("view", "A4C");
    state.echo.result = await apiRequest("/echo/infer", { method: "POST", body });
    state.echo.confirmed = false;
    $("#echoRunState").textContent = "Review the returned values and confirm their use below.";
    renderEchoResult();
    showToast("EchoNet extraction completed.");
  } catch (error) {
    $("#echoRunState").textContent = "Extraction did not complete.";
    showFormError(`EchoNet extraction: ${error.message}`);
  } finally {
    button.innerHTML = "Run EchoNet extraction <span>→</span>";
    button.disabled = !state.echo.file || !isEchoNetAvailable();
  }
}

function buildPatientPayload() {
  const identifier = $("#identifier").value.trim();
  const common = {
    identifier,
    heart_rate_bpm: numberValue("#heartRate"),
    systolic_bp_mmhg: numberValue("#systolicBp"),
    diastolic_bp_mmhg: numberValue("#diastolicBp"),
    lv_diameter_cm: numberValue("#lvDiameter", { optional: true }),
    age_band: $("#ageBand").value,
    sex: $("#sex").value,
    nyha_class: $("#nyha").value
  };
  if (!identifier) throw new Error("Patient / study ID is required.");
  for (const [label, value] of [["Heart rate", common.heart_rate_bpm], ["Systolic BP", common.systolic_bp_mmhg], ["Diastolic BP", common.diastolic_bp_mmhg]]) {
    if (!Number.isFinite(value)) throw new Error(`${label} is required.`);
  }
  if (common.systolic_bp_mmhg <= common.diastolic_bp_mmhg) throw new Error("Systolic BP must be greater than diastolic BP.");
  if (state.source === "echonet") {
    if (!state.echo.result) throw new Error("Run EchoNet extraction first, or switch to Manual entry.");
    if (!state.echo.confirmed) throw new Error("Confirm that the extracted EchoNet values should be used for this twin.");
    return { ...common, lvef_percent: state.echo.result.ef_percent, edv_ml: state.echo.result.edv_ml, esv_ml: state.echo.result.esv_ml };
  }
  const lvef = numberValue("#lvef");
  const edv = numberValue("#edv", { optional: true });
  const esv = numberValue("#esv", { optional: true });
  if (!Number.isFinite(lvef)) throw new Error("LVEF is required.");
  if ((edv === null) !== (esv === null)) throw new Error("Provide both EDV and ESV, or neither.");
  return { ...common, lvef_percent: lvef, edv_ml: edv, esv_ml: esv };
}

function buildRecord(payload) {
  return {
    id: payload.identifier, ageBand: payload.age_band, sex: payload.sex, nyha: payload.nyha_class,
    heartRate: payload.heart_rate_bpm, systolicBp: payload.systolic_bp_mmhg, diastolicBp: payload.diastolic_bp_mmhg,
    lvef: payload.lvef_percent, edv: payload.edv_ml, esv: payload.esv_ml, lvDiameter: payload.lv_diameter_cm,
    source: state.source === "echonet" ? "EchoNet extraction" : "Manual entry",
    echo: state.source === "echonet" ? { result: state.echo.result, confirmed: state.echo.confirmed } : null,
    edited: "Just now"
  };
}

function saveRecord(payload) {
  const record = buildRecord(payload);
  const existing = state.records.findIndex((entry) => entry.id.toLowerCase() === record.id.toLowerCase());
  if (existing >= 0) state.records[existing] = record;
  else state.records.unshift(record);
  state.selectedId = record.id;
  persistRecords();
  setFormSaved(record.id);
  renderPatientList();
}

async function validateInputs({ announce = true } = {}) {
  showFormError("");
  let payload;
  try { payload = buildPatientPayload(); } catch (error) {
    showFormError(error.message);
    return null;
  }
  if (!state.isApiOnline && !(await checkConnection({ quiet: true }))) {
    showFormError("The simulation service is offline. Start the FastAPI backend or set its address in Connection settings.");
    return null;
  }
  const button = $("#validateBtn");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Validating…";
  try {
    const volumes = await apiRequest("/patients/validate", { method: "POST", body: JSON.stringify(payload) });
    $(".step:nth-child(3)")?.classList.add("active");
    $("#sourceSummary").textContent = `${volumes.source === "reported" ? "Reported" : "Resolved"} volumes: EDV ${formatNumber(volumes.edv_ml, 1)} mL · ESV ${formatNumber(volumes.esv_ml, 1)} mL.`;
    if (announce) showToast("Inputs validated. The twin is ready to run.");
    return payload;
  } catch (error) {
    showFormError(error.message);
    return null;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function saveAndRun(event) {
  if (event) event.preventDefault();
  const payload = await validateInputs({ announce: false });
  if (!payload) return;
  saveRecord(payload);
  if (await simulate(payload)) closePatientEditor();
}

async function simulate(existingPayload = null) {
  let payload = existingPayload;
  if (!payload) {
    try { payload = buildPatientPayload(); } catch (error) { showFormError(error.message); return; }
  }
  const button = $("#saveAndRunBtn");
  const rerun = $("#rerunScenario");
  button.disabled = true;
  rerun.disabled = true;
  button.innerHTML = "Calibrating twin…";
  $("#resultState").className = "state-badge running";
  $("#resultState").textContent = "RUNNING";
  $("#resultsSubheading").textContent = "Calibrating the baseline and solving the current scenario…";
  try {
    state.lastResult = await apiRequest("/simulate", { method: "POST", body: JSON.stringify({ patient: payload, controls: controls() }) });
    renderResults(state.lastResult);
    rerun.disabled = false;
    showToast("Twin simulation completed.");
    $("#resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
    return true;
  } catch (error) {
    $("#resultState").className = "state-badge waiting";
    $("#resultState").textContent = "RUN FAILED";
    $("#resultsSubheading").textContent = "The current run could not be completed. Review the input message above.";
    showFormError(error.message);
    return false;
  } finally {
    button.disabled = false;
    button.innerHTML = "Save &amp; run twin <span>→</span>";
    if (state.lastResult) rerun.disabled = false;
  }
}

function formatNumber(value, digits = 1) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function signed(value, digits = 1, suffix = "") {
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value, digits)}${suffix}`;
}

function renderResults(result) {
  const baseline = result.baseline.summary;
  const scenario = result.scenario.summary;
  const metricMap = [
    ["metricEf", "metricEfDelta", scenario.ejection_fraction_percent, baseline.ejection_fraction_percent, "%", " pp"],
    ["metricSv", "metricSvDelta", scenario.stroke_volume_ml, baseline.stroke_volume_ml, " mL", " mL"],
    ["metricCo", "metricCoDelta", scenario.cardiac_output_l_min, baseline.cardiac_output_l_min, " L/min", " L/min"],
    ["metricMap", "metricMapDelta", scenario.mean_arterial_pressure_mmhg, baseline.mean_arterial_pressure_mmhg, " mmHg", " mmHg"]
  ];
  metricMap.forEach(([valueId, deltaId, scenarioValue, baselineValue, suffix, deltaSuffix]) => {
    $("#" + valueId).textContent = `${formatNumber(scenarioValue, suffix.includes("L/min") ? 2 : 1)}${suffix}`;
    const delta = scenarioValue - baselineValue;
    const deltaElement = $("#" + deltaId);
    deltaElement.textContent = `${signed(delta, suffix.includes("L/min") ? 2 : 1, deltaSuffix)} vs baseline`;
    deltaElement.className = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "";
  });
  $("#resultState").className = "state-badge ready";
  $("#resultState").textContent = "CURRENT RUN";
  $("#resultsSubheading").textContent = `${state.selectedId || "Current patient"} · ${result.volume_input.note}`;
  $("#downloadResults").disabled = false;
  renderComparison(baseline, scenario);
  drawResultsCharts(result);
}

function renderComparison(baseline, scenario) {
  const rows = [
    ["Ejection fraction", baseline.ejection_fraction_percent, scenario.ejection_fraction_percent, "%", " pp"],
    ["EDV", baseline.end_diastolic_volume_ml, scenario.end_diastolic_volume_ml, "mL", " mL"],
    ["ESV", baseline.end_systolic_volume_ml, scenario.end_systolic_volume_ml, "mL", " mL"],
    ["Stroke volume", baseline.stroke_volume_ml, scenario.stroke_volume_ml, "mL", " mL"],
    ["Cardiac output", baseline.cardiac_output_l_min, scenario.cardiac_output_l_min, "L/min", " L/min"],
    ["MAP", baseline.mean_arterial_pressure_mmhg, scenario.mean_arterial_pressure_mmhg, "mmHg", " mmHg"],
    ["Emax", baseline.emax_mmhg_per_ml, scenario.emax_mmhg_per_ml, "mmHg/mL", " mmHg/mL"]
  ];
  $("#comparisonBody").innerHTML = rows.map(([label, base, scene, unit, deltaUnit]) => {
    const digits = unit === "L/min" ? 2 : 1;
    const delta = scene - base;
    const className = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "";
    return `<tr><td>${label} <span class="table-unit">(${unit})</span></td><td>${formatNumber(base, digits)}</td><td>${formatNumber(scene, digits)}</td><td class="${className}">${signed(delta, digits, deltaUnit)}</td></tr>`;
  }).join("");
}

function drawResultsCharts(result) {
  const trace = result.scenario.trace;
  drawLineChart($("#pressureChart"), trace.time_s, [
    { values: trace.ventricular_pressure_mmhg, color: "#08a69a" },
    { values: trace.arterial_pressure_mmhg, color: "#ed7166" }
  ], "mmHg");
  drawLineChart($("#volumeChart"), trace.time_s, [{ values: trace.ventricular_volume_ml, color: "#637ed4" }], "mL");
}

function drawEmptyChart(canvas, text = "Run a patient twin to view the trace") {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "#e5ebef";
  ctx.setLineDash([4, 5]);
  ctx.beginPath(); ctx.moveTo(36, rect.height / 2); ctx.lineTo(rect.width - 16, rect.height / 2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#93a1af"; ctx.font = "11px Manrope, Arial"; ctx.textAlign = "center";
  ctx.fillText(text, rect.width / 2, rect.height / 2 - 12);
}

function drawLineChart(canvas, xValues, series, unit) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !xValues?.length) return drawEmptyChart(canvas);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const width = rect.width, height = rect.height;
  const pad = { left: 40, right: 14, top: 14, bottom: 26 };
  const allValues = series.flatMap((item) => item.values).filter(Number.isFinite);
  let minY = Math.min(...allValues), maxY = Math.max(...allValues);
  const spread = Math.max(maxY - minY, 1);
  minY -= spread * .12; maxY += spread * .12;
  const minX = xValues[0], maxX = xValues[xValues.length - 1] || minX + 1;
  const x = (value) => pad.left + ((value - minX) / (maxX - minX || 1)) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((value - minY) / (maxY - minY || 1)) * (height - pad.top - pad.bottom);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "9px Manrope, Arial";
  ctx.fillStyle = "#95a2af";
  ctx.strokeStyle = "#e7edf1";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const value = minY + ((maxY - minY) * index) / 4;
    const pointY = y(value);
    ctx.beginPath(); ctx.moveTo(pad.left, pointY); ctx.lineTo(width - pad.right, pointY); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(formatNumber(value, 0), pad.left - 6, pointY + 3);
  }
  ctx.textAlign = "left"; ctx.fillText(unit, 2, 11);
  ctx.textAlign = "center";
  ctx.fillText(`${formatNumber(minX, 1)} s`, pad.left, height - 7);
  ctx.fillText(`${formatNumber(maxX, 1)} s`, width - pad.right, height - 7);
  series.forEach((item) => {
    ctx.beginPath(); ctx.strokeStyle = item.color; ctx.lineWidth = 2; ctx.lineJoin = "round";
    item.values.forEach((value, index) => {
      if (index === 0) ctx.moveTo(x(xValues[index]), y(value));
      else ctx.lineTo(x(xValues[index]), y(value));
    });
    ctx.stroke();
  });
}

function drawCurrentOrEmptyCharts() {
  if (state.lastResult) drawResultsCharts(state.lastResult);
  else {
    drawEmptyChart($("#pressureChart"));
    drawEmptyChart($("#volumeChart"));
  }
}

function clearRunDisplay() {
  state.lastResult = null;
  $("#rerunScenario").disabled = true;
  $("#downloadResults").disabled = true;
  $("#comparisonBody").innerHTML = '<tr class="empty-row"><td colspan="4">No simulation output yet.</td></tr>';
  $("#resultState").className = "state-badge waiting";
  $("#resultState").textContent = "AWAITING RUN";
  $("#resultsSubheading").textContent = "Run a validated patient twin to view baseline and scenario physiology.";
  ["metricEf", "metricSv", "metricCo", "metricMap"].forEach((id) => { $("#" + id).textContent = "—"; });
  ["metricEfDelta", "metricSvDelta", "metricCoDelta", "metricMapDelta"].forEach((id) => { $("#" + id).textContent = "Scenario − baseline"; $("#" + id).className = ""; });
  drawCurrentOrEmptyCharts();
}

function resetScenario() {
  ["contractility", "preload", "afterload", "heartRateDelta"].forEach((id) => { $("#" + id).value = 0; updateRangeOutput(id); });
}

function updateRangeOutput(id) {
  const output = $("#" + id + "Out");
  const value = Number($("#" + id).value);
  const suffix = id === "heartRateDelta" ? " bpm" : "%";
  output.textContent = `${value > 0 ? "+" : ""}${value}${suffix}`;
}

function renderPatientList() {
  const query = $("#patientSearch")?.value.trim().toLowerCase() || "";
  const records = state.records.filter((record) => record.id.toLowerCase().includes(query));
  $("#patientCount").textContent = `${records.length} ${records.length === 1 ? "record" : "records"}`;
  const list = $("#patientList");
  if (!records.length) {
    list.innerHTML = `<tr class="empty-row"><td colspan="6">No patient record matches “${escapeHtml(query)}”. <button id="addQueryPatient" class="text-button" type="button">Create it as a new draft</button></td></tr>`;
    $("#addQueryPatient")?.addEventListener("click", () => newPatient(query.toUpperCase()));
    return;
  }
  list.innerHTML = records.map((record) => `<tr><td>${escapeHtml(record.id)}</td><td>${escapeHtml(record.ageBand)}</td><td>${escapeHtml(record.nyha)}</td><td><span class="source-chip ${record.source.startsWith("Echo") ? "echo" : "manual"}">${escapeHtml(record.source)}</span></td><td>${escapeHtml(record.edited || "This session")}</td><td><button class="select-patient" data-patient="${encodeURIComponent(record.id)}" type="button">Open →</button></td></tr>`).join("");
  $$(".select-patient").forEach((button) => button.addEventListener("click", () => openPatient(decodeURIComponent(button.dataset.patient))));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

function openPatient(id) {
  const record = state.records.find((entry) => entry.id === id);
  if (!record) return;
  state.selectedId = record.id;
  clearRunDisplay();
  $("#identifier").value = record.id;
  $("#ageBand").value = record.ageBand;
  $("#sex").value = record.sex;
  $("#nyha").value = record.nyha;
  $("#heartRate").value = record.heartRate;
  $("#systolicBp").value = record.systolicBp;
  $("#diastolicBp").value = record.diastolicBp;
  $("#lvDiameter").value = record.lvDiameter ?? "";
  $("#lvef").value = record.lvef;
  $("#edv").value = record.edv ?? "";
  $("#esv").value = record.esv ?? "";
  if (record.source.startsWith("Echo")) {
    state.echo = { file: null, result: record.echo?.result || null, confirmed: Boolean(record.echo?.confirmed) };
    changeSource("echonet");
    $("#uploadLabel").textContent = "Stored extraction values";
    $("#uploadHint").textContent = "Upload a new study to replace this extraction.";
    renderEchoResult();
  } else {
    changeSource("manual");
  }
  setFormSaved(record.id);
  showFormError("");
  switchView("workspace");
  openPatientEditor();
  showToast(`${record.id} opened in the workspace.`);
}

function newPatient(prefill = "") {
  state.selectedId = null;
  state.echo = { file: null, result: null, confirmed: false };
  $("#patientForm").reset();
  $("#identifier").value = prefill;
  changeSource("manual");
  $("#uploadLabel").textContent = "Drop an echo study here";
  $("#uploadHint").textContent = "DICOM, AVI, MP4 or MOV · max 200 MB";
  renderEchoResult();
  resetScenario();
  clearRunDisplay();
  markDraft();
  switchView("workspace");
  openPatientEditor();
  $("#identifier").focus();
}

function downloadResults() {
  if (!state.lastResult) return;
  const base = state.lastResult.baseline.summary;
  const scenario = state.lastResult.scenario.summary;
  const rows = [
    ["Metric", "Baseline", "Scenario", "Change"],
    ["Ejection fraction (%)", base.ejection_fraction_percent, scenario.ejection_fraction_percent, scenario.ejection_fraction_percent - base.ejection_fraction_percent],
    ["EDV (mL)", base.end_diastolic_volume_ml, scenario.end_diastolic_volume_ml, scenario.end_diastolic_volume_ml - base.end_diastolic_volume_ml],
    ["ESV (mL)", base.end_systolic_volume_ml, scenario.end_systolic_volume_ml, scenario.end_systolic_volume_ml - base.end_systolic_volume_ml],
    ["Stroke volume (mL)", base.stroke_volume_ml, scenario.stroke_volume_ml, scenario.stroke_volume_ml - base.stroke_volume_ml],
    ["Cardiac output (L/min)", base.cardiac_output_l_min, scenario.cardiac_output_l_min, scenario.cardiac_output_l_min - base.cardiac_output_l_min],
    ["MAP (mmHg)", base.mean_arterial_pressure_mmhg, scenario.mean_arterial_pressure_mmhg, scenario.mean_arterial_pressure_mmhg - base.mean_arterial_pressure_mmhg]
  ];
  const blob = new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${state.selectedId || "heart-twin"}-scenario.csv`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $("#mobileMenu").addEventListener("click", () => $(".sidebar").classList.toggle("open"));
  $("#newPatient").addEventListener("click", () => newPatient());
  $("#newPatientFromLibrary").addEventListener("click", () => newPatient());
  $("#overviewNewPatient").addEventListener("click", () => newPatient());
  $("#editPatient").addEventListener("click", openPatientEditor);
  $("#changePatient").addEventListener("click", () => switchView("patients"));
  $("#closePatientEditor").addEventListener("click", closePatientEditor);
  $("#choosePatient").addEventListener("click", () => switchView("patients"));
  $$(".source-tab").forEach((button) => button.addEventListener("click", () => { changeSource(button.dataset.source); markDraft(); }));
  $("#patientForm").addEventListener("submit", saveAndRun);
  $("#validateBtn").addEventListener("click", () => validateInputs());
  $("#rerunScenario").addEventListener("click", () => simulate());
  $("#resetScenario").addEventListener("click", resetScenario);
  ["contractility", "preload", "afterload", "heartRateDelta"].forEach((id) => $("#" + id).addEventListener("input", () => updateRangeOutput(id)));
  $$("#patientForm input, #patientForm select").forEach((element) => element.addEventListener("input", markDraft));
  $("#echoFile").addEventListener("change", (event) => setEchoFile(event.target.files[0]));
  $("#runEchoNet").addEventListener("click", runEchoNet);
  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => { const [file] = event.dataTransfer.files; if (file) setEchoFile(file); });
  $("#patientSearch").addEventListener("input", renderPatientList);
  $("#downloadResults").addEventListener("click", downloadResults);
  $("#openSettings").addEventListener("click", () => { $("#apiBaseInput").value = state.apiBase; setSettingsFeedback(""); $("#settingsDialog").showModal(); });
  $("#checkConnection").addEventListener("click", async () => { const candidate = $("#apiBaseInput").value.trim().replace(/\/$/, ""); if (candidate) state.apiBase = candidate; await checkConnection(); });
  $("#saveSettings").addEventListener("click", () => { const candidate = $("#apiBaseInput").value.trim().replace(/\/$/, ""); if (candidate) { state.apiBase = candidate; localStorage.setItem("cardioTwinApiBase", candidate); } });
  $("#whyVolumes").addEventListener("click", () => $("#helpDialog").showModal());
  $("#toggleTheme").addEventListener("click", () => { document.body.classList.toggle("dark-mode"); localStorage.setItem("cardioTwinDark", document.body.classList.contains("dark-mode")); drawCurrentOrEmptyCharts(); });
  window.addEventListener("resize", () => { window.clearTimeout(bindEvents.resizeTimer); bindEvents.resizeTimer = window.setTimeout(drawCurrentOrEmptyCharts, 120); });
}

function initialise() {
  if (localStorage.getItem("cardioTwinDark") === "true") document.body.classList.add("dark-mode");
  preparePatientEditor();
  bindEvents();
  renderPatientList();
  resetScenario();
  renderPatientOverview();
  drawCurrentOrEmptyCharts();
  checkConnection({ quiet: true });
}

initialise();
