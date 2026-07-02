const state = {
  modes: [],
  stateColors: {},
  graphEdges: [],
  selectedModeKey: "Rickshaw_Dhaka",
  minStep: 0,
  maxStep: 999,
  currentStep: 0,
  isPlaying: false,
  timerId: null,
  playbackDelay: 350,
  agentLimit: 20,
  frameData: null,
  liveData: null,
  summaryRows: [],
};

const elements = {
  serverStatus: document.getElementById("serverStatus"),
  modeSelect: document.getElementById("modeSelect"),
  playButton: document.getElementById("playButton"),
  resetButton: document.getElementById("resetButton"),
  stepSlider: document.getElementById("stepSlider"),
  stepValue: document.getElementById("stepValue"),
  speedSelect: document.getElementById("speedSelect"),
  agentLimit: document.getElementById("agentLimit"),
  agentLimitValue: document.getElementById("agentLimitValue"),
  modeInfo: document.getElementById("modeInfo"),
  canvas: document.getElementById("simulationCanvas"),
  stateLegend: document.getElementById("stateLegend"),
  metricLiveTbi: document.getElementById("metricLiveTbi"),
  metricAnchoring: document.getElementById("metricAnchoring"),
  metricThresholding: document.getElementById("metricThresholding"),
  metricInteriority: document.getElementById("metricInteriority"),
  metricEdgeCoupling: document.getElementById("metricEdgeCoupling"),
  metricRhythm: document.getElementById("metricRhythm"),
  dimensionBars: document.getElementById("dimensionBars"),
  currentStateCounts: document.getElementById("currentStateCounts"),
  summaryBars: document.getElementById("summaryBars"),
  summaryTable: document.getElementById("summaryTable"),
};

const ctx = elements.canvas.getContext("2d");

async function fetchJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }

  return response.json();
}

function setServerStatus(label, className) {
  elements.serverStatus.textContent = label;
  elements.serverStatus.className = `status-pill ${className}`;
}

function formatNumber(value) {
  const numberValue = Number(value || 0);
  return numberValue.toFixed(4);
}

function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

async function initializeApp() {
  try {
    setServerStatus("Connecting", "");

    const modesResponse = await fetchJson("/api/modes");
    const graphResponse = await fetchJson("/api/graph");
    const summaryResponse = await fetchJson("/api/summary");

    state.modes = modesResponse.modes.filter((mode) => mode.exists);
    state.stateColors = modesResponse.state_colors;
    state.graphEdges = graphResponse.edges;
    state.summaryRows = summaryResponse.rows || [];

    populateModeSelect();
    renderLegend();
    renderSummary();

    await loadSelectedMode(state.selectedModeKey);

    setServerStatus("Ready", "ready");
  } catch (error) {
    console.error(error);
    setServerStatus("Error", "error");
    alert("Could not initialize the app. Make sure the mode comparison runner has already generated the event logs.");
  }
}

function populateModeSelect() {
  clearChildren(elements.modeSelect);

  state.modes.forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode.mode_key;
    option.textContent = mode.label;

    if (mode.mode_key === state.selectedModeKey) {
      option.selected = true;
    }

    elements.modeSelect.appendChild(option);
  });

  if (!state.modes.some((mode) => mode.mode_key === state.selectedModeKey)) {
    state.selectedModeKey = state.modes[0]?.mode_key || "conventional_baseline";
  }
}

function renderLegend() {
  clearChildren(elements.stateLegend);

  Object.entries(state.stateColors).forEach(([stateName, color]) => {
    const item = document.createElement("div");
    item.className = "legend-item";

    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = color;

    const label = document.createElement("span");
    label.textContent = stateName;

    item.appendChild(swatch);
    item.appendChild(label);
    elements.stateLegend.appendChild(item);
  });
}

async function loadSelectedMode(modeKey) {
  pausePlayback();

  state.selectedModeKey = modeKey;
  state.currentStep = 0;

  const metadata = await fetchJson(`/api/simulation/${modeKey}/metadata`);

  state.minStep = metadata.min_step;
  state.maxStep = metadata.max_step;
  state.currentStep = state.minStep;

  elements.stepSlider.min = state.minStep;
  elements.stepSlider.max = state.maxStep;
  elements.stepSlider.value = state.currentStep;
  elements.stepValue.textContent = state.currentStep;

  renderModeInfo(metadata);

  await updateFrame();
}

function renderModeInfo(metadata) {
  clearChildren(elements.modeInfo);

  const infoRows = [
    ["Mode", metadata.label],
    ["City", metadata.city || "—"],
    ["Country", metadata.country || "—"],
    ["Movement family", metadata.movement_family || "—"],
    ["Agents", metadata.agent_count],
    ["Rows", metadata.row_count],
    ["Enclosure", metadata.enclosure_factor],
  ];

  infoRows.forEach(([label, value]) => {
    const row = document.createElement("div");

    const dt = document.createElement("dt");
    dt.textContent = label;

    const dd = document.createElement("dd");
    dd.textContent = value;

    row.appendChild(dt);
    row.appendChild(dd);
    elements.modeInfo.appendChild(row);
  });
}

async function updateFrame() {
  const frameUrl = `/api/simulation/${state.selectedModeKey}/frame?step=${state.currentStep}&agent_limit=${state.agentLimit}`;
  const liveUrl = `/api/simulation/${state.selectedModeKey}/live?step=${state.currentStep}`;

  const [frameResponse, liveResponse] = await Promise.all([
    fetchJson(frameUrl),
    fetchJson(liveUrl),
  ]);

  state.frameData = frameResponse;
  state.liveData = liveResponse;

  elements.stepSlider.value = state.currentStep;
  elements.stepValue.textContent = state.currentStep;

  drawSimulation();
  renderLiveMetrics();
  renderStateCounts(frameResponse.state_counts || []);
}

function graphToCanvasX(x) {
  const padding = 44;
  const usableWidth = elements.canvas.width - padding * 2;
  return padding + (x / 11) * usableWidth;
}

function graphToCanvasY(y) {
  const padding = 44;
  const usableHeight = elements.canvas.height - padding * 2;
  return elements.canvas.height - padding - (y / 11) * usableHeight;
}

function drawSimulation() {
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

  drawCanvasBackground();
  drawGraphEdges();
  drawAgents();
}

function drawCanvasBackground() {
  ctx.fillStyle = "#10131b";
  ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 11; i += 1) {
    const x = graphToCanvasX(i);
    const y = graphToCanvasY(i);

    ctx.beginPath();
    ctx.moveTo(x, graphToCanvasY(0));
    ctx.lineTo(x, graphToCanvasY(11));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(graphToCanvasX(0), y);
    ctx.lineTo(graphToCanvasX(11), y);
    ctx.stroke();
  }
}

function drawGraphEdges() {
  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.lineWidth = 1.2;

  state.graphEdges.forEach((edge) => {
    ctx.beginPath();
    ctx.moveTo(graphToCanvasX(edge.x0), graphToCanvasY(edge.y0));
    ctx.lineTo(graphToCanvasX(edge.x1), graphToCanvasY(edge.y1));
    ctx.stroke();
  });
}

function drawAgents() {
  const agents = state.frameData?.agents || [];

  agents.forEach((agent) => {
    const x = graphToCanvasX(agent.x);
    const y = graphToCanvasY(agent.y);

    const occupancy = Number(agent.occupancy || 0);
    const radius = 7 + occupancy * 2;

    ctx.beginPath();
    ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = state.stateColors[agent.state] || "#ffffff";
    ctx.fill();

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.stroke();

    if (agent.near_anchor) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 6, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(125,211,252,0.8)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  });
}

function renderLiveMetrics() {
  const scores = state.liveData?.live_scores || {};

  elements.metricLiveTbi.textContent = formatNumber(scores.live_tbi);
  elements.metricAnchoring.textContent = formatNumber(scores.anchoring);
  elements.metricThresholding.textContent = formatNumber(scores.thresholding);
  elements.metricInteriority.textContent = formatNumber(scores.interiority);
  elements.metricEdgeCoupling.textContent = formatNumber(scores.edge_coupling);
  elements.metricRhythm.textContent = formatNumber(scores.rhythmic_repetition);

  renderDimensionBars(scores);
}

function renderDimensionBars(scores) {
  clearChildren(elements.dimensionBars);

  const dimensions = [
    ["Anchoring", scores.anchoring],
    ["Thresholding", scores.thresholding],
    ["Interiority", scores.interiority],
    ["Pausing", scores.pausing],
    ["Edge Coupling", scores.edge_coupling],
    ["Rhythmic Repetition", scores.rhythmic_repetition],
  ];

  dimensions.forEach(([label, value]) => {
    const numericValue = Number(value || 0);

    const row = document.createElement("div");
    row.className = "bar-row";

    const labelElement = document.createElement("div");
    labelElement.className = "bar-label";
    labelElement.textContent = label;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.min(100, numericValue * 100)}%`;

    const valueElement = document.createElement("div");
    valueElement.className = "bar-value";
    valueElement.textContent = formatNumber(numericValue);

    track.appendChild(fill);
    row.appendChild(labelElement);
    row.appendChild(track);
    row.appendChild(valueElement);

    elements.dimensionBars.appendChild(row);
  });
}

function renderStateCounts(rows) {
  clearChildren(elements.currentStateCounts);

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "compact-row";
    empty.textContent = "No state data";
    elements.currentStateCounts.appendChild(empty);
    return;
  }

  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "compact-row";

    const stateName = document.createElement("span");
    stateName.textContent = row.state;

    const count = document.createElement("strong");
    count.textContent = row.count;

    item.appendChild(stateName);
    item.appendChild(count);

    elements.currentStateCounts.appendChild(item);
  });
}

function renderSummary() {
  renderSummaryBars();
  renderSummaryTable();
}

function makeSummaryLabel(row) {
  if (row.scenario_name === "conventional_baseline") {
    return "Conventional baseline";
  }

  const displayName = row.display_name || row.scenario_name;
  const city = row.city || "";

  return city ? `${displayName} (${city})` : displayName;
}

function renderSummaryBars() {
  clearChildren(elements.summaryBars);

  const rows = [...state.summaryRows].sort((a, b) => {
    return Number(b.mean_TBI || 0) - Number(a.mean_TBI || 0);
  });

  rows.forEach((row) => {
    const tbi = Number(row.mean_TBI || 0);

    const item = document.createElement("div");
    item.className = "summary-row";

    const label = document.createElement("div");
    label.className = "summary-label";
    label.textContent = makeSummaryLabel(row);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.min(100, tbi * 100)}%`;

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = formatNumber(tbi);

    track.appendChild(fill);
    item.appendChild(label);
    item.appendChild(track);
    item.appendChild(value);

    elements.summaryBars.appendChild(item);
  });
}

function renderSummaryTable() {
  clearChildren(elements.summaryTable);

  if (!state.summaryRows.length) {
    const row = elements.summaryTable.insertRow();
    const cell = row.insertCell();
    cell.textContent = "No summary data found. Run python src/simulation/mode_comparison_runner.py first.";
    return;
  }

  const columns = [
    "scenario_name",
    "display_name",
    "city",
    "country",
    "movement_family",
    "mean_TBI",
    "mean_anchoring_score",
    "mean_thresholding_score",
    "mean_interiority_score",
    "mean_pausing_score",
    "mean_edge_coupling_score",
    "mean_rhythmic_repetition_score",
    "mean_completed_cycles",
  ];

  const thead = elements.summaryTable.createTHead();
  const headerRow = thead.insertRow();

  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  });

  const tbody = elements.summaryTable.createTBody();

  const rows = [...state.summaryRows].sort((a, b) => {
    return Number(b.mean_TBI || 0) - Number(a.mean_TBI || 0);
  });

  rows.forEach((row) => {
    const tableRow = tbody.insertRow();

    columns.forEach((column) => {
      const cell = tableRow.insertCell();
      cell.textContent = row[column] ?? "";
    });
  });
}

function playPlayback() {
  if (state.isPlaying) {
    return;
  }

  state.isPlaying = true;
  elements.playButton.textContent = "Pause";

  state.timerId = window.setInterval(async () => {
    if (state.currentStep >= state.maxStep) {
      pausePlayback();
      return;
    }

    state.currentStep += 1;

    try {
      await updateFrame();
    } catch (error) {
      console.error(error);
      pausePlayback();
    }
  }, state.playbackDelay);
}

function pausePlayback() {
  state.isPlaying = false;
  elements.playButton.textContent = "Play";

  if (state.timerId !== null) {
    window.clearInterval(state.timerId);
    state.timerId = null;
  }
}

function togglePlayback() {
  if (state.isPlaying) {
    pausePlayback();
  } else {
    playPlayback();
  }
}

function resetPlayback() {
  pausePlayback();
  state.currentStep = state.minStep;
  updateFrame();
}

elements.modeSelect.addEventListener("change", async (event) => {
  await loadSelectedMode(event.target.value);
});

elements.playButton.addEventListener("click", () => {
  togglePlayback();
});

elements.resetButton.addEventListener("click", () => {
  resetPlayback();
});

elements.stepSlider.addEventListener("input", async (event) => {
  pausePlayback();
  state.currentStep = Number(event.target.value);
  await updateFrame();
});

elements.speedSelect.addEventListener("change", (event) => {
  state.playbackDelay = Number(event.target.value);

  if (state.isPlaying) {
    pausePlayback();
    playPlayback();
  }
});

elements.agentLimit.addEventListener("input", async (event) => {
  state.agentLimit = Number(event.target.value);
  elements.agentLimitValue.textContent = state.agentLimit;
  await updateFrame();
});

window.addEventListener("resize", () => {
  drawSimulation();
});

initializeApp();