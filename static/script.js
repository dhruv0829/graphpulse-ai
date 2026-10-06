"use strict";
/* GraphPulse AI — vanilla JS client for the FastAPI + ONNX GCN backend */

const API_BASE = "";
const NUM_NODES_DEFAULT = 2708;
const SAMPLE_SIZE = 160;
const FALLBACK_CLASSES = ["Case_Based","Genetic_Algorithms","Neural_Networks","Probabilistic_Methods","Reinforcement_Learning","Rule_Learning","Theory"];
const COLORS = ["#f5a524","#4be3a0","#38e1ff","#8b6cff","#e050c0","#ff6b6b","#7fa4ff"];
const NEUTRAL = "#5d6b82";
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const RING_C = 2 * Math.PI * 88;

const $ = (id) => document.getElementById(id);
const state = {
  classes: FALLBACK_CLASSES.slice(),
  numNodes: NUM_NODES_DEFAULT,
  providers: [],
  health: null, info: null,
  datasetOk: null, inferenceOk: null,
  nodes: [], byId: new Map(), edges: [], adj: new Map(),
  preds: new Map(),
  selected: null, hover: null,
  view: { k: 1, x: 0, y: 0 },
  last: null,
  busy: false,
};

/* ---------- utilities ---------- */
const fmtPct = (p) => (p * 100).toFixed(3) + "%";
const pad2 = (n) => String(n).padStart(2, "0");
function seeded(seed) { let s = (seed * 2654435761) >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

function countUp(el, to, ms = 1100, fmt = (v) => Math.round(v).toLocaleString("en-US")) {
  if (REDUCED) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - t0) / ms);
    el.textContent = fmt(to * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

async function api(path, options) {
  let res;
  try { res = await fetch(API_BASE + path, options); }
  catch (e) { const err = new Error("network"); err.kind = "network"; throw err; }
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON body */ }
  if (!res.ok) {
    const err = new Error("http"); err.kind = "http"; err.status = res.status;
    const d = body && body.detail;
    err.detail = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : "";
    throw err;
  }
  if (body === null) { const err = new Error("empty"); err.kind = "empty"; throw err; }
  return body;
}

function describeError(e) {
  if (e.kind === "network") return "Cannot reach the API. Check that uvicorn is running.";
  if (e.kind === "empty") return "The API returned an empty response.";
  if (e.kind === "malformed") return "The API response was not in the expected format.";
  if (e.kind === "http") {
    if (e.status === 400) return e.detail || "Request rejected: node index is out of range.";
    if (e.status === 422) return e.detail || "Request validation failed.";
    if (e.status >= 500) return e.detail || "The server failed while running the model.";
    return `Unexpected response (HTTP ${e.status}).`;
  }
  return "Something went wrong.";
}

function showError(title, message) {
  const t = document.createElement("div");
  t.className = "toast";
  const b = document.createElement("b"); b.textContent = title;
  const m = document.createElement("span"); m.textContent = message;
  t.append(b, m);
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 6000);
}

/* ---------- health / info / diagnostics ---------- */
function setChip(id, textId, ok, text) {
  const c = $(id); c.classList.toggle("ok", ok); c.classList.toggle("bad", !ok);
  $(textId).textContent = text;
}

async function checkHealth() {
  try {
    const h = await api("/health");
    state.health = h;
    state.providers = Array.isArray(h.providers) ? h.providers : [];
    const healthy = h.status === "healthy";
    setChip("chipApi", "chipApiText", healthy, healthy ? "API ONLINE" : "API DEGRADED");
    setChip("chipModel", "chipModelText", healthy, healthy ? "MODEL READY" : "MODEL UNKNOWN");
    setChip("chipOrt", "chipOrtText", state.providers.length > 0, "ONNX RUNTIME");
  } catch (e) {
    state.health = null; state.providers = [];
    setChip("chipApi", "chipApiText", false, "API OFFLINE");
    setChip("chipModel", "chipModelText", false, "MODEL UNAVAILABLE");
    setChip("chipOrt", "chipOrtText", false, "ONNX RUNTIME");
  }
  updateSystemStatus();
  renderModelCard();
}

async function loadModelInfo() {
  try {
    const info = await api("/info");
    state.info = info;
    if (info.class_mapping && typeof info.class_mapping === "object") {
      const arr = [];
      Object.keys(info.class_mapping).forEach((k) => { arr[Number(k)] = info.class_mapping[k]; });
      if (arr.length && arr.every(Boolean)) state.classes = arr;
    }
    if (info.feature_dimension) $("mFeat").dataset.target = info.feature_dimension;
    if (info.num_classes) $("mClass").dataset.target = info.num_classes;
  } catch (e) {
    state.info = null;
    showError("MODEL INFO UNAVAILABLE", describeError(e));
  }
  buildLegend(); renderModelCard(); updateSystemStatus();
}

function renderModelCard() {
  const i = state.info, box = $("modelCard");
  if (!i) { box.innerHTML = ""; addRow(box, "STATUS", "Model info unavailable"); return; }
  box.innerHTML = "";
  const names = (arr) => (arr || []).map((x) => x.name).join(", ") || "—";
  addRow(box, "MODEL", i["Model Name"] || i.model_name || "—");
  addRow(box, "RUNTIME", "ONNX Runtime");
  addRow(box, "EXECUTION PROVIDER", state.providers.join(", ") || "—");
  addRow(box, "INPUTS", names(i.inputs));
  addRow(box, "FEATURE DIMENSION", i.feature_dimension ?? "—");
  addRow(box, "OUTPUT", names(i.outputs));
  addRow(box, "CLASSES", i.num_classes ?? "—");
}
function addRow(box, k, v) {
  const d = document.createElement("div");
  const dt = document.createElement("dt"); dt.textContent = k;
  const dd = document.createElement("dd"); dd.textContent = v;
  d.append(dt, dd); box.appendChild(d);
}

function updateSystemStatus() {
  const rows = [
    ["API CONNECTION", state.health ? true : false],
    ["MODEL LOADED", state.health ? state.health.status === "healthy" && !!state.info : false],
    ["ONNX RUNTIME", state.providers.length > 0],
    ["CORA DATASET", state.datasetOk],
    ["INFERENCE ENGINE", state.inferenceOk],
    ["GRAPH EXPLORER", !!ctx],
  ];
  const ul = $("diag"); ul.innerHTML = "";
  rows.forEach(([label, ok]) => {
    const li = document.createElement("li");
    if (ok === true) { li.className = "ok"; li.textContent = `[✓] ${label}`; }
    else if (ok === false) { li.className = "bad"; li.textContent = `[!] ${label} FAILED`; }
    else { li.className = "wait"; li.textContent = `[…] ${label} PENDING`; }
    ul.appendChild(li);
  });
}

/* ---------- prediction ---------- */
function extractPrediction(data, nodeId) {
  if (!data || typeof data !== "object" || !Array.isArray(data.predictions) || !data.predictions.length) {
    const e = new Error("malformed"); e.kind = "malformed"; throw e;
  }
  return data.predictions.map((p) => {
    const probabilities = p.probabilities || p.probabilites;
    if (!Array.isArray(probabilities) || !Array.isArray(p.logits) || probabilities.length !== p.logits.length ||
        !probabilities.every(Number.isFinite) || !p.logits.every(Number.isFinite) || typeof p.node_index !== "number") {
      const e = new Error("malformed"); e.kind = "malformed"; throw e;
    }
    return { nodeIndex: p.node_index, classId: p.predicted_class_id, className: p.predicted_class_name, probabilities, logits: p.logits };
  });
}

async function postNodes(ids) {
  const data = await api("/predict/cora_node", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_indices: ids }),
  });
  if (Number.isFinite(data.num_nodes)) { state.numNodes = data.num_nodes; $("mNodes").dataset.target = data.num_nodes; }
  if (Number.isFinite(data.num_edges)) $("mEdges").dataset.target = data.num_edges;
  return extractPrediction(data);
}

async function predictCoraNode(nodeId) {
  showLoading();
  try {
    const [pred] = await postNodes([nodeId]);
    state.inferenceOk = true; state.datasetOk = true;
    await finishLoading();
    cachePrediction(pred); ensureNode(pred.nodeIndex); relayout();
    state.selected = pred.nodeIndex; state.last = pred;
    renderPrediction(pred);
    updateSystemStatus();
    return pred;
  } catch (e) {
    if (e.kind === "network") { state.inferenceOk = false; checkHealth(); }
    else if (e.kind === "http" && e.status >= 500) { state.inferenceOk = false; }
    hideLoading(); updateSystemStatus();
    showError("NODE INFERENCE FAILED", describeError(e));
    return null;
  }
}

function cachePrediction(p) {
  const conf = Math.max(...p.probabilities);
  state.preds.set(p.nodeIndex, { ...p, conf });
}

function renderPrediction(p) {
  const conf = Math.max(...p.probabilities);
  const name = p.className || state.classes[p.classId] || "—";
  $("infTag").textContent = "COMPLETE";
  $("infNode").textContent = "#" + p.nodeIndex;
  $("infClass").textContent = name.toUpperCase();
  $("infClassId").textContent = pad2(p.classId);
  renderConfidence(conf, p.classId);
  $("ndId").textContent = "#" + p.nodeIndex;
  $("ndPos").textContent = "ACTIVE";
  $("ndPred").textContent = name;
  $("ndClass").textContent = p.classId;
  $("ndConf").textContent = fmtPct(conf);
  renderProbabilities(p.probabilities, p.classId);
  renderLogits(p.logits, p.probabilities, p.classId);
  renderClassConstellation(p);
  $("nodeInput").value = p.nodeIndex;
}

function renderConfidence(value, classId) {
  const ring = $("ringFg"), color = COLORS[classId] || COLORS[2];
  ring.style.stroke = color; ring.style.filter = `drop-shadow(0 0 6px ${color})`;
  ring.style.strokeDashoffset = RING_C * (1 - value);
  countUp($("confPct"), value * 100, 1000, (v) => v.toFixed(3) + "%");
}

function renderProbabilities(probs, top) {
  const box = $("bars"); box.innerHTML = "";
  probs.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "bar" + (i === top ? " top" : "");
    row.style.setProperty("--cc", COLORS[i] || "#fff");
    row.innerHTML = '<span class="name"></span><div class="track"><div class="fill"></div></div><span class="pct"></span>';
    row.querySelector(".name").textContent = state.classes[i] || "Class " + i;
    row.querySelector(".pct").textContent = fmtPct(p);
    box.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => { row.querySelector(".fill").style.width = Math.max(p * 100, p > 0 ? 0.6 : 0) + "%"; }));
  });
}

function renderLogits(logits, probs, top) {
  const body = $("logitBody"); body.innerHTML = "";
  logits.forEach((l, i) => {
    const tr = document.createElement("tr"); if (i === top) tr.className = "top";
    [state.classes[i] || "Class " + i, l.toFixed(4), fmtPct(probs[i])].forEach((v) => {
      const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function renderClassConstellation(p) {
  const svg = $("constellation"), cx = 220, cy = 200, R = 150;
  const n = state.classes.length;
  let s = `<defs><radialGradient id="cg"><stop offset="0" stop-color="#38e1ff" stop-opacity=".5"/><stop offset="1" stop-color="#38e1ff" stop-opacity="0"/></radialGradient></defs>`;
  s += `<circle cx="${cx}" cy="${cy}" r="${R + 30}" fill="none" stroke="rgba(120,160,210,.12)" stroke-dasharray="2 6"/>`;
  const pts = state.classes.map((_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
  });
  let lines = "", nodes = "";
  pts.forEach(([x, y], i) => {
    const pr = p ? p.probabilities[i] : 0, top = p && i === p.classId;
    const col = COLORS[i];
    lines += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${col}" stroke-opacity="${p ? 0.15 + 0.85 * pr : 0.2}" stroke-width="${top ? 3.5 : 1 + 2.5 * pr}" ${top ? "" : 'stroke-dasharray="4 5"'}/>`;
    const r = top ? 30 : 17 + 8 * Math.sqrt(pr);
    const ty = y < cy ? y - r - 8 : y + r + 14;
    nodes += `<g class="cn"><title>${state.classes[i]}: ${p ? fmtPct(pr) : "no prediction"}</title>
      <circle cx="${x}" cy="${y}" r="${r + 7}" fill="${col}" opacity="${top ? 0.25 : 0.07}"/>
      <circle cx="${x}" cy="${y}" r="${r}" fill="#0a0f18" stroke="${col}" stroke-width="${top ? 3 : 1.5}"/>
      <text x="${x}" y="${y + 4}" text-anchor="middle" font-size="${top ? 14 : 11}" font-weight="700">${pad2(i)}</text>
      <text x="${x}" y="${ty}" text-anchor="middle" font-size="10" fill-opacity=".8">${state.classes[i].replace(/_/g, " ")}</text>
      <text x="${x}" y="${ty + 12}" text-anchor="middle" font-size="10" style="fill:${col}">${p ? fmtPct(pr) : "—"}</text></g>`;
  });
  s += lines + `<circle cx="${cx}" cy="${cy}" r="70" fill="url(#cg)"/><circle cx="${cx}" cy="${cy}" r="30" fill="#0a0f18" stroke="#38e1ff" stroke-width="2"/><text x="${cx}" y="${cy + 5}" text-anchor="middle" font-size="14" font-weight="700">GCN</text>` + nodes;
  svg.innerHTML = s;
}

/* ---------- loading experience ---------- */
const STEPS = ["INITIALIZING NODE", "READING GRAPH FEATURES", "RUNNING GCN", "COMPUTING LOGITS", "NORMALIZING PROBABILITIES", "PREDICTION READY"];
let loadTimer = null, loadIdx = 0;
function setStep(i) {
  loadIdx = i;
  $("loaderStep").textContent = `${i < STEPS.length - 1 ? "↓ " : "✓ "}${STEPS[i]}`;
  $("loaderFill").style.width = ((i + 1) / STEPS.length) * 100 + "%";
}
function showLoading() {
  state.busy = true;
  $("loader").hidden = false; $("infTag").textContent = "RUNNING";
  setStep(0);
  clearInterval(loadTimer);
  loadTimer = setInterval(() => { if (loadIdx < STEPS.length - 2) setStep(loadIdx + 1); }, 220);
}
async function finishLoading() {
  clearInterval(loadTimer);
  setStep(STEPS.length - 1);
  await new Promise((r) => setTimeout(r, REDUCED ? 0 : 280));
  hideLoading();
}
function hideLoading() {
  clearInterval(loadTimer);
  state.busy = false; $("loader").hidden = true;
  if ($("infTag").textContent === "RUNNING") $("infTag").textContent = state.last ? "COMPLETE" : "IDLE";
}

/* ---------- search / selection ---------- */
function validateNodeId(raw) {
  const s = String(raw).trim();
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n >= 0 && n <= (state.numNodes - 1) ? n : null;
}
async function searchNode() {
  const err = $("searchError"), n = validateNodeId($("nodeInput").value);
  if (n === null) {
    err.hidden = false; err.textContent = `Node ID must be between 0 and ${state.numNodes - 1}.`;
    $("nodeInput").focus(); return;
  }
  err.hidden = true;
  await selectNode(n);
}
async function selectNode(nodeId) {
  if (state.busy) return;
  $("searchError").hidden = true;
  const p = await predictCoraNode(nodeId);
  if (p) focusNode(nodeId);
}
function randomNode() { selectNode(Math.floor(Math.random() * state.numNodes)); }

/* ---------- graph model ---------- */
function posFor(id, cls) {
  const r = seeded(id + 7);
  const a0 = cls == null ? r() * 6.283 : -Math.PI / 2 + (2 * Math.PI * cls) / state.classes.length;
  const cxp = cls == null ? 0 : 270 * Math.cos(a0), cyp = cls == null ? 0 : 270 * Math.sin(a0);
  const rad = (cls == null ? 220 : 85) * Math.sqrt(r()), a = r() * 6.283;
  return [cxp + rad * Math.cos(a), cyp + rad * Math.sin(a)];
}
function ensureNode(id) {
  if (state.byId.has(id)) return;
  const n = { id, x: 0, y: 0, cls: null };
  state.nodes.push(n); state.byId.set(id, n);
}
function relayout() {
  state.nodes.forEach((n) => {
    const p = state.preds.get(n.id);
    n.cls = p ? p.classId : null;
    [n.x, n.y] = posFor(n.id, n.cls);
  });
  buildEdges();
}
function buildEdges() {
  const nodes = state.nodes, edges = [], seen = new Set();
  state.adj = new Map(nodes.map((n) => [n.id, new Set()]));
  const add = (a, b) => {
    const k = a.id < b.id ? a.id + "-" + b.id : b.id + "-" + a.id;
    if (a === b || seen.has(k)) return; seen.add(k);
    edges.push([a, b]); state.adj.get(a.id).add(b.id); state.adj.get(b.id).add(a.id);
  };
  nodes.forEach((a) => {
    const near = nodes.filter((b) => b !== a && b.cls === a.cls)
      .map((b) => [b, (a.x - b.x) ** 2 + (a.y - b.y) ** 2]).sort((u, v) => u[1] - v[1]).slice(0, 2);
    near.forEach(([b]) => add(a, b));
    const r = seeded(a.id + 99);
    if (r() < 0.12) add(a, nodes[Math.floor(r() * nodes.length)]);
  });
  state.edges = edges;
}
async function loadSample() {
  const r = seeded(2708), ids = new Set();
  while (ids.size < SAMPLE_SIZE) ids.add(Math.floor(r() * NUM_NODES_DEFAULT));
  [...ids].sort((a, b) => a - b).forEach(ensureNode);
  relayout(); resetGraph();
  try {
    const preds = await postNodes([...ids]);
    preds.forEach(cachePrediction);
    state.datasetOk = true; state.inferenceOk = true;
    $("graphTag").textContent = `SAMPLE · ${state.nodes.length} OF ${state.numNodes.toLocaleString("en-US")} NODES`;
    relayout();
  } catch (e) {
    state.datasetOk = false;
    $("graphTag").textContent = "SAMPLE · CLASSES UNAVAILABLE";
    showError("GRAPH SAMPLE UNAVAILABLE", describeError(e));
  }
  updateSystemStatus(); runCounters();
}

/* ---------- graph canvas ---------- */
let ctx = null, canvas = null, W = 0, H = 0, DPR = 1;
const toScreen = (x, y) => [x * state.view.k + state.view.x, y * state.view.k + state.view.y];
const toWorld = (sx, sy) => [(sx - state.view.x) / state.view.k, (sy - state.view.y) / state.view.k];

function resizeGraph() {
  const b = canvas.getBoundingClientRect();
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = b.width; H = b.height;
  canvas.width = W * DPR; canvas.height = H * DPR;
}
function resetGraph() {
  if (!canvas) return;
  const k = Math.min(W, H) / 760;
  state.view = { k: k || 1, x: W / 2, y: H / 2 };
}
function zoomBy(f, sx = W / 2, sy = H / 2) {
  const v = state.view, nk = Math.min(6, Math.max(0.25, v.k * f)), r = nk / v.k;
  v.x = sx - (sx - v.x) * r; v.y = sy - (sy - v.y) * r; v.k = nk;
}
function focusNode(id) {
  const n = state.byId.get(id); if (!n) return;
  state.view.x = W / 2 - n.x * state.view.k; state.view.y = H / 2 - n.y * state.view.k;
}
function nodeRadius(n) { return n.id === state.selected ? 11 : n.id === state.hover ? 8.5 : 5; }
function hitTest(sx, sy) {
  const [wx, wy] = toWorld(sx, sy);
  let best = null, bd = Infinity;
  for (const n of state.nodes) {
    const d = Math.hypot(n.x - wx, n.y - wy) * state.view.k;
    if (d < Math.max(10, nodeRadius(n) + 4) && d < bd) { bd = d; best = n; }
  }
  return best;
}

function drawGraph(t) {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const sel = state.selected, nb = sel != null ? state.adj.get(sel) : null, k = state.view.k;
  const pulse = REDUCED ? 0 : Math.sin(t / 500);
  // edges
  ctx.lineWidth = 1;
  state.edges.forEach(([a, b]) => {
    const hot = sel != null && (a.id === sel || b.id === sel);
    const [x1, y1] = toScreen(a.x, a.y), [x2, y2] = toScreen(b.x, b.y);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    if (hot) { ctx.strokeStyle = "rgba(56,225,255,.9)"; ctx.lineWidth = 1.8; ctx.setLineDash(REDUCED ? [] : [6, 4]); ctx.lineDashOffset = -t / 40; }
    else { ctx.strokeStyle = sel != null ? "rgba(120,160,210,.07)" : "rgba(120,160,210,.2)"; ctx.lineWidth = 1; ctx.setLineDash([]); }
    ctx.stroke();
  });
  ctx.setLineDash([]);
  // nodes
  state.nodes.forEach((n) => {
    const [x, y] = toScreen(n.x, n.y);
    if (x < -20 || y < -20 || x > W + 20 || y > H + 20) return;
    const col = n.cls == null ? NEUTRAL : COLORS[n.cls];
    const isSel = n.id === sel, isNb = nb && nb.has(n.id);
    let r = nodeRadius(n) * Math.min(1.6, Math.max(0.8, Math.sqrt(k)));
    if (isNb) r *= 1.35;
    if (isSel) r += pulse * 1.5;
    ctx.globalAlpha = sel != null && !isSel && !isNb ? 0.35 : 1;
    ctx.shadowColor = col; ctx.shadowBlur = isSel ? 22 : isNb || n.id === state.hover ? 14 : 7;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
    ctx.shadowBlur = 0;
    if (isSel || isNb) { ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.arc(x, y, r + 3, 0, 6.2832); ctx.stroke(); }
    if (isSel) { ctx.fillStyle = "#fff"; ctx.font = "600 12px JetBrains Mono, monospace"; ctx.fillText("#" + n.id, x + r + 6, y - r - 2); }
  });
  ctx.globalAlpha = 1;
}
function frame(t) {
  if (!document.hidden) drawGraph(t);
  requestAnimationFrame(frame);
}

function buildLegend() {
  const lg = $("legend"); lg.innerHTML = "";
  state.classes.forEach((c, i) => {
    const s = document.createElement("span");
    const dot = document.createElement("i"); dot.style.background = COLORS[i];
    s.append(dot, document.createTextNode(c)); lg.appendChild(s);
  });
}

function initGraph() {
  canvas = $("graphCanvas"); ctx = canvas.getContext("2d");
  resizeGraph(); resetGraph();
  new ResizeObserver(() => { const wasEmpty = !W; resizeGraph(); if (wasEmpty) resetGraph(); }).observe(canvas);
  const tip = $("tooltip");
  let drag = null;
  const pos = (e) => { const b = canvas.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const [sx, sy] = pos(e); drag = { sx, sy, vx: state.view.x, vy: state.view.y, moved: false };
  });
  canvas.addEventListener("pointermove", (e) => {
    const [sx, sy] = pos(e);
    if (drag) {
      const dx = sx - drag.sx, dy = sy - drag.sy;
      if (Math.hypot(dx, dy) > 4) { drag.moved = true; canvas.classList.add("drag"); tip.hidden = true; }
      if (drag.moved) { state.view.x = drag.vx + dx; state.view.y = drag.vy + dy; return; }
    }
    const n = hitTest(sx, sy);
    state.hover = n ? n.id : null;
    canvas.style.cursor = n ? "pointer" : "";
    if (n) {
      const p = state.preds.get(n.id);
      tip.hidden = false;
      tip.innerHTML = "";
      const a = document.createElement("div"); a.textContent = "NODE #" + n.id;
      const b = document.createElement("div"); b.textContent = "CLASS: " + (p ? p.className || state.classes[p.classId] : "not classified");
      tip.append(a, b);
      tip.style.left = Math.min(sx + 14, W - 190) + "px"; tip.style.top = Math.max(sy - 46, 4) + "px";
    } else tip.hidden = true;
  });
  const end = (e) => {
    if (drag && !drag.moved) { const [sx, sy] = pos(e); const n = hitTest(sx, sy); if (n) selectNode(n.id); }
    drag = null; canvas.classList.remove("drag");
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", () => { drag = null; canvas.classList.remove("drag"); });
  canvas.addEventListener("pointerleave", () => { state.hover = null; tip.hidden = true; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); const [sx, sy] = pos(e); zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, sx, sy); }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    if (e.key === "+" || e.key === "=") zoomBy(1.2);
    else if (e.key === "-") zoomBy(1 / 1.2);
    else if (e.key === "0") resetGraph();
    else if (e.key.toLowerCase() === "r") randomNode();
  });
  $("zoomIn").onclick = () => zoomBy(1.25);
  $("zoomOut").onclick = () => zoomBy(1 / 1.25);
  $("zoomReset").onclick = resetGraph;
  requestAnimationFrame(frame);
}

/* ---------- hero background ---------- */
function initHero() {
  const c = $("heroCanvas"), g = c.getContext("2d");
  let w = 0, h = 0, run = true;
  const P = [];
  const size = () => { const b = c.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2); w = b.width; h = b.height; c.width = w * d; c.height = h * d; g.setTransform(d, 0, 0, d, 0, 0); };
  size(); window.addEventListener("resize", size);
  const count = Math.min(70, Math.floor(innerWidth / 18));
  for (let i = 0; i < count; i++) P.push({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * .25, vy: (Math.random() - .5) * .25, c: COLORS[i % 7] });
  const draw = () => {
    if (run && !document.hidden) {
      g.clearRect(0, 0, w, h);
      P.forEach((p) => { if (!REDUCED) { p.x += p.vx; p.y += p.vy; } if (p.x < 0 || p.x > w) p.vx *= -1; if (p.y < 0 || p.y > h) p.vy *= -1; });
      for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
        const d = Math.hypot(P[i].x - P[j].x, P[i].y - P[j].y);
        if (d < 140) { g.strokeStyle = `rgba(56,225,255,${0.16 * (1 - d / 140)})`; g.beginPath(); g.moveTo(P[i].x, P[i].y); g.lineTo(P[j].x, P[j].y); g.stroke(); }
      }
      P.forEach((p) => { g.fillStyle = p.c; g.globalAlpha = .7; g.beginPath(); g.arc(p.x, p.y, 2, 0, 6.2832); g.fill(); });
      g.globalAlpha = 1;
    }
    requestAnimationFrame(draw);
  };
  new IntersectionObserver((en) => { run = en[0].isIntersecting; }).observe(c);
  draw();
}

/* ---------- metrics ---------- */
function runCounters() {
  ["mNodes", "mEdges", "mFeat", "mClass"].forEach((id) => countUp($(id), Number($(id).dataset.target)));
}

/* ---------- boot ---------- */
async function init() {
  buildLegend();
  renderClassConstellation(null);
  updateSystemStatus();
  initHero();
  initGraph();
  runCounters();
  $("searchForm").addEventListener("submit", (e) => { e.preventDefault(); searchNode(); });
  $("btnRandom").addEventListener("click", randomNode);
  $("btnExplore").addEventListener("click", () => $("explorer").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }));
  $("btnModelInfo").addEventListener("click", () => $("modelSection").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }));
  await Promise.all([checkHealth(), loadModelInfo()]);
  if (state.health) await loadSample();
  else { state.datasetOk = false; updateSystemStatus(); showError("API OFFLINE", "Start the server with: uvicorn main:app --reload"); }
  runCounters();
  setInterval(checkHealth, 30000);
}
document.addEventListener("DOMContentLoaded", init);
