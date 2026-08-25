"use strict";
const DUR = 4.0;                       // analysed window (s)
const COLORS = { kick: "#ff5a5f", snare: "#ffd23f", hh: "#4cc9f0" };
const INSTS = ["kick", "snare", "hh"];
const LABELS = { kick: "Kick", snare: "Snare", hh: "Hi-hat" };

let ctx = null;                        // AudioContext (lazy)
const buffers = {};                    // url -> AudioBuffer
let current = null;                    // active source
let raf = null;
let lastData = null;
// step-sequencer state
let gridMeta = null, gridState = null, patternSources = [], gridRaf = null, patternT0 = 0, playingPattern = false;
const LABEL_W = 64;   // .srow label column (must match CSS grid-template-columns)

// ---------------------------------------------------------------- upload
const dz = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");

dz.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => { if (e.target.files[0]) analyze(e.target.files[0]); });
["dragover", "dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", e => { if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]); });

async function analyze(file) {
  stopAll();
  statusEl.hidden = false;
  statusText.textContent = "Analysing “" + file.name + "” — extracting one-shots can take ~30 s…";
  document.getElementById("results").hidden = true;
  const fd = new FormData(); fd.append("file", file);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "analysis failed");
    lastData = data;
    renderResults(data);
  } catch (err) {
    statusText.textContent = "Error: " + err.message;
    return;
  }
  statusEl.hidden = true;
}

// ---------------------------------------------------------------- render
function renderResults(d) {
  document.getElementById("results").hidden = false;
  document.getElementById("bpmBadge").textContent = d.tempo_bpm + " BPM";
  document.getElementById("statTempo").textContent = d.tempo_bpm + " BPM";

  let total = 0;
  const tbody = document.querySelector("#instTable tbody"); tbody.innerHTML = "";
  INSTS.forEach(inst => {
    const o = d.instruments[inst];
    const n = o.onsets_s.length; total += n;
    const vs = o.velocities || [];
    const mean = vs.length ? (vs.reduce((a, b) => a + b, 0) / vs.length) : 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td style="color:${COLORS[inst]}">${LABELS[inst]}</td><td>${n}</td><td>${mean.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById("statOnsets").textContent = total;
  document.getElementById("statLoss").textContent =
    d.reconstruction_loss != null ? d.reconstruction_loss.toFixed(4) : "—";

  buildLanes(d);
  drawWaveform(d);
  buildOneshots(d);
  if (d.quantized) renderStepGrid(d.quantized);
  // preload audio for playback
  if (d.urls.loop) loadBuffer(d.urls.loop);
  if (d.urls.recon) loadBuffer(d.urls.recon);
}

function buildLanes(d) {
  const lanes = document.getElementById("lanes");
  lanes.innerHTML = "";
  INSTS.forEach(inst => {
    const lane = document.createElement("div");
    lane.className = "lane " + inst;
    lane.innerHTML = `<span class="lane-label">${LABELS[inst]}</span>`;
    for (let i = 1; i < 16; i++) {                       // 16-step gridlines
      const g = document.createElement("div"); g.className = "gridline";
      g.style.left = (i / 16 * 100) + "%"; lane.appendChild(g);
    }
    const o = d.instruments[inst], vs = o.velocities || [];
    o.onsets_s.forEach((t, i) => {
      const v = Math.max(0.04, Math.min(1, vs[i] != null ? vs[i] : 1));
      const hit = document.createElement("div"); hit.className = "hit";
      hit.style.left = (t / DUR * 100) + "%";
      hit.style.height = (v * 82 + 8) + "%";
      hit.style.opacity = (0.35 + 0.65 * v).toFixed(2);
      hit.title = `${LABELS[inst]}  t=${t.toFixed(3)}s  vel=${(vs[i] ?? 1).toFixed(3)}`;
      lane.appendChild(hit);
    });
    lanes.appendChild(lane);
  });
  const ph = document.createElement("div"); ph.className = "playhead"; ph.id = "playhead";
  lanes.appendChild(ph);
}

function drawWaveform(d) {
  const cv = document.getElementById("waveform");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.height;
  cv.width = W * dpr; cv.height = H * dpr;
  const c = cv.getContext("2d"); c.scale(dpr, dpr);
  c.clearRect(0, 0, W, H);
  if (d.urls.loop) {
    fetch(d.urls.loop).then(r => r.arrayBuffer()).then(b => audioCtx().decodeAudioData(b)).then(buf => {
      const ch = buf.getChannelData(0), mid = H / 2;
      c.strokeStyle = "#4a5262"; c.beginPath();
      const step = Math.max(1, Math.floor(ch.length / W));
      for (let x = 0; x < W; x++) {
        let mn = 1, mx = -1;
        for (let i = 0; i < step; i++) { const s = ch[x * step + i] || 0; if (s < mn) mn = s; if (s > mx) mx = s; }
        c.moveTo(x, mid + mn * mid * 0.95); c.lineTo(x, mid + mx * mid * 0.95);
      }
      c.stroke();
      INSTS.forEach(inst => d.instruments[inst].onsets_s.forEach(t => {
        const x = t / DUR * W; c.strokeStyle = COLORS[inst]; c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(x, 0); c.lineTo(x, 14); c.stroke();
      }));
    });
  }
}

function buildOneshots(d) {
  const wrap = document.getElementById("oneshots"); wrap.innerHTML = "";
  INSTS.forEach(inst => {
    const url = d.urls.oneshots[inst]; if (!url) return;
    const row = document.createElement("div"); row.className = "oneshot";
    row.innerHTML = `<span class="dot" style="background:${COLORS[inst]}"></span>
      <span class="name">${LABELS[inst]}</span>
      <audio controls preload="none" src="${url}"></audio>`;
    wrap.appendChild(row);
  });
}

// ---------------------------------------------------------------- playback
function audioCtx() { if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)(); return ctx; }
async function loadBuffer(url) {
  if (buffers[url]) return buffers[url];
  const b = await fetch(url).then(r => r.arrayBuffer());
  buffers[url] = await audioCtx().decodeAudioData(b);
  return buffers[url];
}
async function play(url) {
  if (!url) return;
  stopAll();
  const buf = await loadBuffer(url);
  const src = audioCtx().createBufferSource();
  src.buffer = buf; src.connect(audioCtx().destination); src.start();
  const t0 = audioCtx().currentTime;
  current = src;
  const ph = document.getElementById("playhead"); ph.style.display = "block";
  (function tick() {
    const e = audioCtx().currentTime - t0;
    if (e >= DUR || current !== src) { ph.style.display = "none"; return; }
    ph.style.left = (e / DUR * 100) + "%";
    raf = requestAnimationFrame(tick);
  })();
  src.onended = () => { ph.style.display = "none"; if (current === src) current = null; };
}
function stopAll() {
  if (current) { try { current.stop(); } catch (e) {} current = null; }
  if (raf) cancelAnimationFrame(raf);
  const ph = document.getElementById("playhead"); if (ph) ph.style.display = "none";
  // pattern playback
  playingPattern = false;
  patternSources.forEach(s => { try { s.stop(); } catch (e) {} });
  patternSources = [];
  if (gridRaf) cancelAnimationFrame(gridRaf);
  const gph = document.getElementById("gridPlayhead"); if (gph) gph.style.display = "none";
}

// ---------------------------------------------------------------- step grid
function renderStepGrid(q) {
  gridMeta = q;
  gridState = q.step_velocities.map(row => row.slice());   // per-track [16] velocities (0 = off)
  document.getElementById("beatType").textContent =
    q.beat_type + " · swing " + q.swing.map(s => s.toFixed(2)).join(" / ");
  drawGrid();
}

function drawGrid() {
  const g = document.getElementById("stepGrid"); g.innerHTML = "";
  gridMeta.instruments.forEach((inst, j) => {
    const row = document.createElement("div"); row.className = "srow";
    const lab = document.createElement("div"); lab.className = "slabel";
    lab.style.color = COLORS[inst]; lab.textContent = LABELS[inst]; row.appendChild(lab);
    for (let s = 0; s < gridMeta.num_steps; s++) {
      const cell = document.createElement("div");
      cell.className = "scell" + (s % 4 === 0 ? " beat" : "");
      row.appendChild(cell);
      updateCell(cell, j, s);
      wireCell(cell, j, s);
    }
    g.appendChild(row);
  });
  const ph = document.createElement("div"); ph.className = "gridplayhead"; ph.id = "gridPlayhead";
  g.appendChild(ph);
}

function updateCell(cell, j, s) {
  const vel = gridState[j][s];
  cell.classList.toggle("on", vel > 0);
  let bar = cell.querySelector(".vbar");
  if (vel > 0) {
    if (!bar) { bar = document.createElement("div"); bar.className = "vbar"; cell.appendChild(bar); }
    bar.style.background = COLORS[gridMeta.instruments[j]];
    bar.style.height = (Math.max(0.05, vel) * 100) + "%";
    cell.title = `${LABELS[gridMeta.instruments[j]]}  step ${s}  vel ${vel.toFixed(2)}`;
  } else if (bar) { bar.remove(); cell.title = ""; }
}

let drag = { down: false, moved: false, j: 0, s: 0 };
function wireCell(cell, j, s) {
  cell.addEventListener("pointerdown", e => {
    e.preventDefault(); drag = { down: true, moved: false, j, s };
    try { cell.setPointerCapture(e.pointerId); } catch (err) {}
  });
  cell.addEventListener("pointermove", e => {
    if (!drag.down || drag.j !== j || drag.s !== s) return;
    const r = cell.getBoundingClientRect();
    let v = 1 - (e.clientY - r.top) / r.height;
    gridState[j][s] = Math.max(0.05, Math.min(1, v));
    drag.moved = true; updateCell(cell, j, s);
  });
  cell.addEventListener("pointerup", () => {
    if (!drag.down) return;
    if (!drag.moved) { gridState[j][s] = gridState[j][s] > 0 ? 0 : 0.8; updateCell(cell, j, s); }
    drag.down = false;
  });
}

async function playPattern() {
  if (!gridMeta || !lastData) return;
  stopAll();
  const tempo = lastData.tempo_bpm, N = gridMeta.num_steps, sp = gridMeta.steps_per_beat;
  const stepDur = 60 / (tempo * sp), bar = N * stepDur;
  const swingSteps = gridMeta.beat_type === "8th" ? [2, 6, 10, 14] : [1, 3, 5, 7, 9, 11, 13, 15];
  const bufs = {};
  for (const inst of INSTS) bufs[inst] = await loadBuffer(lastData.urls.oneshots[inst]);
  const t0 = audioCtx().currentTime + 0.06;
  patternT0 = t0; playingPattern = true; patternSources = [];
  const reps = Math.ceil(DUR / bar);
  for (let r = 0; r < reps; r++) {
    for (let j = 0; j < gridMeta.instruments.length; j++) {
      const inst = gridMeta.instruments[j];
      for (let s = 0; s < N; s++) {
        const vel = gridState[j][s]; if (vel <= 0) continue;
        let ts = s * stepDur;
        if (swingSteps.includes(s)) ts += (2 * gridMeta.swing[j] - 1) * stepDur;
        const when = t0 + r * bar + ts; if (when - t0 >= DUR) continue;
        const src = audioCtx().createBufferSource(); src.buffer = bufs[inst];
        const gain = audioCtx().createGain(); gain.gain.value = vel;
        src.connect(gain).connect(audioCtx().destination); src.start(when);
        patternSources.push(src);
      }
    }
  }
  const grid = document.getElementById("stepGrid");
  const gph = document.getElementById("gridPlayhead"); gph.style.display = "block";
  (function tick() {
    const e = audioCtx().currentTime - patternT0;
    if (e >= DUR || !playingPattern) { gph.style.display = "none"; playingPattern = false; return; }
    const phase = (e % bar) / bar, cellsW = grid.clientWidth - LABEL_W;
    gph.style.left = (LABEL_W + phase * cellsW) + "px";
    gridRaf = requestAnimationFrame(tick);
  })();
}

document.getElementById("playPattern").addEventListener("click", playPattern);
document.getElementById("stopPattern").addEventListener("click", stopAll);
document.getElementById("resetGrid").addEventListener("click", () => {
  if (lastData && lastData.quantized) renderStepGrid(lastData.quantized);
});

document.getElementById("playOriginal").addEventListener("click", () => play(lastData?.urls.loop));
document.getElementById("playRecon").addEventListener("click", () => play(lastData?.urls.recon));
document.getElementById("stopBtn").addEventListener("click", stopAll);
document.getElementById("downloadJson").addEventListener("click", () => {
  if (!lastData) return;
  const blob = new Blob([JSON.stringify(lastData, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "tao_params_" + lastData.cache_key + ".json"; a.click();
});
