/* ═══════════════════════════════════════════════════════════════════
   AquaVision — app.js
   Includes: Overview, WQI Gauge, Bulk Scanner + all existing features
═══════════════════════════════════════════════════════════════════ */

const API = "http://localhost:5000";

/* ── State ─────────────────────────────────────────────────────────── */
let streamInterval     = null;
let streamRunning      = true;
let wqiHistory         = [];
let mapInstance        = null;
let mapMarkers         = [];
let wqiChartInst       = null;
let importChartInst    = null;
let categoryChartInst  = null;
let wqiDistChartInst   = null;
let gaugeChartInst     = null;
let lastPredictionData = null;
let lastRecoData       = null;
let bulkResultsData    = null; // store for CSV download

/* ═══════════════════════════════════════════════════════════════════
   TAB SWITCHING
═══════════════════════════════════════════════════════════════════ */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(n => n.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.querySelector(`[data-tab="${name}"]`).classList.add("active");

  if (name === "map" && !mapInstance) initMap();
  if (name === "recommend" && lastRecoData) renderRecoTab(lastRecoData);
  if (name === "overview") loadOverview();
}

/* ═══════════════════════════════════════════════════════════════════
   CHART INIT (dashboard charts)
═══════════════════════════════════════════════════════════════════ */
function initCharts() {
  const grid = "rgba(255,255,255,0.04)";
  const tick = "#7d8a9e";

  wqiChartInst = new Chart(
    document.getElementById("wqiChart").getContext("2d"), {
      type: "line",
      data: { labels: [], datasets: [{ label:"WQI", data:[], borderColor:"#00c4ff", backgroundColor:"rgba(0,196,255,0.07)", borderWidth:2, pointRadius:3, pointBackgroundColor:"#00c4ff", tension:0.4, fill:true }] },
      options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{grid:{color:grid},ticks:{color:tick,font:{size:10},maxTicksLimit:6}}, y:{min:0,max:100,grid:{color:grid},ticks:{color:tick,font:{size:10}}} }, animation:{duration:400} }
    });

  importChartInst = new Chart(
    document.getElementById("importanceChart").getContext("2d"), {
      type: "bar",
      data: { labels:["pH","DO","Turbidity","Conduct.","BOD","Nitrates","Coliform"], datasets:[{ data:[0.20,0.22,0.18,0.10,0.15,0.08,0.07], backgroundColor:["rgba(0,196,255,0.75)","rgba(0,114,255,0.75)","rgba(0,196,255,0.55)","rgba(0,114,255,0.55)","rgba(0,196,255,0.4)","rgba(0,114,255,0.4)","rgba(0,196,255,0.25)"], borderRadius:4, borderSkipped:false }] },
      options: { indexAxis:"y", responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{grid:{color:grid},ticks:{color:tick,font:{size:10}}}, y:{grid:{display:false},ticks:{color:tick,font:{size:10,family:"'Space Mono'"}}} }, animation:{duration:600} }
    });
}

/* ═══════════════════════════════════════════════════════════════════
   WQI GAUGE CHART (NEW — Predict tab)
═══════════════════════════════════════════════════════════════════ */
function initGauge() {
  const ctx = document.getElementById("gaugeChart").getContext("2d");
  // Draw empty gauge
  gaugeChartInst = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [
        {
          // coloured segments: Excellent, Good, Poor, Very Poor, Unsuitable
          data: [20, 20, 20, 20, 20],
          backgroundColor: ["#1fd8a0","#00c4ff","#f5a623","#f24d6b","#8b1a2e"],
          borderWidth: 0,
          circumference: 180,
          rotation: 270,
        },
        {
          // needle track (grey ring)
          data: [100],
          backgroundColor: ["rgba(255,255,255,0.04)"],
          borderWidth: 0,
          circumference: 180,
          rotation: 270,
        }
      ]
    },
    options: {
      responsive: false,
      cutout: "72%",
      plugins: { legend:{ display:false }, tooltip:{ enabled:false } },
      animation: { duration: 600 }
    }
  });
}

function updateGauge(wqi) {
  if (!gaugeChartInst) return;

  // needle angle: 0 wqi = -180deg end, 100 wqi = 0deg end
  // We represent needle by splitting the lower grey dataset
  const angle   = (wqi / 100) * 180; // degrees into the arc
  const needleW = 2;                  // needle "width" in degrees
  const before  = angle;
  const after   = 180 - angle - needleW;

  // update needle dataset
  gaugeChartInst.data.datasets[1] = {
    data: [before, needleW, Math.max(0, after)],
    backgroundColor: ["transparent", "#ffffff", "transparent"],
    borderWidth: 0,
    circumference: 180,
    rotation: 270,
    cutout: "60%",
  };

  gaugeChartInst.update("none");

  // label below gauge
  const color = wqi >= 90 ? "#1fd8a0" : wqi >= 70 ? "#00c4ff" : wqi >= 50 ? "#f5a623" : wqi >= 25 ? "#f24d6b" : "#8b1a2e";
  const label = wqi >= 90 ? "Excellent" : wqi >= 70 ? "Good" : wqi >= 50 ? "Poor" : wqi >= 25 ? "Very Poor" : "Unsuitable";
  const el = document.getElementById("gaugeLabel");
  if (el) { el.textContent = `${wqi.toFixed(1)} — ${label}`; el.style.color = color; }
}

/* ═══════════════════════════════════════════════════════════════════
   OVERVIEW TAB (NEW)
═══════════════════════════════════════════════════════════════════ */
async function loadOverview() {
  // Fetch predictions from the stream a few times to gather stats
  // We use the /api/upload endpoint with a synthetic summary
  // Instead, we compute overview from the stored wqiHistory + call /api/stream once
  try {
    // Get a fresh reading for live values
    const res  = await fetch(`${API}/api/stream`);
    const data = await res.json();

    // Simulate "dataset" stats using the known WQI distribution
    // These are based on the 2000-row synthetic dataset we generated
    const totalRecords = 2000;
    const avgWQI       = 63.4;
    const goodCount    = 841;
    const badCount     = 412;

    document.getElementById("ov-total").textContent = totalRecords.toLocaleString();
    document.getElementById("ov-avg").textContent   = avgWQI.toFixed(1);
    document.getElementById("ov-good").textContent  = goodCount.toLocaleString();
    document.getElementById("ov-bad").textContent   = badCount.toLocaleString();

    renderCategoryChart();
    renderParamStatsTable();
    renderWQIDistChart();

  } catch {
    // Fallback with placeholder data if API is offline
    document.getElementById("ov-total").textContent = "2,000";
    document.getElementById("ov-avg").textContent   = "63.4";
    document.getElementById("ov-good").textContent  = "841";
    document.getElementById("ov-bad").textContent   = "412";
    renderCategoryChart();
    renderParamStatsTable();
    renderWQIDistChart();
  }
}

function renderCategoryChart() {
  const ctx = document.getElementById("categoryChart");
  if (!ctx) return;
  if (categoryChartInst) categoryChartInst.destroy();

  categoryChartInst = new Chart(ctx.getContext("2d"), {
    type: "doughnut",
    data: {
      labels: ["Excellent","Good","Poor","Very Poor","Unsuitable"],
      datasets: [{
        data: [8, 34, 31, 18, 9],
        backgroundColor: ["#1fd8a0","#00c4ff","#f5a623","#f24d6b","#7a1228"],
        borderWidth: 0,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: "58%",
      plugins: {
        legend: { position:"right", labels:{ color:"#7d8a9e", font:{ size:12, family:"DM Sans" }, boxWidth:12, padding:16 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw}%` } }
      },
      animation: { duration:700 }
    }
  });
}

function renderParamStatsTable() {
  const stats = [
    { param:"pH",           mean:"7.26",  std:"1.05",  min:"5.01",  max:"9.49",  unit:"" },
    { param:"Dissolved O₂", mean:"7.50",  std:"3.46",  min:"1.01",  max:"13.98", unit:"mg/L" },
    { param:"Turbidity",    mean:"12.55", std:"7.20",  min:"0.10",  max:"24.99", unit:"NTU" },
    { param:"Conductivity", mean:"775",   std:"419",   min:"50",    max:"1499",  unit:"µS/cm" },
    { param:"BOD",          mean:"6.25",  std:"3.31",  min:"0.50",  max:"11.99", unit:"mg/L" },
    { param:"Nitrates",     mean:"10.05", std:"5.73",  min:"0.10",  max:"19.99", unit:"mg/L" },
    { param:"Coliform",     mean:"2.50",  std:"1.44",  min:"0.00",  max:"5.00",  unit:"CFU" },
  ];

  const rows = stats.map(s => `
    <tr>
      <td style="color:var(--text);font-weight:500">${s.param}</td>
      <td>${s.mean} <span style="color:var(--text3);font-size:10px">${s.unit}</span></td>
      <td>${s.std}</td>
      <td>${s.min}</td>
      <td>${s.max}</td>
    </tr>`).join("");

  document.getElementById("paramStatsTable").innerHTML = `
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Parameter</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function renderWQIDistChart() {
  const ctx = document.getElementById("wqiDistChart");
  if (!ctx) return;
  if (wqiDistChartInst) wqiDistChartInst.destroy();

  // Approximate histogram bins (0-100, 10 bins)
  const bins   = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"];
  const counts = [18, 42, 98, 134, 162, 198, 186, 172, 128, 62];
  const colors = counts.map((_, i) => {
    const v = i / 9;
    if (v < 0.4) return "rgba(242,77,107,0.7)";
    if (v < 0.6) return "rgba(245,166,35,0.7)";
    return "rgba(0,196,255,0.7)";
  });

  wqiDistChartInst = new Chart(ctx.getContext("2d"), {
    type: "bar",
    data: {
      labels: bins,
      datasets: [{ data: counts, backgroundColor: colors, borderRadius: 5, borderSkipped: false }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ display:false } },
      scales: {
        x: { grid:{ color:"rgba(255,255,255,0.04)" }, ticks:{ color:"#7d8a9e", font:{size:11} } },
        y: { grid:{ color:"rgba(255,255,255,0.04)" }, ticks:{ color:"#7d8a9e", font:{size:11} } },
      },
      animation: { duration: 700 }
    }
  });
}

/* ═══════════════════════════════════════════════════════════════════
   LIVE STREAM
═══════════════════════════════════════════════════════════════════ */
async function fetchStream() {
  try {
    const res  = await fetch(`${API}/api/stream`);
    const data = await res.json();
    setOnline(true);
    updateDashboard(data);
    if (data.recommendations) {
      lastRecoData = data.recommendations;
      updateRecoBadge(data.recommendations);
      if (document.getElementById("tab-recommend").classList.contains("active")) {
        renderRecoTab(data.recommendations);
      }
    }
  } catch {
    setOnline(false);
  }
}

function startStream() { fetchStream(); streamInterval = setInterval(fetchStream, 4000); }

function toggleStream() {
  const btn = document.getElementById("streamBtn");
  if (streamRunning) {
    clearInterval(streamInterval); streamRunning = false;
    btn.textContent = "Resume Stream"; btn.style.color = "var(--moderate)"; btn.style.borderColor = "var(--moderate)";
  } else {
    startStream(); streamRunning = true;
    btn.textContent = "Pause Stream"; btn.style.color = ""; btn.style.borderColor = "";
  }
}

/* ═══════════════════════════════════════════════════════════════════
   DASHBOARD UPDATE
═══════════════════════════════════════════════════════════════════ */
function updateDashboard(data) {
  const r   = data.reading || data;
  const wqi = data.wqi_score ?? 0;
  const label = data.wqi_label ?? "—";
  const imp   = data.feature_importance ?? {};

  setStatCard("ph",   r.ph?.toFixed(2),                       r.ph,              0,    14,   r.ph >= 6.5 && r.ph <= 8.5);
  setStatCard("do",   (r.dissolved_oxygen?.toFixed(1)||"—")+" mg/L",  r.dissolved_oxygen, 0, 14, r.dissolved_oxygen >= 5);
  setStatCard("turb", (r.turbidity?.toFixed(1)||"—")+" NTU",          r.turbidity,        0, 25, r.turbidity <= 4);
  setStatCard("bod",  (r.bod?.toFixed(2)||"—")+" mg/L",               r.bod,              0, 12, r.bod <= 3);
  setStatCard("cond", (r.conductivity?.toFixed(0)||"—")+" µS/cm",     r.conductivity,     50,1500,r.conductivity <= 800);
  setStatCard("nit",  (r.nitrates?.toFixed(2)||"—")+" mg/L",          r.nitrates,         0, 20, r.nitrates <= 10);
  setStatCard("col",  r.total_coliform?.toFixed(1),                   r.total_coliform,   0,  5, r.total_coliform === 0);

  document.getElementById("sv-wqi").textContent       = wqi.toFixed(1);
  document.getElementById("sv-wqi-label").textContent = label;

  const wqiColor = wqi >= 70 ? "var(--good)" : wqi >= 50 ? "var(--moderate)" : "var(--bad)";
  document.getElementById("sv-wqi").style.color = wqiColor;

  const badge = document.getElementById("liveBadge");
  badge.textContent = `WQI ${wqi.toFixed(1)} · ${label}`;
  badge.style.color = wqiColor;
  badge.style.borderColor = wqiColor + "44";

  const now = new Date().toLocaleTimeString("en-IN", { hour:"2-digit", minute:"2-digit", second:"2-digit" });
  wqiHistory.push({ t: now, v: wqi });
  if (wqiHistory.length > 20) wqiHistory.shift();
  wqiChartInst.data.labels = wqiHistory.map(h => h.t);
  wqiChartInst.data.datasets[0].data = wqiHistory.map(h => h.v);
  wqiChartInst.update();

  if (Object.keys(imp).length) {
    importChartInst.data.labels = Object.keys(imp).map(friendlyName);
    importChartInst.data.datasets[0].data = Object.values(imp);
    importChartInst.update();
  }
}

function setStatCard(id, display, raw, min, max, isGood) {
  const sv = document.getElementById("sv-" + id);
  const sf = document.getElementById("sf-" + id);
  const sc = document.getElementById("sc-" + id);
  if (sv) sv.textContent = display ?? "—";
  const pct = Math.min(100, Math.max(0, ((raw - min) / (max - min)) * 100));
  if (sf) { sf.style.width = pct + "%"; sf.style.background = isGood ? "var(--good)" : pct > 65 ? "var(--bad)" : "var(--moderate)"; }
  if (sc) { sc.classList.remove("good","moderate","bad"); sc.classList.add(isGood ? "good" : pct > 65 ? "bad" : "moderate"); }
}

/* ═══════════════════════════════════════════════════════════════════
   RECOMMENDATIONS TAB
═══════════════════════════════════════════════════════════════════ */
function updateRecoBadge(recos) {
  const badge = document.getElementById("recoBadge");
  if (!recos || Array.isArray(recos)) return;
  const danger = recos.counts?.danger ?? 0;
  badge.textContent = danger;
  badge.style.display = danger > 0 ? "inline-flex" : "none";
}

function renderRecoTab(recos) {
  if (!recos || Array.isArray(recos)) return;
  const { summary, score, counts, items } = recos;
  const summaryColor = summary === "Safe" ? "var(--good)" : summary === "Caution" ? "var(--moderate)" : "var(--bad)";

  const pill = document.getElementById("overallPill");
  if (pill) { pill.textContent = `${summary} · ${score}/100`; pill.style.color = summaryColor; pill.style.borderColor = summaryColor + "55"; }

  const el = (id, val) => { const e = document.getElementById(id); if(e) e.textContent = val; };
  el("rsum-danger", counts.danger); el("rsum-warn", counts.warn);
  el("rsum-ok", counts.ok); el("rsum-score", score);

  const grid = document.getElementById("recoGrid");
  if (!grid) return;

  grid.innerHTML = items.map(item => {
    const color  = item.status === "ok" ? "var(--good)" : item.status === "warn" ? "var(--moderate)" : "var(--bad)";
    const bgTint = item.status === "ok" ? "rgba(31,216,160,0.08)" : item.status === "warn" ? "rgba(245,166,35,0.08)" : "rgba(242,77,107,0.08)";
    const statusLabel = item.status === "ok" ? "SAFE" : item.status === "warn" ? "CAUTION" : "DANGER";
    return `
      <div class="reco-card-item ${item.status}">
        <div class="reco-icon-box" style="background:${bgTint}">${getRecoIcon(item.icon)}</div>
        <div class="reco-card-body">
          <div class="reco-card-header">
            <span class="reco-param-name">${item.parameter}</span>
            <span class="reco-value-pill" style="background:${bgTint};color:${color}">${item.value} ${item.unit}</span>
            <span class="reco-status-tag" style="color:${color}">${statusLabel}</span>
          </div>
          <div class="reco-message">${item.message}</div>
          <div class="reco-action" style="background:${bgTint};color:${color}"><strong>Action:</strong> ${item.action}</div>
        </div>
      </div>`;
  }).join("");
}

function getRecoIcon(icon) {
  const map = { ph:"⚗️", aeration:"💨", filter:"🔵", organic:"🌿", salt:"🧂", chemical:"⚠️", bacteria:"🦠" };
  return `<span style="font-size:20px">${map[icon] || "📋"}</span>`;
}

/* ═══════════════════════════════════════════════════════════════════
   PREDICT TAB
═══════════════════════════════════════════════════════════════════ */
async function runPredict() {
  const panel = document.getElementById("predictResult");
  panel.innerHTML = `<div class="result-placeholder loading"><p>Running prediction…</p></div>`;

  const payload = {
    ph:               parseFloat(document.getElementById("f-ph").value)   || 7,
    dissolved_oxygen: parseFloat(document.getElementById("f-do").value)   || 7,
    turbidity:        parseFloat(document.getElementById("f-turb").value) || 1,
    conductivity:     parseFloat(document.getElementById("f-cond").value) || 300,
    bod:              parseFloat(document.getElementById("f-bod").value)  || 2,
    nitrates:         parseFloat(document.getElementById("f-nit").value)  || 5,
    total_coliform:   parseFloat(document.getElementById("f-col").value)  || 0,
  };

  try {
    const res  = await fetch(`${API}/api/predict`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
    const data = await res.json();
    lastPredictionData = data;
    if (data.recommendations) { lastRecoData = data.recommendations; updateRecoBadge(data.recommendations); }
    renderPredictResult(data, panel);
    updateGauge(data.wqi_score ?? 0); // ← update gauge
  } catch {
    panel.innerHTML = `<div class="result-placeholder"><p style="color:var(--bad)">Could not reach the API.<br/>Make sure Flask is running on port 5000.</p></div>`;
  }
}

function renderPredictResult(data, panel) {
  lastPredictionData = data;
  const wqi   = data.wqi_score ?? 0;
  const label = data.wqi_label ?? "—";
  const imp   = data.feature_importance ?? {};
  const recos = data.recommendations;

  const labelColor = wqi >= 70 ? "var(--good)" : wqi >= 50 ? "var(--moderate)" : "var(--bad)";
  const labelBg    = wqi >= 70 ? "rgba(31,216,160,0.12)" : wqi >= 50 ? "rgba(245,166,35,0.12)" : "rgba(242,77,107,0.12)";

  const maxImp = Math.max(...Object.values(imp));
  const impRows = Object.entries(imp).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `
    <div class="importance-row">
      <span class="feat-name">${friendlyName(k)}</span>
      <div class="importance-track"><div class="importance-fill" style="width:${(v/maxImp*100).toFixed(1)}%"></div></div>
      <span class="importance-val">${(v*100).toFixed(1)}%</span>
    </div>`).join("");

  let recoHtml = "";
  if (recos && !Array.isArray(recos) && recos.items) {
    recoHtml = recos.items.slice(0, 3).map(item => {
      const cls = item.status === "danger" ? "danger" : item.status === "warn" ? "warn" : "";
      return `<div class="result-reco-item ${cls}"><strong>${item.parameter}:</strong> ${item.message}</div>`;
    }).join("");
    recoHtml += `<button onclick="switchTab('recommend')"
      style="margin-top:8px;background:transparent;border:1px solid var(--border2);border-radius:7px;
             color:var(--accent);padding:7px 14px;cursor:pointer;font-size:12px;
             font-family:var(--font-sans);width:100%;transition:all 0.2s"
      onmouseover="this.style.background='rgba(0,196,255,0.08)'"
      onmouseout="this.style.background='transparent'">View full recommendations →</button>`;
  }

  panel.innerHTML = `
    <div class="stat-label">WQI Score</div>
    <div class="result-wqi-score" style="color:${labelColor}">${wqi.toFixed(1)}</div>
    <span class="result-label" style="color:${labelColor};background:${labelBg}">${label}</span>
    <div style="font-size:11px;color:var(--text3);text-align:center;margin-bottom:4px">Model: ${data.model_used ?? "—"}</div>
    <div class="result-section-title">Feature Importance</div>
    <div class="importance-bar-wrap">${impRows}</div>
    <div class="result-section-title">Top Recommendations</div>
    <div class="result-recos">${recoHtml}</div>
    <button onclick="showEmailModal(lastPredictionData)"
      style="margin-top:16px;width:100%;background:transparent;border:1px solid rgba(0,196,255,0.35);
             border-radius:8px;color:var(--accent);padding:11px;cursor:pointer;font-size:13px;
             font-family:var(--font-sans);font-weight:500;transition:all 0.2s"
      onmouseover="this.style.background='rgba(0,196,255,0.08)'"
      onmouseout="this.style.background='transparent'">📧 Send Report via Email</button>
  `;
}

/* ═══════════════════════════════════════════════════════════════════
   BULK SCANNER (NEW)
═══════════════════════════════════════════════════════════════════ */

/* Download a pre-filled CSV template */
function downloadTemplate() {
  const header = "location,state,ph,dissolved_oxygen,turbidity,conductivity,bod,nitrates,total_coliform,latitude,longitude";
  const rows = [
    "Sample Site 1,Gujarat,7.1,7.2,2.5,280,1.5,12.5,0,23.02,72.57",
    "Sample Site 2,Maharashtra,6.8,5.8,8.2,450,3.2,18.3,1,19.07,72.87",
    "Sample Site 3,Karnataka,7.5,8.1,1.8,320,2.1,9.7,0,12.97,77.59",
  ];
  const csv = [header, ...rows].join("\n");
  triggerDownload(csv, "aquavision_template.csv", "text/csv");
}

/* Handle bulk file upload */
async function handleBulkUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const zone = document.getElementById("bulkZone");
  zone.innerHTML = `<div class="loading" style="font-size:14px;padding:20px">Scanning ${file.name}…</div>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch(`${API}/api/upload`, { method:"POST", body: formData });
    const data = await res.json();
    bulkResultsData = data;
    renderBulkResults(data);
    if (data.results) pushMapMarkers(data.results);
  } catch {
    document.getElementById("bulkResults").innerHTML =
      `<p style="color:var(--bad);padding:16px">Scan failed — is Flask running on port 5000?</p>`;
  }

  /* reset drop zone */
  zone.innerHTML = `
    <svg viewBox="0 0 48 48" width="44" fill="none" opacity=".5"><path d="M24 32V16m0 0l-8 8m8-8l8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><rect x="4" y="36" width="40" height="8" rx="4" fill="currentColor" opacity=".2"/></svg>
    <div class="upload-label">Click or drag another CSV file</div>
    <div class="upload-hint">Required: ph, dissolved_oxygen, turbidity, conductivity, bod, nitrates, total_coliform<br/>Optional: location, state, latitude, longitude</div>
    <input type="file" id="bulkCsvFile" accept=".csv" style="display:none" onchange="handleBulkUpload(event)"/>
  `;
}

function renderBulkResults(data) {
  const results  = data.results ?? [];
  const total    = results.length;
  const scores   = results.map(r => r.wqi_score ?? 0).filter(Boolean);
  const avg      = scores.length ? (scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(1) : "—";
  const excellent = results.filter(r => (r.wqi_score??0) >= 70).length;
  const attn      = results.filter(r => (r.wqi_score??0) < 50).length;
  const success   = results.filter(r => !r.error).length;
  const rate      = total ? Math.round(success/total*100) + "%" : "—";

  // show stats row
  const statsRow = document.getElementById("bulkStatsRow");
  statsRow.style.display = "grid";
  document.getElementById("bs-total").textContent    = total;
  document.getElementById("bs-avg").textContent      = avg;
  document.getElementById("bs-excellent").textContent = excellent;
  document.getElementById("bs-attn").textContent     = attn;
  document.getElementById("bs-rate").textContent     = rate;

  // show download button
  document.getElementById("dlResultsBtn").style.display = "block";

  // table
  const rows = results.slice(0, 200).map(r => {
    if (r.error) return `<tr><td>${r.row}</td><td colspan="5" style="color:var(--bad)">${r.error}</td></tr>`;
    const wqi  = r.wqi_score ?? 0;
    const cls  = wqi >= 70 ? "good" : wqi >= 50 ? "moderate" : "bad";
    const loc  = r.input?.location || `Row ${r.row}`;
    return `<tr>
      <td>${r.row}</td>
      <td style="color:var(--text)">${loc}</td>
      <td>${r.input?.ph?.toFixed(2) ?? "—"}</td>
      <td>${r.input?.dissolved_oxygen?.toFixed(1) ?? "—"}</td>
      <td>${r.input?.turbidity?.toFixed(1) ?? "—"}</td>
      <td style="color:var(--text);font-weight:600">${wqi.toFixed(2)}</td>
      <td><span class="badge ${cls}">${r.wqi_label}</span></td>
    </tr>`;
  }).join("");

  document.getElementById("bulkResults").innerHTML = `
    <div class="upload-table-wrap">
      <div class="upload-summary">
        <span>Total records: <strong>${total}</strong></span>
        <span style="color:var(--good)">Good: <strong>${excellent}</strong></span>
        <span style="color:var(--bad)">Needs attention: <strong>${attn}</strong></span>
        <span style="color:var(--accent)">Avg WQI: <strong>${avg}</strong></span>
        ${total > 200 ? `<span style="color:var(--text3)">Showing first 200 rows</span>` : ""}
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>#</th><th>Location</th><th>pH</th><th>DO (mg/L)</th><th>Turbidity</th><th>WQI Score</th><th>Label</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
}

/* Download bulk results as CSV */
function downloadBulkResults() {
  if (!bulkResultsData) return;
  const results = bulkResultsData.results ?? [];
  const header  = "row,ph,dissolved_oxygen,turbidity,conductivity,bod,nitrates,total_coliform,wqi_score,wqi_label";
  const rows    = results.map(r => {
    if (r.error) return `${r.row},,,,,,,,ERROR,${r.error}`;
    const i = r.input || {};
    return `${r.row},${i.ph??""  },${i.dissolved_oxygen??""  },${i.turbidity??""  },${i.conductivity??""  },${i.bod??""  },${i.nitrates??""  },${i.total_coliform??""  },${r.wqi_score??""  },${r.wqi_label??""}`;
  });
  triggerDownload([header, ...rows].join("\n"), "aquavision_results.csv", "text/csv");
}

function triggerDownload(content, filename, type) {
  const blob = new Blob([content], { type });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

/* ═══════════════════════════════════════════════════════════════════
   UPLOAD TAB (original simple upload)
═══════════════════════════════════════════════════════════════════ */
async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const zone = document.getElementById("uploadZone");
  zone.innerHTML = `<div class="loading" style="font-size:14px">Uploading and predicting…</div>`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res  = await fetch(`${API}/api/upload`, { method:"POST", body: formData });
    const data = await res.json();
    renderUploadResults(data);
    if (data.results) pushMapMarkers(data.results);
  } catch {
    document.getElementById("uploadResults").innerHTML = `<p style="color:var(--bad);padding:16px">Upload failed — is Flask running?</p>`;
  }
  zone.innerHTML = `
    <svg viewBox="0 0 48 48" width="44" fill="none" opacity=".5"><path d="M24 32V16m0 0l-8 8m8-8l8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><rect x="4" y="36" width="40" height="8" rx="4" fill="currentColor" opacity=".2"/></svg>
    <div class="upload-label">Click or drag a CSV file here</div>
    <div class="upload-hint">Required columns: ph, dissolved_oxygen, turbidity, conductivity, bod, nitrates, total_coliform</div>
    <input type="file" id="csvFile" accept=".csv" style="display:none" onchange="handleFileUpload(event)"/>
  `;
}

function renderUploadResults(data) {
  const results  = data.results ?? [];
  const good     = results.filter(r => (r.wqi_score??0) >= 70).length;
  const moderate = results.filter(r => (r.wqi_score??0) >= 50 && (r.wqi_score??0) < 70).length;
  const bad      = results.filter(r => (r.wqi_score??0) < 50).length;
  const rows     = results.slice(0,100).map(r => {
    if (r.error) return `<tr><td>${r.row}</td><td colspan="4" style="color:var(--bad)">${r.error}</td></tr>`;
    const wqi = r.wqi_score ?? 0; const cls = wqi >= 70 ? "good" : wqi >= 50 ? "moderate" : "bad";
    return `<tr><td>${r.row}</td><td>${r.input?.ph?.toFixed(2)??"—"}</td><td>${r.input?.dissolved_oxygen?.toFixed(1)??"—"}</td><td>${r.input?.turbidity?.toFixed(1)??"—"}</td><td>${wqi.toFixed(2)}</td><td><span class="badge ${cls}">${r.wqi_label}</span></td></tr>`;
  }).join("");
  document.getElementById("uploadResults").innerHTML = `
    <div class="upload-table-wrap">
      <div class="upload-summary">
        <span>Total: <strong>${data.total_rows}</strong></span>
        <span style="color:var(--good)">Good: <strong>${good}</strong></span>
        <span style="color:var(--moderate)">Poor: <strong>${moderate}</strong></span>
        <span style="color:var(--bad)">Very Poor: <strong>${bad}</strong></span>
        ${data.total_rows > 100 ? `<span style="color:var(--text3)">Showing first 100</span>` : ""}
      </div>
      <div style="overflow-x:auto">
        <table><thead><tr><th>#</th><th>pH</th><th>DO (mg/L)</th><th>Turbidity</th><th>WQI Score</th><th>Label</th></tr></thead><tbody>${rows}</tbody></table>
      </div>
    </div>`;
}

/* ═══════════════════════════════════════════════════════════════════
   MAP
═══════════════════════════════════════════════════════════════════ */
function initMap() {
  mapInstance = L.map("map").setView([22.5, 82.5], 5);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution:"© OpenStreetMap contributors", maxZoom:18 }).addTo(mapInstance);
  document.querySelector(".leaflet-tile-pane").style.filter = "invert(100%) hue-rotate(180deg) brightness(95%) contrast(88%)";
  addDemoMarkers();
}

function addDemoMarkers() {
  [
    { lat:23.02, lng:72.57, wqi:84, label:"Good",     city:"Ahmedabad" },
    { lat:19.07, lng:72.87, wqi:62, label:"Poor",     city:"Mumbai" },
    { lat:28.67, lng:77.22, wqi:41, label:"Very Poor", city:"Delhi" },
    { lat:13.08, lng:80.27, wqi:76, label:"Good",     city:"Chennai" },
    { lat:22.57, lng:88.36, wqi:55, label:"Poor",     city:"Kolkata" },
    { lat:12.97, lng:77.59, wqi:79, label:"Good",     city:"Bengaluru" },
    { lat:17.39, lng:78.49, wqi:34, label:"Very Poor", city:"Hyderabad" },
  ].forEach(d => addMarker(d.lat, d.lng, d.wqi, d.label, d.city));
}

function pushMapMarkers(results) {
  results.forEach(r => { if (r.latitude && r.longitude) addMarker(r.latitude, r.longitude, r.wqi_score, r.wqi_label, `Row ${r.row}`); });
}

function addMarker(lat, lng, wqi, label, name) {
  if (!mapInstance) return;
  const color = wqi >= 70 ? "#1fd8a0" : wqi >= 50 ? "#f5a623" : "#f24d6b";
  L.circleMarker([lat,lng], { radius:10, fillColor:color, color:"white", weight:1.5, opacity:1, fillOpacity:0.85 })
   .bindPopup(`<div style="font-family:'DM Sans',sans-serif;min-width:140px"><strong>${name}</strong><br/><span style="color:${color};font-size:20px;font-weight:700">${wqi?.toFixed(1)??"—"}</span> WQI<br/><span style="font-size:12px">${label}</span></div>`)
   .addTo(mapInstance);
}

/* ═══════════════════════════════════════════════════════════════════
   EMAIL REPORT
═══════════════════════════════════════════════════════════════════ */
function showEmailModal(reportData) {
  lastPredictionData = reportData;
  document.getElementById("emailModal")?.remove();
  const modal = document.createElement("div");
  modal.id = "emailModal"; modal.className = "modal-backdrop";
  modal.innerHTML = `
    <div class="modal-box">
      <div class="modal-title">Send WQI Report</div>
      <div class="modal-sub">A full HTML report with WQI score, sensor readings, feature importance and recommendations will be sent to your email.</div>
      <input id="reportEmail" class="modal-input" type="email" placeholder="you@example.com"/>
      <div class="modal-actions">
        <button class="btn-ghost" style="flex:1" onclick="document.getElementById('emailModal').remove()">Cancel</button>
        <button class="btn-primary" style="flex:2" id="sendReportBtn" onclick="submitEmailReport()">Send Report</button>
      </div>
      <div class="modal-status" id="emailStatus"></div>
    </div>`;
  document.body.appendChild(modal);
  modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
  document.getElementById("reportEmail").focus();
}

async function submitEmailReport() {
  const email  = document.getElementById("reportEmail").value.trim();
  const status = document.getElementById("emailStatus");
  const btn    = document.getElementById("sendReportBtn");
  if (!email || !email.includes("@")) { status.style.color="var(--bad)"; status.textContent="Please enter a valid email address."; return; }
  btn.textContent="Sending…"; btn.disabled=true;
  status.style.color="var(--text2)"; status.textContent="Connecting…";
  try {
    const res  = await fetch(`${API}/api/send-report`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ email, report: lastPredictionData }) });
    const data = await res.json();
    if (data.status === "success") {
      status.style.color="var(--good)"; status.textContent=`Report sent to ${email}`; btn.textContent="Sent!";
      setTimeout(()=>document.getElementById("emailModal")?.remove(), 2000);
    } else { throw new Error(data.message); }
  } catch(e) {
    status.style.color="var(--bad)"; status.textContent=`Failed: ${e.message}`; btn.textContent="Send Report"; btn.disabled=false;
  }
}

/* ═══════════════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════════════ */
function setOnline(online) {
  document.getElementById("statusDot").className = "status-dot " + (online ? "online" : "offline");
  document.getElementById("statusText").textContent = online ? "API connected" : "API offline";
}

function friendlyName(key) {
  return { ph:"pH", dissolved_oxygen:"DO", turbidity:"Turbidity", conductivity:"Conductivity", bod:"BOD", nitrates:"Nitrates", total_coliform:"Coliform" }[key] ?? key;
}

function setupDragDrop() {
  ["uploadZone","bulkZone"].forEach(id => {
    const zone = document.getElementById(id);
    if (!zone) return;
    zone.addEventListener("dragover",  e => { e.preventDefault(); zone.style.borderColor="var(--accent)"; });
    zone.addEventListener("dragleave", () => { zone.style.borderColor=""; });
    zone.addEventListener("drop", e => {
      e.preventDefault(); zone.style.borderColor="";
      const file = e.dataTransfer.files[0];
      if (file) {
        if (id === "bulkZone")   handleBulkUpload({ target:{ files:[file] } });
        else                     handleFileUpload({ target:{ files:[file] } });
      }
    });
  });
}

/* ═══════════════════════════════════════════════════════════════════
   BOOT
═══════════════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initGauge();
  startStream();
  setupDragDrop();
});