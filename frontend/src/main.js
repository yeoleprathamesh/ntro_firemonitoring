import './style.css';
import L from 'leaflet';
import Chart from 'chart.js/auto';

// Ensure Leaflet default marker icons resolve properly if needed
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

/* ==========================================================================
   State & Configuration
   ========================================================================== */
const state = {
  currentLayerMode: 'dark', // 'dark' | 'satellite'
  activeClassFilter: 'all', // 'all' | 'industrial' | 'vegetation'
  minFRP: 0,
  minRecurrence: 1,
  limit: 5000,
  audioEnabled: true,
  currentSimMarker: null,
  recentIntercepts: [],
  featureChart: null,
};

// Tactical Audio Synthesizer (Web Audio API)
let audioCtx = null;
function playTacticalAlertSound(isCritical = false) {
  if (!state.audioEnabled) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = isCritical ? 'sawtooth' : 'sine';
    osc.frequency.setValueAtTime(isCritical ? 880 : 540, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(isCritical ? 1760 : 720, audioCtx.currentTime + 0.12);

    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
  } catch (e) {
    console.debug('Audio context not allowed yet:', e);
  }
}

/* ==========================================================================
   Map Setup & Layers
   ========================================================================== */
const map = L.map('map', {
  center: [21.8, 82.5],
  zoom: 5,
  minZoom: 4,
  maxZoom: 18,
  preferCanvas: true,
  zoomControl: false,
});

// Custom zoom control in bottom right
L.control.zoom({ position: 'bottomright' }).addTo(map);

// Base Tile Layers (Free, High-Performance, Zero-Key ArcGIS REST Services)
const darkBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles &copy; Esri, DeLorme, NAVTEQ',
  maxZoom: 16,
});

const darkLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16,
});

const darkTileGroup = L.layerGroup([darkBase, darkLabels]);

const satTileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles &copy; Esri & Earthstar Geographics',
  maxZoom: 18,
});

const satLabelsLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 18,
});

const satTileGroup = L.layerGroup([satTileLayer, satLabelsLayer]);

// Initial Layer: Clean Dark Tactical Canvas (No API Key Required)
darkTileGroup.addTo(map);

// Layer group for thermal anomalies
const anomalyLayerGroup = L.layerGroup().addTo(map);
const simLayerGroup = L.layerGroup().addTo(map);

/* ==========================================================================
   Live UTC Clock
   ========================================================================== */
function initClock() {
  const clockEl = document.getElementById('utc-clock');
  function update() {
    const now = new Date();
    const h = String(now.getUTCHours()).padStart(2, '0');
    const m = String(now.getUTCMinutes()).padStart(2, '0');
    const s = String(now.getUTCSeconds()).padStart(2, '0');
    if (clockEl) clockEl.textContent = `${h}:${m}:${s} UTC`;
  }
  update();
  setInterval(update, 1000);
}

/* ==========================================================================
   Telemetry & Backend Communication
   ========================================================================== */
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('API offline');
    const data = await res.json();
    
    document.getElementById('server-status-dot').style.background = 'var(--green-vegetation)';
    document.getElementById('server-status-dot').classList.add('status-pulse');
    document.getElementById('server-status-text').textContent = 'ONLINE (STABLE)';
    document.getElementById('header-records-count').textContent = data.total_records.toLocaleString();
  } catch (err) {
    document.getElementById('server-status-dot').style.background = 'var(--red-industrial)';
    document.getElementById('server-status-dot').classList.remove('status-pulse');
    document.getElementById('server-status-text').textContent = 'OFFLINE';
  }
}

async function loadAnalytics() {
  try {
    const [statsRes, modelRes, hotspotsRes] = await Promise.all([
      fetch('/api/stats/overview'),
      fetch('/api/model/info'),
      fetch('/api/fires/hotspots?top_n=15')
    ]);

    if (statsRes.ok) {
      const stats = await statsRes.json();
      document.getElementById('kpi-industrial-pct').textContent = `${stats.industrial_percentage}%`;
      document.getElementById('kpi-max-frp').textContent = `${stats.max_frp_recorded_mw} MW`;
      document.getElementById('kpi-recurring-sites').textContent = stats.high_recurrence_sites_count.toLocaleString();
    }

    if (modelRes.ok) {
      const modelInfo = await modelRes.json();
      renderFeatureChart(modelInfo.feature_importances);
    }

    if (hotspotsRes.ok) {
      const hotspotsData = await hotspotsRes.json();
      renderHotspotsList(hotspotsData.hotspots);
    }
  } catch (err) {
    console.error('Error loading analytics:', err);
  }
}

function renderHotspotsList(hotspots) {
  const container = document.getElementById('hotspots-list');
  if (!container || !hotspots) return;

  container.innerHTML = hotspots.map((h, i) => `
    <div class="hotspot-item" data-lon="${h.center_coordinates[0]}" data-lat="${h.center_coordinates[1]}">
      <div class="hotspot-item-left">
        <span class="hotspot-coords">#${i + 1} Cluster [${h.center_coordinates[1].toFixed(2)}°N, ${h.center_coordinates[0].toFixed(2)}°E]</span>
        <span class="hotspot-metrics">Detections: <strong>${h.detection_count}</strong> | Mean FRP: <strong>${h.mean_frp_mw} MW</strong></span>
      </div>
      <button class="btn-micro">FLY TO</button>
    </div>
  `).join('');

  container.querySelectorAll('.hotspot-item').forEach(item => {
    item.addEventListener('click', () => {
      const lat = parseFloat(item.getAttribute('data-lat'));
      const lon = parseFloat(item.getAttribute('data-lon'));
      map.flyTo([lat, lon], 12, { duration: 1.5 });
      highlightLocationOnMap(lat, lon, 'Industrial Hub Target');
    });
  });
}

function renderFeatureChart(importances) {
  const canvas = document.getElementById('features-chart');
  if (!canvas || !importances) return;

  if (state.featureChart) {
    state.featureChart.destroy();
  }

  const labels = Object.keys(importances);
  const values = Object.values(importances);

  state.featureChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Weight',
        data: values,
        backgroundColor: 'rgba(0, 242, 254, 0.45)',
        borderColor: '#00f2fe',
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10, 15, 26, 0.95)',
          titleColor: '#00f2fe',
          bodyColor: '#ffffff',
          borderColor: 'rgba(0, 242, 254, 0.3)',
          borderWidth: 1,
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { family: 'Inter', size: 10.5 } }
        }
      }
    }
  });
}

/* ==========================================================================
   Load Thermal Points GeoJSON
   ========================================================================== */
async function loadThermalAnomalies() {
  const badge = document.getElementById('points-loaded-badge');
  badge.textContent = 'Scanning Orbit...';

  try {
    const url = `/api/fires/geojson?classification=${state.activeClassFilter}&min_frp=${state.minFRP}&min_recurrence=${state.minRecurrence}&limit=${state.limit}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load anomaly GeoJSON');
    const geojson = await res.json();

    anomalyLayerGroup.clearLayers();

    const features = geojson.features || [];
    badge.textContent = `Active Detections: ${features.length.toLocaleString()}`;

    // Render features with canvas CircleMarkers for blazing speed
    features.forEach(f => {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const isInd = p.class_id === 1;
      const isFlare = p.frp >= 60;

      // Dynamic sizing based on Fire Radiative Power (MW)
      const radius = Math.min(16, Math.max(3.5, Math.log2(p.frp + 2) * 2.2));

      const marker = L.circleMarker([lat, lon], {
        radius: radius,
        fillColor: isFlare ? '#f59e0b' : (isInd ? '#ff3366' : '#10b981'),
        color: isFlare ? '#fef08a' : (isInd ? '#fda4af' : '#6ee7b7'),
        weight: isFlare ? 1.8 : 1,
        opacity: 0.9,
        fillOpacity: isFlare ? 0.85 : 0.65,
      });

      // Bind rich tactical popup
      marker.bindPopup(`
        <div class="tactical-popup">
          <div class="popup-header">
            <span class="popup-tag" style="color: ${isInd ? '#ff4d79' : '#34d399'}">
              ${isInd ? '🏭 INDUSTRIAL ANOMALY' : '🌲 VEGETATION / WILDFIRE'}
            </span>
            <span class="risk-badge ${isFlare ? 'badge-critical' : (isInd ? 'badge-high' : 'badge-low')}">
              ${isFlare ? 'CRITICAL FLARE' : (isInd ? 'HIGH CONFIDENCE' : 'ROUTINE')}
            </span>
          </div>
          <table class="popup-table">
            <tr><td>COORDINATES</td><td>${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</td></tr>
            <tr><td>DATE RECORDED</td><td>${p.date || 'Historical Pass'}</td></tr>
            <tr><td>THERMAL POWER (FRP)</td><td><strong style="color: #f59e0b">${p.frp} MW</strong></td></tr>
            <tr><td>BRIGHTNESS TEMP</td><td>${p.temp} K</td></tr>
            <tr><td>SPATIAL RECURRENCE</td><td>${p.recurrence}x persistent</td></tr>
            <tr><td>INDUSTRIAL PROB</td><td>${(p.prob_industrial * 100).toFixed(1)}%</td></tr>
          </table>
          <div class="popup-footer-expl">
            ${isInd ? 'Continuous industrial high-temperature emitter (smelter, power plant, refinery flaring).' : 'Transient biomass / agricultural burning or wildfire signature.'}
          </div>
          <button class="tactical-btn secondary full-width btn-inspect-here" data-lat="${lat}" data-lon="${lon}" data-temp="${p.temp}" data-b31="${p.bright_t31 || (p.temp - 40)}" data-frp="${p.frp}" data-rec="${p.recurrence || 1}">
            🎯 LOAD INTO INSPECTOR
          </button>
        </div>
      `, { maxWidth: 300 });

      marker.addTo(anomalyLayerGroup);
    });

  } catch (err) {
    console.error('Failed to load points:', err);
    badge.textContent = 'Error Fetching Feed';
  }
}

// Delegated listener for popup "LOAD INTO INSPECTOR"
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.btn-inspect-here');
  if (btn) {
    const lat = parseFloat(btn.getAttribute('data-lat'));
    const lon = parseFloat(btn.getAttribute('data-lon'));
    const frp = parseFloat(btn.getAttribute('data-frp'));
    const temp = parseFloat(btn.getAttribute('data-temp'));
    const b31 = parseFloat(btn.getAttribute('data-b31'));
    const rec = parseInt(btn.getAttribute('data-rec'));

    // Switch to Inspector Tab
    switchTab('tab-inspector');

    // Fill inputs
    document.getElementById('inp-lat').value = lat;
    document.getElementById('inp-lon').value = lon;
    if (!isNaN(frp)) document.getElementById('inp-frp').value = frp;
    if (!isNaN(temp)) document.getElementById('inp-b21').value = temp;
    if (!isNaN(b31)) document.getElementById('inp-b31').value = b31;
    if (!isNaN(rec)) document.getElementById('inp-recurrence').value = rec;

    // Trigger run prediction
    document.getElementById('btn-submit-prediction').click();
  }
});

/* ==========================================================================
   Highlight Pin on Map
   ========================================================================== */
function highlightLocationOnMap(lat, lon, label = 'Target Location') {
  simLayerGroup.clearLayers();

  const ping = L.circleMarker([lat, lon], {
    radius: 18,
    fillColor: '#00f2fe',
    color: '#ffffff',
    weight: 2,
    opacity: 1,
    fillOpacity: 0.35,
  }).addTo(simLayerGroup);

  const centerDot = L.circleMarker([lat, lon], {
    radius: 5,
    fillColor: '#ffffff',
    color: '#00f2fe',
    weight: 2,
    fillOpacity: 1,
  }).addTo(simLayerGroup);

  centerDot.bindPopup(`<strong>${label}</strong><br/>${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`).openPopup();
}

/* ==========================================================================
   Live Satellite Pass Simulator
   ========================================================================== */
async function triggerSimulation() {
  const scenarioSelect = document.getElementById('sim-scenario-select');
  const scenario = scenarioSelect ? scenarioSelect.value : 'industrial_cluster';

  try {
    const res = await fetch(`/api/simulate?scenario=${scenario}`, { method: 'POST' });
    if (!res.ok) throw new Error('Simulation failed');
    const detection = await res.json();

    playTacticalAlertSound(detection.risk_level === 'CRITICAL');
    displaySimulatedDetection(detection);
  } catch (err) {
    console.error('Error during simulation:', err);
  }
}

function displaySimulatedDetection(d) {
  // Update latest alert card
  document.getElementById('alert-class-name').textContent = d.class_name;
  document.getElementById('alert-class-name').style.color = d.color;
  document.getElementById('alert-confidence').textContent = `${(d.confidence_score * 100).toFixed(1)}%`;
  document.getElementById('alert-coordinates').textContent = `${d.latitude.toFixed(4)}°N, ${d.longitude.toFixed(4)}°E`;
  document.getElementById('alert-frp').textContent = `${d.features_used.frp?.toFixed(1) || '--'} MW`;
  document.getElementById('alert-temp').textContent = `${d.features_used.brightness?.toFixed(1) || '--'} K`;
  document.getElementById('alert-recurrence').textContent = `${d.recurrence_count} detections`;

  const riskBadge = document.getElementById('alert-risk-badge');
  riskBadge.textContent = d.risk_level;
  riskBadge.className = `risk-badge badge-${d.risk_level.toLowerCase()}`;

  const pct = (d.probability_industrial * 100).toFixed(1);
  document.getElementById('alert-prob-value').textContent = `${pct}%`;
  document.getElementById('alert-prob-fill').style.width = `${pct}%`;
  document.getElementById('alert-explanation-text').textContent = d.explanation;

  const flyBtn = document.getElementById('btn-fly-to-latest');
  flyBtn.style.display = 'block';
  flyBtn.onclick = () => {
    map.flyTo([d.latitude, d.longitude], 12, { duration: 1.5 });
  };

  // Focus map on the target
  map.flyTo([d.latitude, d.longitude], 10, { duration: 1.2 });
  highlightLocationOnMap(d.latitude, d.longitude, `${d.class_name} (${d.risk_level})`);

  // Append to recent feed
  state.recentIntercepts.unshift({
    time: new Date().toLocaleTimeString(),
    ...d
  });
  if (state.recentIntercepts.length > 20) state.recentIntercepts.pop();
  renderRecentFeed();
}

function renderRecentFeed() {
  const container = document.getElementById('stream-feed-list');
  const countTag = document.getElementById('feed-count-tag');
  if (!container) return;

  countTag.textContent = `${state.recentIntercepts.length} logged`;

  if (state.recentIntercepts.length === 0) {
    container.innerHTML = `
      <div class="empty-feed-placeholder">
        <div class="radar-pulse-ring"></div>
        <span>No simulated anomalies ingested yet this session.</span>
      </div>`;
    return;
  }

  container.innerHTML = state.recentIntercepts.map((item, idx) => `
    <div class="feed-item" data-idx="${idx}">
      <div class="feed-item-left">
        <span class="feed-item-title" style="color: ${item.color}">
          ${item.class_id === 1 ? '🏭' : '🌲'} ${item.class_name}
        </span>
        <span class="feed-item-coords">${item.latitude.toFixed(3)}°N, ${item.longitude.toFixed(3)}°E • ${item.time}</span>
      </div>
      <div class="feed-item-right">
        <span class="risk-badge badge-${item.risk_level.toLowerCase()}">${item.risk_level}</span>
        <span class="feed-item-coords monospace">${item.features_used.frp?.toFixed(1) || '--'} MW</span>
      </div>
    </div>
  `).join('');

  container.querySelectorAll('.feed-item').forEach(el => {
    el.addEventListener('click', () => {
      const idx = parseInt(el.getAttribute('data-idx'));
      const item = state.recentIntercepts[idx];
      if (item) {
        map.flyTo([item.latitude, item.longitude], 12, { duration: 1.5 });
        highlightLocationOnMap(item.latitude, item.longitude, item.class_name);
      }
    });
  });
}

/* ==========================================================================
   Manual Inspector (ML Sandbox Form)
   ========================================================================== */
async function handleManualPrediction(e) {
  e.preventDefault();

  const lat = parseFloat(document.getElementById('inp-lat').value);
  const lon = parseFloat(document.getElementById('inp-lon').value);
  const b21 = parseFloat(document.getElementById('inp-b21').value);
  const b31 = parseFloat(document.getElementById('inp-b31').value);
  const frp = parseFloat(document.getElementById('inp-frp').value);
  const conf = parseFloat(document.getElementById('inp-confidence').value);
  const daynight = document.getElementById('inp-daynight').value;
  const recInput = document.getElementById('inp-recurrence').value;
  const recurrence = recInput ? parseInt(recInput) : null;
  const threshold = parseFloat(document.getElementById('inp-threshold').value);

  const payload = {
    latitude: lat,
    longitude: lon,
    brightness: b21,
    bright_t31: b31,
    frp: frp,
    confidence: conf,
    daynight: daynight,
    acq_time: daynight === 'N' ? 1930 : 1130,
    recurrence_count: recurrence,
  };

  try {
    const res = await fetch(`/api/predict?threshold=${threshold}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('ML inference failed');
    const result = await res.json();

    // Render results
    const outBox = document.getElementById('prediction-output-box');
    outBox.style.display = 'flex';

    document.getElementById('pred-class-name').textContent = result.class_name;
    document.getElementById('pred-class-name').style.color = result.color;

    const riskBadge = document.getElementById('pred-risk-badge');
    riskBadge.textContent = result.risk_level;
    riskBadge.className = `risk-badge badge-${result.risk_level.toLowerCase()}`;

    document.getElementById('pred-prob-ind').textContent = `${(result.probability_industrial * 100).toFixed(1)}%`;
    document.getElementById('pred-prob-veg').textContent = `${(result.probability_vegetation * 100).toFixed(1)}%`;
    document.getElementById('pred-explanation').textContent = result.explanation;

    playTacticalAlertSound(result.risk_level === 'CRITICAL');
    map.flyTo([lat, lon], 11, { duration: 1.2 });
    highlightLocationOnMap(lat, lon, `ML Classification: ${result.class_name}`);

  } catch (err) {
    console.error('Error running manual prediction:', err);
    alert('Error connecting to ML inference endpoint.');
  }
}

// Map Click -> Auto-fill manual inspector
map.on('click', (e) => {
  const { lat, lng } = e.latlng;
  const latInp = document.getElementById('inp-lat');
  const lonInp = document.getElementById('inp-lon');
  if (latInp && lonInp) {
    latInp.value = lat.toFixed(4);
    lonInp.value = lng.toFixed(4);
  }
});

/* ==========================================================================
   Tab Navigation & UI Event Handlers
   ========================================================================== */
function switchTab(targetTabId) {
  document.querySelectorAll('.sidebar-tab').forEach(tab => {
    const isActive = tab.getAttribute('data-tab') === targetTabId;
    tab.classList.toggle('active', isActive);
    tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === targetTabId);
  });
}

function initUIEvents() {
  // Tab Switching
  document.querySelectorAll('.sidebar-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      switchTab(tab.getAttribute('data-tab'));
    });
  });

  // Layer Mode Toggle (Dark Matter vs Satellite)
  const layerBtn = document.getElementById('btn-toggle-layer');
  const layerText = document.getElementById('layer-mode-text');
  layerBtn.addEventListener('click', () => {
    if (state.currentLayerMode === 'dark') {
      map.removeLayer(darkTileGroup);
      satTileGroup.addTo(map);
      state.currentLayerMode = 'satellite';
      layerText.textContent = 'DARK MAP VIEW';
    } else {
      map.removeLayer(satTileGroup);
      darkTileGroup.addTo(map);
      state.currentLayerMode = 'dark';
      layerText.textContent = 'SATELLITE VIEW';
    }
  });

  // Sound Toggle
  const soundBtn = document.getElementById('btn-toggle-sound');
  const soundIcon = document.getElementById('sound-icon');
  const soundText = document.getElementById('sound-text');
  soundBtn.addEventListener('click', () => {
    state.audioEnabled = !state.audioEnabled;
    soundIcon.textContent = state.audioEnabled ? '🔊' : '🔇';
    soundText.textContent = state.audioEnabled ? 'AUDIO ON' : 'AUDIO MUTED';
    soundBtn.classList.toggle('active', state.audioEnabled);
  });

  // Classification Segment Buttons
  document.querySelectorAll('#filter-class-segment .segment-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#filter-class-segment .segment-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeClassFilter = btn.getAttribute('data-class');
      loadThermalAnomalies();
    });
  });

  // FRP Slider
  const frpSlider = document.getElementById('slider-frp');
  const frpVal = document.getElementById('slider-frp-val');
  frpSlider.addEventListener('input', (e) => {
    state.minFRP = parseFloat(e.target.value);
    frpVal.textContent = `${state.minFRP} MW`;
  });
  frpSlider.addEventListener('change', () => loadThermalAnomalies());

  // Recurrence Slider
  const recSlider = document.getElementById('slider-recurrence');
  const recVal = document.getElementById('slider-recurrence-val');
  recSlider.addEventListener('input', (e) => {
    state.minRecurrence = parseInt(e.target.value);
    recVal.textContent = state.minRecurrence === 1 ? '1x (All)' : `${state.minRecurrence}x+`;
  });
  recSlider.addEventListener('change', () => loadThermalAnomalies());

  // Limit Selector
  const limitSelect = document.getElementById('select-limit');
  limitSelect.addEventListener('change', (e) => {
    state.limit = parseInt(e.target.value);
    loadThermalAnomalies();
  });

  // Refresh Radar Button
  document.getElementById('btn-apply-filters').addEventListener('click', () => {
    loadThermalAnomalies();
  });

  // Quick Hotspots Chips
  document.querySelectorAll('.hotspot-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const lat = parseFloat(chip.getAttribute('data-lat'));
      const lon = parseFloat(chip.getAttribute('data-lon'));
      const zoom = parseInt(chip.getAttribute('data-zoom'));
      map.flyTo([lat, lon], zoom, { duration: 1.5 });
    });
  });

  // Quick Simulator Header Button & Tab Button
  document.getElementById('btn-quick-simulate').addEventListener('click', () => {
    switchTab('tab-stream');
    triggerSimulation();
  });
  document.getElementById('btn-trigger-simulation').addEventListener('click', () => {
    triggerSimulation();
  });

  // Inspector Form Presets
  document.getElementById('btn-fill-preset-ind').addEventListener('click', () => {
    document.getElementById('inp-lat').value = '20.952';
    document.getElementById('inp-lon').value = '85.215';
    document.getElementById('inp-b21').value = '388.5';
    document.getElementById('inp-b31').value = '302.1';
    document.getElementById('inp-frp').value = '145.0';
    document.getElementById('inp-confidence').value = '98';
    document.getElementById('inp-daynight').value = 'N';
    document.getElementById('inp-recurrence').value = '12';
  });

  document.getElementById('btn-fill-preset-wild').addEventListener('click', () => {
    document.getElementById('inp-lat').value = '19.420';
    document.getElementById('inp-lon').value = '80.112';
    document.getElementById('inp-b21').value = '325.2';
    document.getElementById('inp-b31').value = '304.5';
    document.getElementById('inp-frp').value = '18.4';
    document.getElementById('inp-confidence').value = '65';
    document.getElementById('inp-daynight').value = 'D';
    document.getElementById('inp-recurrence').value = '1';
  });

  // Threshold Slider Display
  const threshSlider = document.getElementById('inp-threshold');
  const threshDisplay = document.getElementById('threshold-val-display');
  threshSlider.addEventListener('input', (e) => {
    threshDisplay.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Manual Inspector Form Submit
  document.getElementById('inspector-form').addEventListener('submit', handleManualPrediction);
}

/* ==========================================================================
   Init App
   ========================================================================== */
window.addEventListener('DOMContentLoaded', () => {
  initClock();
  initUIEvents();
  checkSystemHealth();
  loadAnalytics();
  loadThermalAnomalies();
});
