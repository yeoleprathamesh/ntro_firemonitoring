import './style.css';
import L from 'leaflet';
window.L = L;
import 'leaflet.markercluster';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import 'leaflet.heat';
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
  activeClassFilter: 'all',  // 'all' | 'industrial' | 'vegetation'
  displayMode: 'points',    // 'points' | 'heatmap' | 'clusters'
  timeDays: 0,              // 0 (all time) | 1 | 7 | 30
  minFRP: 0,
  minRecurrence: 1,
  limit: 5000,
  audioEnabled: true,
  currentSimMarker: null,
  recentIntercepts: [],
  featureChart: null,
  activeFeedFilter: 'all',  // 'all' | 'critical' | 'industrial' | 'vegetation'
  feedSort: 'newest',       // 'newest' | 'frp' | 'conf'
  catalogTotal: 78459,
  viewportBoundsActive: false,
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

// Dedicated pane for Critical Flares with elevated z-index so they never get buried
map.createPane('flaresPane');
map.getPane('flaresPane').style.zIndex = '650';

// Dedicated pane for Plumes
map.createPane('plumePane');
map.getPane('plumePane').style.zIndex = '620';

// Base Tile Layers (High Performance, Dark Gray Canvas & Crisp Place Labels)
const darkBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles &copy; Esri, DeLorme, NAVTEQ',
  maxZoom: 16,
});

// Crisp high-contrast reference labels
const darkLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16,
  opacity: 0.95,
});

const darkTileGroup = L.layerGroup([darkBase, darkLabels]);

const satTileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles &copy; Esri & Earthstar Geographics',
  maxZoom: 18,
});

const satLabelsLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 18,
  opacity: 0.95,
});

const satTileGroup = L.layerGroup([satTileLayer, satLabelsLayer]);

// Initial Layer: Dark Tactical Canvas
darkTileGroup.addTo(map);

// Layer groups for active data representation
const pointsLayerGroup = L.layerGroup().addTo(map);
const clusterGroup = L.markerClusterGroup({
  chunkedLoading: true,
  maxClusterRadius: 45,
  showCoverageOnHover: false,
});
let heatLayer = null;

const simLayerGroup = L.layerGroup().addTo(map);
const plumeLayerGroup = L.layerGroup({ pane: 'plumePane' }).addTo(map);

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

function getFormattedUTCTime() {
  const now = new Date();
  const h = String(now.getUTCHours()).padStart(2, '0');
  const m = String(now.getUTCMinutes()).padStart(2, '0');
  const s = String(now.getUTCSeconds()).padStart(2, '0');
  return `${h}:${m}:${s} UTC`;
}

/* ==========================================================================
   Telemetry & Backend Communication
   ========================================================================== */
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('API offline');
    const data = await res.json();
    
    document.getElementById('server-status-dot').style.background = 'var(--teal-vegetation)';
    document.getElementById('server-status-dot').classList.add('status-pulse');
    document.getElementById('server-status-text').textContent = 'ONLINE (STABLE)';
    state.catalogTotal = data.total_records;
    document.getElementById('header-records-count').textContent = data.total_records.toLocaleString();
  } catch (err) {
    document.getElementById('server-status-dot').style.background = '#ef4444';
    document.getElementById('server-status-dot').classList.remove('status-pulse');
    document.getElementById('server-status-text').textContent = 'OFFLINE';
  }
}

async function loadAnalytics() {
  try {
    const [statsRes, modelRes, hotspotsRes, defconRes, chipsRes, histRes] = await Promise.all([
      fetch('/api/stats/overview'),
      fetch('/api/model/info'),
      fetch('/api/fires/hotspots?top_n=15'),
      fetch('/api/defcon/explosions?limit=8'),
      fetch('/api/hotspots/chips'),
      fetch('/api/stats/frp-histogram')
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

    if (defconRes.ok) {
      const defconData = await defconRes.json();
      renderDefconList(defconData.explosions);
    }

    if (chipsRes.ok) {
      const chipsData = await chipsRes.json();
      renderHotspotChips(chipsData.chips);
    }

    if (histRes.ok) {
      const histData = await histRes.json();
      renderHistogramHUD(histData.bins);
    }
  } catch (err) {
    console.error('Error loading analytics:', err);
  }
}

function renderHotspotChips(chips) {
  const container = document.getElementById('hotspots-chips-container');
  if (!container || !chips) return;

  container.innerHTML = chips.map(c => `
    <button class="hotspot-chip" data-lat="${c.lat}" data-lon="${c.lon}" data-zoom="${c.zoom}">
      <span>${c.label}</span>
      <span class="chip-count">(${c.count >= 1000 ? (c.count / 1000).toFixed(1) + 'K' : c.count})</span>
    </button>
  `).join('');

  container.querySelectorAll('.hotspot-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const lat = parseFloat(chip.getAttribute('data-lat'));
      const lon = parseFloat(chip.getAttribute('data-lon'));
      const zoom = parseInt(chip.getAttribute('data-zoom'));
      map.flyTo([lat, lon], zoom, { duration: 1.5 });
    });
  });
}

function renderHistogramHUD(bins) {
  const barsContainer = document.getElementById('histogram-bars');
  if (!barsContainer || !bins) return;

  const maxPct = Math.max(...bins.map(b => b.percentage || 1));
  barsContainer.innerHTML = bins.map(b => {
    const heightPct = Math.max(12, (b.percentage / maxPct) * 100);
    const isHighlighted = state.minFRP <= 10;
    return `
      <div class="hud-bar ${isHighlighted ? 'highlighted' : ''}" 
           style="height: ${heightPct}%;" 
           title="${b.range} MW: ${b.count.toLocaleString()} detections (${b.percentage}%)">
      </div>
    `;
  }).join('');
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

function renderDefconList(explosions) {
  const container = document.getElementById('defcon-list');
  if (!container || !explosions) return;

  if (explosions.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding: 10px; color: var(--text-dim); font-size: 11px;">No critical 3-sigma anomalies currently detected.</div>';
    return;
  }

  container.innerHTML = explosions.map((exp) => `
    <div class="defcon-item" data-lat="${exp.latitude}" data-lon="${exp.longitude}" data-frp="${exp.frp_mw}">
      <div class="defcon-item-left">
        <div class="defcon-item-title">
          <span>⚠️ ${exp.facility_name}</span>
        </div>
        <div class="defcon-item-sub">
          ${exp.operator} &bull; ${exp.latitude.toFixed(2)}°N, ${exp.longitude.toFixed(2)}°E
        </div>
        <div class="defcon-item-sub" style="color: #fca5a5; font-size: 10px;">
          FRP: <strong>${exp.frp_mw} MW</strong> (Mean: ${exp.baseline_mean_frp} MW, +${exp.pct_above_baseline.toFixed(0)}%)
        </div>
      </div>
      <div class="defcon-item-right">
        <span class="zscore-tag">+${exp.z_score.toFixed(1)}σ</span>
        <button class="btn-micro" style="border-color: #ef4444; color: #fca5a5;">RECON</button>
      </div>
    </div>
  `).join('');

  container.querySelectorAll('.defcon-item').forEach(item => {
    item.addEventListener('click', async () => {
      const lat = parseFloat(item.getAttribute('data-lat'));
      const lon = parseFloat(item.getAttribute('data-lon'));
      const frp = parseFloat(item.getAttribute('data-frp'));
      map.flyTo([lat, lon], 12, { duration: 1.5 });
      highlightLocationOnMap(lat, lon, 'DEFCON-1 Anomaly Target');
      await drawPlumeCorridor(lat, lon, frp);
    });
  });
}

async function drawPlumeCorridor(lat, lon, frp) {
  try {
    plumeLayerGroup.clearLayers();
    const res = await fetch(`/api/plume?lat=${lat}&lon=${lon}&frp=${frp}`);
    if (!res.ok) throw new Error('Plume query failed');
    const data = await res.json();

    const latLngs = data.polygon_coordinates.map(pt => [pt[1], pt[0]]);
    const isCritical = frp >= 80 || data.wind_speed_kmh >= 25;

    const polygon = L.polygon(latLngs, {
      pane: 'plumePane',
      color: isCritical ? '#ef4444' : '#ff9e40',
      weight: 1.8,
      dashArray: '5, 5',
      fillColor: isCritical ? '#dc2626' : '#ff6d00',
      fillOpacity: 0.35,
    }).addTo(plumeLayerGroup);

    polygon.bindTooltip(`
      <div style="font-family: var(--font-sans); font-size: 11.5px; color: #fff; padding: 4px;">
        <strong style="color: ${isCritical ? '#f87171' : '#ffb74d'};">${data.risk_category}</strong><br>
        Downwind Bearing: <strong>${data.downwind_bearing_deg}°</strong> @ <strong>${data.wind_speed_kmh} km/h</strong><br>
        Dispersion Length: <strong>${data.corridor_length_km} km</strong><br>
        Evacuation Radius: <strong>${data.recommended_containment_radius_km} km</strong><br>
        <span style="font-size: 9.5px; color: #cbd5e1;">Meteo Source: ${data.meteorological_source}</span>
      </div>
    `, { sticky: true, opacity: 0.95 }).openTooltip();

    playTacticalAlertSound(isCritical);
  } catch (err) {
    console.warn('Atmospheric plume notice:', err);
  }
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
          ticks: { color: '#cbd5e1', font: { family: 'Inter', size: 11 } }
        }
      }
    }
  });
}

/* ==========================================================================
   Load Thermal Points GeoJSON (Supports Viewport, Modes, Colorblind-Safe)
   ========================================================================== */
let lastFetchedFeatures = [];

async function loadThermalAnomalies() {
  const badge = document.getElementById('points-loaded-badge');
  const dockBadge = document.getElementById('dock-points-badge');
  badge.textContent = 'Scanning Orbit...';

  try {
    let url = `/api/fires/geojson?classification=${state.activeClassFilter}&min_frp=${state.minFRP}&min_recurrence=${state.minRecurrence}&limit=${state.limit}`;
    
    if (state.timeDays > 0) {
      url += `&days=${state.timeDays}`;
    }

    // If zoomed into regional detail (zoom >= 7), send viewport bbox to get complete local anomalies
    if (map.getZoom() >= 7) {
      const b = map.getBounds();
      const bboxStr = `${b.getWest().toFixed(3)},${b.getSouth().toFixed(3)},${b.getEast().toFixed(3)},${b.getNorth().toFixed(3)}`;
      url += `&bbox=${bboxStr}`;
      state.viewportBoundsActive = true;
    } else {
      state.viewportBoundsActive = false;
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load anomaly GeoJSON');
    const geojson = await res.json();

    const features = geojson.features || [];
    lastFetchedFeatures = features;

    // Count in current viewport
    const curBounds = map.getBounds();
    const inViewCount = features.filter(f => {
      const [lon, lat] = f.geometry.coordinates;
      return curBounds.contains([lat, lon]);
    }).length;

    const totalMatching = geojson.total_matching || features.length;
    const catTotal = geojson.catalog_total || state.catalogTotal;
    badge.textContent = `Showing ${features.length.toLocaleString()} of ${catTotal.toLocaleString()} (${inViewCount.toLocaleString()} in view)`;
    dockBadge.textContent = `${features.length.toLocaleString()} pts`;

    renderDataLayers(features);

  } catch (err) {
    console.error('Failed to load points:', err);
    badge.textContent = 'Error Fetching Feed';
  }
}

function renderDataLayers(features) {
  // Clear all layer representations
  pointsLayerGroup.clearLayers();
  clusterGroup.clearLayers();
  if (heatLayer) {
    map.removeLayer(heatLayer);
    heatLayer = null;
  }
  map.removeLayer(clusterGroup);

  if (state.displayMode === 'heatmap') {
    // Heatmap Mode using Leaflet.heat
    const heatData = features.map(f => {
      const [lon, lat] = f.geometry.coordinates;
      const intensity = Math.min(1.0, Math.max(0.2, f.properties.frp / 150.0));
      return [lat, lon, intensity];
    });

    heatLayer = L.heatLayer(heatData, {
      radius: 18,
      blur: 14,
      maxZoom: 12,
      gradient: {
        0.2: '#00e5ff',
        0.5: '#ff9e40',
        0.8: '#ff6d00',
        1.0: '#ffd600'
      }
    }).addTo(map);

  } else if (state.displayMode === 'clusters') {
    // Marker Clusters Mode
    features.forEach(f => {
      const marker = createCircleMarker(f);
      clusterGroup.addLayer(marker);
    });
    map.addLayer(clusterGroup);

  } else {
    // Default Points Mode: Canvas Circle Markers with FRP-scaled sizing
    features.forEach(f => {
      const marker = createCircleMarker(f);
      pointsLayerGroup.addLayer(marker);
    });
  }
}

function createCircleMarker(f) {
  const [lon, lat] = f.geometry.coordinates;
  const p = f.properties;
  const isInd = p.class_id === 1;
  const isFlare = p.frp >= 60;

  // Scale marker radius proportionally by FRP (MW)
  const radius = Math.min(18, Math.max(4.0, Math.log2(p.frp + 2) * 2.3));

  // Colorblind-Safe Palette:
  // Industrial -> Flame Orange (#ff6d00)
  // Wildfire -> Ocean Cyan (#00e5ff)
  // Flare -> Solar Gold (#ffd600)
  const fillColor = isFlare ? '#ffd600' : (isInd ? '#ff6d00' : '#00e5ff');
  const strokeColor = isFlare ? '#ffffff' : (isInd ? '#ffa040' : '#80f0ff');

  const markerOptions = {
    radius: radius,
    fillColor: fillColor,
    color: strokeColor,
    weight: isFlare ? 2.2 : 1.2,
    opacity: 0.95,
    fillOpacity: isFlare ? 0.9 : 0.7,
  };

  // Assign critical flares to high z-index pane so they are never buried
  if (isFlare) {
    markerOptions.pane = 'flaresPane';
  }

  const marker = L.circleMarker([lat, lon], markerOptions);

  // Unique ID for facility lookup in popup
  const facBoxId = `fac-${lat.toFixed(4).replace('.', '_')}-${lon.toFixed(4).replace('.', '_')}`;

  // Bind rich tactical popup
  marker.bindPopup(`
    <div class="tactical-popup">
      <div class="popup-header">
        <span class="popup-tag" style="color: ${isInd ? 'var(--orange-bright)' : 'var(--teal-vegetation)'}">
          ${isInd ? '🏭 INDUSTRIAL ANOMALY' : '🌲 WILDFIRE / BIOMASS'}
        </span>
        <span class="risk-badge ${isFlare ? 'badge-critical' : (isInd ? 'badge-high' : 'badge-low')}">
          ${isFlare ? 'CRITICAL FLARE' : (isInd ? 'HIGH CONFIDENCE' : 'ROUTINE')}
        </span>
      </div>
      ${isInd ? `
        <div class="popup-facility-box" id="${facBoxId}">
          <span class="facility-loading" style="color: #cbd5e1; font-size: 10px;">📍 Resolving Strategic Defense Facility...</span>
        </div>
      ` : ''}
      <table class="popup-table">
        <tr><td>COORDINATES</td><td>${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</td></tr>
        <tr><td>DATE RECORDED</td><td>${p.date || 'Historical Pass'}</td></tr>
        <tr><td>RADIATIVE POWER (FRP)</td><td><strong style="color: var(--amber-flare)">${p.frp} MW</strong></td></tr>
        <tr><td>BRIGHTNESS TEMP</td><td>${p.temp} K</td></tr>
        <tr><td>SPATIAL RECURRENCE</td><td>${p.recurrence}x persistent</td></tr>
        <tr><td>INDUSTRIAL PROB</td><td>${(p.prob_industrial * 100).toFixed(1)}%</td></tr>
      </table>
      <div class="popup-footer-expl">
        ${isInd ? 'Continuous industrial high-temperature emitter (smelter, power plant, refinery flaring).' : 'Transient biomass / agricultural burning or wildfire signature.'}
      </div>
      <div class="popup-actions-grid">
        <button class="tactical-btn secondary btn-inspect-here" data-lat="${lat}" data-lon="${lon}" data-temp="${p.temp}" data-b31="${p.bright_t31 || (p.temp - 40)}" data-frp="${p.frp}" data-rec="${p.recurrence || 1}">
          🎯 INSPECT
        </button>
        <button class="tactical-btn btn-view-plume" data-lat="${lat}" data-lon="${lon}" data-frp="${p.frp}">
          💨 PLUME CONE
        </button>
      </div>
    </div>
  `, { maxWidth: 320 });

  // Auto-resolve facility when clicked
  marker.on('click', async () => {
    if (isInd) {
      try {
        const facRes = await fetch(`/api/facility?lat=${lat}&lon=${lon}`);
        if (facRes.ok) {
          const fac = await facRes.json();
          const facEl = document.getElementById(facBoxId);
          if (facEl) {
            facEl.innerHTML = `
              <div class="facility-name">🏢 ${fac.facility_name}</div>
              <div class="facility-operator">${fac.operator} &bull; ${fac.facility_type}</div>
              <div style="color: ${fac.hazard_rating.includes('CRITICAL') ? '#ef4444' : '#fbbf24'}; font-size: 10px; font-weight: 600;">
                ${fac.hazard_rating} (${fac.verified_source})
              </div>
            `;
          }
        }
      } catch (e) {
        console.debug('Facility lookup notice:', e);
      }
    }
  });

  return marker;
}

// Delegated listener for popup "LOAD INTO INSPECTOR" & "PLUME CONE"
document.addEventListener('click', async (e) => {
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

    // Reveal form and hide empty state
    document.getElementById('inspector-empty-state').style.display = 'none';
    document.getElementById('inspector-form').style.display = 'flex';

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

  const plumeBtn = e.target.closest('.btn-view-plume');
  if (plumeBtn) {
    const lat = parseFloat(plumeBtn.getAttribute('data-lat'));
    const lon = parseFloat(plumeBtn.getAttribute('data-lon'));
    const frp = parseFloat(plumeBtn.getAttribute('data-frp'));
    await drawPlumeCorridor(lat, lon, frp);
  }
});

/* ==========================================================================
   Highlight Pin on Map
   ========================================================================== */
function highlightLocationOnMap(lat, lon, label = 'Target Location') {
  simLayerGroup.clearLayers();

  const ping = L.circleMarker([lat, lon], {
    radius: 20,
    fillColor: '#00f2fe',
    color: '#ffffff',
    weight: 2,
    opacity: 1,
    fillOpacity: 0.35,
  }).addTo(simLayerGroup);

  const centerDot = L.circleMarker([lat, lon], {
    radius: 6,
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
  const isInd = d.class_id === 1;
  const color = isInd ? 'var(--orange-bright)' : 'var(--teal-vegetation)';

  document.getElementById('alert-class-name').textContent = d.class_name;
  document.getElementById('alert-class-name').style.color = color;
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

  // Neutral multi-gradient progress bar
  const fillEl = document.getElementById('alert-prob-fill');
  fillEl.style.width = `${pct}%`;
  if (pct < 35) {
    fillEl.style.background = 'var(--teal-vegetation)';
  } else if (pct < 70) {
    fillEl.style.background = 'var(--amber-flare)';
  } else {
    fillEl.style.background = 'var(--orange-industrial)';
  }

  document.getElementById('alert-explanation-text').textContent = d.explanation;

  const flyBtn = document.getElementById('btn-fly-to-latest');
  flyBtn.style.display = 'block';
  flyBtn.onclick = () => {
    map.flyTo([d.latitude, d.longitude], 12, { duration: 1.5 });
  };

  const plumeBtn = document.getElementById('btn-plume-latest');
  plumeBtn.style.display = 'block';
  plumeBtn.onclick = async () => {
    await drawPlumeCorridor(d.latitude, d.longitude, d.features_used.frp || 50);
  };

  // Focus map on the target
  map.flyTo([d.latitude, d.longitude], 10, { duration: 1.2 });
  highlightLocationOnMap(d.latitude, d.longitude, `${d.class_name} (${d.risk_level})`);

  // Append to recent feed with UTC timestamp
  state.recentIntercepts.unshift({
    time: getFormattedUTCTime(),
    ...d
  });

  if (state.recentIntercepts.length > 25) state.recentIntercepts.pop();
  renderRecentFeed();
}

function renderRecentFeed() {
  const container = document.getElementById('stream-feed-list');
  const countTag = document.getElementById('feed-count-tag');
  if (!container) return;

  // Filter items
  let items = [...state.recentIntercepts];
  if (state.activeFeedFilter === 'critical') {
    items = items.filter(it => it.risk_level === 'CRITICAL' || it.features_used.frp >= 60);
  } else if (state.activeFeedFilter === 'industrial') {
    items = items.filter(it => it.class_id === 1);
  } else if (state.activeFeedFilter === 'vegetation') {
    items = items.filter(it => it.class_id === 0);
  }

  // Sort items
  if (state.feedSort === 'frp') {
    items.sort((a, b) => (b.features_used.frp || 0) - (a.features_used.frp || 0));
  } else if (state.feedSort === 'conf') {
    items.sort((a, b) => (b.confidence_score || 0) - (a.confidence_score || 0));
  }

  countTag.textContent = `${items.length} of ${state.recentIntercepts.length} logged`;

  if (items.length === 0) {
    container.innerHTML = `
      <div class="empty-feed-placeholder">
        <div class="radar-pulse-ring"></div>
        <span>${state.recentIntercepts.length === 0 ? 'No orbital detections ingested yet. Click "Simulate Pass" above.' : 'No detections match the selected filter.'}</span>
      </div>`;
    return;
  }

  container.innerHTML = items.map((item, idx) => {
    const isInd = item.class_id === 1;
    const color = isInd ? 'var(--orange-bright)' : 'var(--teal-vegetation)';
    return `
      <div class="feed-item" data-idx="${idx}">
        <div class="feed-item-left">
          <span class="feed-item-title" style="color: ${color}">
            ${isInd ? '🏭' : '🌲'} ${item.class_name}
          </span>
          <span class="feed-item-coords">${item.latitude.toFixed(3)}°N, ${item.longitude.toFixed(3)}°E &bull; ${item.time}</span>
        </div>
        <div class="feed-item-right">
          <span class="risk-badge badge-${item.risk_level.toLowerCase()}">${item.risk_level}</span>
          <span class="feed-item-coords monospace">${item.features_used.frp?.toFixed(1) || '--'} MW</span>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('.feed-item').forEach(el => {
    el.addEventListener('click', () => {
      const idx = parseInt(el.getAttribute('data-idx'));
      const item = items[idx];
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

    const isInd = result.class_id === 1;
    const color = isInd ? 'var(--orange-bright)' : 'var(--teal-vegetation)';

    document.getElementById('pred-class-name').textContent = result.class_name;
    document.getElementById('pred-class-name').style.color = color;

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
  const emptyState = document.getElementById('inspector-empty-state');
  const form = document.getElementById('inspector-form');
  if (emptyState && form) {
    emptyState.style.display = 'none';
    form.style.display = 'flex';
  }
  const latInp = document.getElementById('inp-lat');
  const lonInp = document.getElementById('inp-lon');
  if (latInp && lonInp) {
    latInp.value = lat.toFixed(4);
    lonInp.value = lng.toFixed(4);
  }
});

// Map MoveEnd -> Debounced update if bounds filtering is active
let moveEndTimer = null;
map.on('moveend', () => {
  if (moveEndTimer) clearTimeout(moveEndTimer);
  moveEndTimer = setTimeout(() => {
    // Only refresh if zoomed into regional level (zoom >= 7)
    if (map.getZoom() >= 7) {
      loadThermalAnomalies();
    }
  }, 350);
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

  // Filter Panel Collapse / Expand
  const panel = document.getElementById('map-controls-panel');
  const dock = document.getElementById('collapsed-filter-dock');
  const btnCollapse = document.getElementById('btn-collapse-filters');
  const btnExpand = document.getElementById('btn-expand-filters');

  function toggleFilterPanel(collapsed) {
    if (collapsed) {
      panel.style.display = 'none';
      dock.style.display = 'block';
    } else {
      dock.style.display = 'none';
      panel.style.display = 'flex';
    }
  }

  btnCollapse.addEventListener('click', () => toggleFilterPanel(true));
  btnExpand.addEventListener('click', () => toggleFilterPanel(false));

  // Legend Collapse Toggle
  const legend = document.getElementById('map-legend');
  const legendContent = document.getElementById('legend-content');
  const btnToggleLegend = document.getElementById('btn-toggle-legend');
  let legendCollapsed = false;

  btnToggleLegend.addEventListener('click', () => {
    legendCollapsed = !legendCollapsed;
    legendContent.style.display = legendCollapsed ? 'none' : 'flex';
    btnToggleLegend.textContent = legendCollapsed ? '+' : '−';
  });

  // Clickable Legend Items for Class Filtering
  document.querySelectorAll('.legend-item[data-legend-filter]').forEach(item => {
    item.addEventListener('click', () => {
      const filter = item.getAttribute('data-legend-filter');
      if (filter === 'flare') {
        state.minFRP = 60;
        document.getElementById('slider-frp').value = 60;
        document.getElementById('slider-frp-val').textContent = '60 MW';
      } else {
        state.activeClassFilter = filter;
        document.querySelectorAll('#filter-class-segment .segment-btn').forEach(btn => {
          btn.classList.toggle('active', btn.getAttribute('data-class') === filter);
        });
      }
      loadThermalAnomalies();
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

  // Map Display Mode Buttons (Points | Heatmap | Clusters)
  document.querySelectorAll('#filter-viewmode-segment .segment-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#filter-viewmode-segment .segment-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.displayMode = btn.getAttribute('data-mode');
      if (lastFetchedFeatures.length > 0) {
        renderDataLayers(lastFetchedFeatures);
      } else {
        loadThermalAnomalies();
      }
    });
  });

  // Time Filter Segment Buttons
  document.querySelectorAll('#filter-time-segment .segment-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#filter-time-segment .segment-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.timeDays = parseInt(btn.getAttribute('data-days'));
      loadThermalAnomalies();
    });
  });

  // FRP Slider & Histogram Update
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

  // Refresh Radar & Live FIRMS Sync Button
  const refreshRadarBtn = document.getElementById('btn-apply-filters');
  refreshRadarBtn.addEventListener('click', async () => {
    const originalText = refreshRadarBtn.innerHTML;
    refreshRadarBtn.disabled = true;
    refreshRadarBtn.innerHTML = '<span class="btn-icon">🛰️</span><span>SYNCING FIRMS...</span>';

    try {
      const syncRes = await fetch('/api/sync-daily', { method: 'POST' });
      if (syncRes.ok) {
        const syncData = await syncRes.json();
        if (syncData.status === 'success') {
          const added = syncData.new_detections || 0;
          refreshRadarBtn.innerHTML = `<span class="btn-icon">✅</span><span>SYNCED (+${added} NEW)</span>`;
          await checkSystemHealth();
          await loadAnalytics();
        } else {
          console.warn('Sync notice:', syncData.message);
          refreshRadarBtn.innerHTML = '<span class="btn-icon">🔄</span><span>REFRESHED</span>';
        }
      }
    } catch (err) {
      console.warn('Could not complete live sync, reloading cached anomalies:', err);
      refreshRadarBtn.innerHTML = '<span class="btn-icon">🔄</span><span>REFRESHED</span>';
    } finally {
      await loadThermalAnomalies();
      setTimeout(() => {
        refreshRadarBtn.disabled = false;
        refreshRadarBtn.innerHTML = originalText;
      }, 2500);
    }
  });

  // Feed Filter Chips
  document.querySelectorAll('.feed-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.feed-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.activeFeedFilter = chip.getAttribute('data-feed-filter');
      renderRecentFeed();
    });
  });

  // Feed Sort Selector
  document.getElementById('feed-sort-select').addEventListener('change', (e) => {
    state.feedSort = e.target.value;
    renderRecentFeed();
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
  function applyPreset(lat, lon, b21, b31, frp, conf, daynight, rec) {
    document.getElementById('inspector-empty-state').style.display = 'none';
    document.getElementById('inspector-form').style.display = 'flex';
    document.getElementById('inp-lat').value = lat;
    document.getElementById('inp-lon').value = lon;
    document.getElementById('inp-b21').value = b21;
    document.getElementById('inp-b31').value = b31;
    document.getElementById('inp-frp').value = frp;
    document.getElementById('inp-confidence').value = conf;
    document.getElementById('inp-daynight').value = daynight;
    document.getElementById('inp-recurrence').value = rec;
  }

  document.getElementById('btn-fill-preset-ind').addEventListener('click', () => {
    applyPreset('20.952', '85.215', '388.5', '302.1', '145.0', '98', 'N', '12');
  });

  document.getElementById('btn-fill-preset-wild').addEventListener('click', () => {
    applyPreset('19.420', '80.112', '325.2', '304.5', '18.4', '65', 'D', '1');
  });

  document.getElementById('btn-fill-preset-flare').addEventListener('click', () => {
    applyPreset('22.470', '69.855', '420.0', '310.0', '210.0', '99', 'N', '24');
  });

  // Threshold Slider Display
  const threshSlider = document.getElementById('inp-threshold');
  const threshDisplay = document.getElementById('threshold-val-display');
  threshSlider.addEventListener('input', (e) => {
    threshDisplay.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Manual Inspector Form Submit
  document.getElementById('inspector-form').addEventListener('submit', handleManualPrediction);

  // Keyboard Shortcuts Modal Toggle
  const shortcutsModal = document.getElementById('shortcuts-modal');
  const btnShowShortcuts = document.getElementById('btn-show-shortcuts');
  const btnCloseShortcuts = document.getElementById('btn-close-shortcuts');
  const btnDismissShortcuts = document.getElementById('btn-dismiss-shortcuts');

  function openShortcuts() {
    shortcutsModal.style.display = 'flex';
  }
  function closeShortcuts() {
    shortcutsModal.style.display = 'none';
  }

  btnShowShortcuts.addEventListener('click', openShortcuts);
  btnCloseShortcuts.addEventListener('click', closeShortcuts);
  btnDismissShortcuts.addEventListener('click', closeShortcuts);
  shortcutsModal.addEventListener('click', (e) => {
    if (e.target === shortcutsModal) closeShortcuts();
  });

  // Global Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    // Avoid triggering when user is typing inside an input or select
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

    if (e.key === 'f' || e.key === 'F') {
      const isCollapsed = panel.style.display === 'none';
      toggleFilterPanel(!isCollapsed);
    } else if (e.key === 'l' || e.key === 'L') {
      btnToggleLegend.click();
    } else if (e.key === 's' || e.key === 'S') {
      switchTab('tab-stream');
      triggerSimulation();
    } else if (e.key === 'm' || e.key === 'M') {
      layerBtn.click();
    } else if (e.key === 'a' || e.key === 'A') {
      soundBtn.click();
    } else if (e.key === '1') {
      switchTab('tab-stream');
    } else if (e.key === '2') {
      switchTab('tab-inspector');
    } else if (e.key === '3') {
      switchTab('tab-analytics');
    } else if (e.key === '?' || e.key === '/') {
      openShortcuts();
    } else if (e.key === 'Escape') {
      closeShortcuts();
      simLayerGroup.clearLayers();
      plumeLayerGroup.clearLayers();
    }
  });
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
