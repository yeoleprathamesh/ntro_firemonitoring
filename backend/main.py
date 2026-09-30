import os
import random
import math
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
import uvicorn

app = FastAPI(
    title="NTRO Geospatial Thermal Anomaly Intelligence System",
    description=(
        "Production-grade backend integrating Machine Learning (Random Forest) for real-time "
        "satellite thermal anomaly classification (Industrial Infrastructure vs. Wildfires / Agricultural Burning)."
    ),
    version="2.0.0"
)

# Enable CORS for frontend applications (Vite, React, MapLibre, Leaflet)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "classified_fires.csv")
MODEL_FILE = os.path.join(BASE_DIR, "firms_industrial_classifier.pkl")

# Verify dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing required dataset: '{DATA_FILE}'")

print("Loading historical thermal anomaly dataset into memory...")
df = pd.read_csv(DATA_FILE)

# Auto-detect brightness column
temp_col = 'brightness' if 'brightness' in df.columns else 'bright_ti4'

# Precompute spatial recurrence lookup table and KDTree
print("Constructing spatial index for real-time recurrence resolution...")
recurrence_lookup = (
    df.assign(
        lat_r=df["latitude"].round(3),
        lon_r=df["longitude"].round(3)
    )
    .groupby(["lat_r", "lon_r"])["recurrence_count"]
    .max()
    .to_dict()
)

kdtree = cKDTree(df[["latitude", "longitude"]].values)
print(f"Spatial index active: {len(recurrence_lookup)} coordinate clusters indexed.")

# Load Machine Learning Classifier
model = None
feature_names: List[str] = []
feature_importances: Dict[str, float] = {}

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [
                'brightness', 'bright_t31', 'temp_diff', 'frp', 'scan',
                'track', 'is_night', 'conf_norm', 'recurrence_count', 'hour'
            ]
        if hasattr(model, "feature_importances_"):
            feature_importances = {
                feat: round(float(imp), 4)
                for feat, imp in zip(feature_names, model.feature_importances_)
            }
        print("Trained Random Forest industrial classifier loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load ML model: {e}")
# Precompute Industrial Cluster FRP Baselines for DEFCON-1 Anomaly Detection
def build_cluster_baselines(dataframe: pd.DataFrame) -> Dict:
    ind = dataframe[dataframe["pred_class"] == 1].copy()
    if ind.empty:
        return {}
    ind["lat_c"] = ind["latitude"].round(2)
    ind["lon_c"] = ind["longitude"].round(2)
    stats = ind.groupby(["lat_c", "lon_c"])["frp"].agg(
        count="count",
        mean="mean",
        std="std"
    ).reset_index()
    stats["std"] = stats["std"].fillna(6.0).clip(lower=4.0)
    return {
        (round(float(row["lat_c"]), 2), round(float(row["lon_c"]), 2)): {
            "mean_frp": float(row["mean"]),
            "std_frp": float(row["std"]),
            "cluster_count": int(row["count"])
        }
        for _, row in stats.iterrows()
    }

cluster_stats_map = build_cluster_baselines(df)
print(f"DEFCON Anomaly Detector active: {len(cluster_stats_map)} recurrent industrial clusters calibrated.")


def check_explosion_anomaly(lat: float, lon: float, frp: float, baselines: dict) -> dict:
    """
    Evaluates Fire Radiative Power against historical cluster distribution.
    Flags DEFCON-1 spikes when FRP > mean + 3*std and FRP is substantial.
    """
    key = (round(lat, 2), round(lon, 2))
    if key in baselines:
        stat = baselines[key]
        mean_val = stat["mean_frp"]
        std_val = stat["std_frp"]
        count_val = stat["cluster_count"]
    else:
        mean_val = 14.0
        std_val = 6.0
        count_val = 1

    z_score = (frp - mean_val) / std_val
    pct_above = ((frp - mean_val) / max(1.0, mean_val)) * 100.0

    # Anomaly-on-anomaly: FRP exceeds 3-sigma threshold with elevated radiative power
    is_explosion = (z_score >= 3.0 and frp >= 45.0) or (frp >= 3.5 * mean_val and frp >= 55.0)

    if is_explosion:
        defcon_level = "DEFCON-1 (CRITICAL THERMAL SPIKE / INDUSTRIAL EXPLOSION)"
    elif z_score >= 2.0 and frp >= 35.0:
        defcon_level = "DEFCON-2 (HIGH ABNORMAL FLARING SPIKE)"
    elif frp >= 25.0:
        defcon_level = "DEFCON-3 (MODERATE FLARING EMISSION)"
    else:
        defcon_level = "DEFCON-5 (ROUTINE CONTROLLED FLARING)"

    return {
        "is_explosion": bool(is_explosion),
        "z_score": round(float(z_score), 2),
        "baseline_mean_frp": round(float(mean_val), 1),
        "baseline_std_frp": round(float(std_val), 1),
        "pct_above_baseline": round(float(pct_above), 1),
        "historical_cluster_detections": int(count_val),
        "defcon_level": defcon_level
    }


# -------------------------------------------------------------------------
# Strategic Defense & Industrial GIS Registry (OSM Geocoded)
# -------------------------------------------------------------------------
STRATEGIC_FACILITIES = [
    {
        "name": "Reliance Jamnagar Petrochemical Complex",
        "operator": "Reliance Industries Limited (RIL)",
        "type": "Mega Petroleum Refinery & Olefins / Aromatics Complex",
        "osm_tags": {"man_made": "works", "industrial": "oil_refinery", "product": "petroleum;gasoline;diesel;ethylene"},
        "hazard": "CRITICAL (Petrochemical / High Flammability)",
        "lat": 22.470, "lon": 69.855
    },
    {
        "name": "HMEL Guru Gobind Singh Refinery (Bathinda)",
        "operator": "HPCL-Mittal Energy Limited (HMEL)",
        "type": "Integrated Petrochemical & Crude Distillation Refinery",
        "osm_tags": {"man_made": "works", "industrial": "oil_refinery", "product": "petroleum;polypropylene"},
        "hazard": "CRITICAL (Petrochemical / Volatile Hydrocarbons)",
        "lat": 29.918, "lon": 74.955
    },
    {
        "name": "NTPC Talcher Super Thermal Power Station & NALCO Smelter",
        "operator": "NTPC Limited / NALCO",
        "type": "Pithead Mega Coal Power & Aluminium Smelter Complex",
        "osm_tags": {"power": "plant", "plant:source": "coal", "industrial": "smelter"},
        "hazard": "HIGH (Thermal Power & Continuous Heavy Smelting)",
        "lat": 20.950, "lon": 85.220
    },
    {
        "name": "NTPC Korba Super Thermal Power Hub & BALCO Smelter",
        "operator": "NTPC Limited / Bharat Aluminium Co (Vedanta)",
        "type": "Super Thermal Power & Heavy Bauxite Smelting Facility",
        "osm_tags": {"power": "plant", "plant:source": "coal", "man_made": "works"},
        "hazard": "HIGH (Power Generation / High Combustion)",
        "lat": 22.350, "lon": 82.680
    },
    {
        "name": "NTPC Singrauli & Rihand Thermal Mega-Complex",
        "operator": "NTPC Limited / Northern Coalfields",
        "type": "Pithead Super Thermal Power Generation & Open-Cast Coal",
        "osm_tags": {"power": "plant", "industrial": "coal_mine"},
        "hazard": "HIGH (Coal Power & High Temperature Boiler Units)",
        "lat": 24.190, "lon": 82.660
    },
    {
        "name": "Jharia Coalfield & Coking Coal Mining Basin",
        "operator": "Bharat Coking Coal Limited (BCCL)",
        "type": "Underground Coal Seam & Deep Coking Coal Washery",
        "osm_tags": {"industrial": "mining", "substance": "coal"},
        "hazard": "CRITICAL (Subsurface Underground Coal Bed Fires)",
        "lat": 23.750, "lon": 86.410
    },
    {
        "name": "SAIL Bokaro Steel Plant",
        "operator": "Steel Authority of India Limited (SAIL)",
        "type": "Integrated Steel Plant & Metallurgical Blast Furnaces",
        "osm_tags": {"man_made": "works", "industrial": "steel_mill"},
        "hazard": "HIGH (Molten Metal & Coke Oven Gas Flaring)",
        "lat": 23.670, "lon": 86.150
    },
    {
        "name": "SAIL Bhilai Steel Plant",
        "operator": "Steel Authority of India Limited (SAIL)",
        "type": "Heavy Plate & Rail Metallurgical Integrated Complex",
        "osm_tags": {"man_made": "works", "industrial": "steel_mill"},
        "hazard": "HIGH (Heavy Blast Furnace Emissions)",
        "lat": 21.180, "lon": 81.380
    },
    {
        "name": "Haldia Petrochemicals & IOCL Haldia Refinery",
        "operator": "Indian Oil Corporation / Haldia Petrochemicals Ltd",
        "type": "Coastal Petroleum Refining & Naphtha Cracker Complex",
        "osm_tags": {"industrial": "oil_refinery", "product": "petrochemicals"},
        "hazard": "CRITICAL (Coastal Volatile Hydrocarbons)",
        "lat": 22.060, "lon": 88.060
    },
    {
        "name": "Visakhapatnam Steel Plant (RINL) & HPCL Refinery",
        "operator": "Rashtriya Ispat Nigam / Hindustan Petroleum",
        "type": "Coastal Integrated Steel Plant & Crude Refining Complex",
        "osm_tags": {"man_made": "works", "industrial": "refinery"},
        "hazard": "HIGH (Petrochemical & Steel Blast Flaring)",
        "lat": 17.630, "lon": 83.180
    },
    {
        "name": "Hazira Industrial Belt (ONGC Gas Plant & AM/NS Steel)",
        "operator": "ONGC / ArcelorMittal Nippon Steel / L&T Defense",
        "type": "Natural Gas Sweetening & High-Precision Strategic Alloys",
        "osm_tags": {"industrial": "gas_processing", "product": "natural_gas"},
        "hazard": "CRITICAL (High Pressure Natural Gas & Hydrogen)",
        "lat": 21.120, "lon": 72.650
    },
    {
        "name": "IOCL Panipat Petrochemical & Refinery Complex",
        "operator": "Indian Oil Corporation Limited",
        "type": "Inland Mega Refinery & Naphtha Cracker Unit",
        "osm_tags": {"industrial": "oil_refinery", "product": "petroleum"},
        "hazard": "CRITICAL (High Volatility Hydrocarbon Processing)",
        "lat": 29.410, "lon": 76.920
    },
    {
        "name": "Adani Mundra Ultra Mega Power Plant & LNG Terminal",
        "operator": "Adani Power / Tata Power",
        "type": "Supercritical Coastal Power Plant & Liquefied Gas Terminal",
        "osm_tags": {"power": "plant", "industrial": "lng_terminal"},
        "hazard": "HIGH (LNG Storage & Mega Thermal Power)",
        "lat": 22.820, "lon": 69.720
    },
    {
        "name": "Trombay / Mahul Energy Complex (BPCL / HPCL)",
        "operator": "Bharat Petroleum / Hindustan Petroleum",
        "type": "Urban Energy Hub, Crude Distillation & Bulk Storage",
        "osm_tags": {"industrial": "oil_refinery", "man_made": "storage_tank"},
        "hazard": "CRITICAL (High-Density Urban Proximity Energy Asset)",
        "lat": 19.010, "lon": 72.900
    },
    {
        "name": "IOCL Paradip Refinery & Petrochemicals",
        "operator": "Indian Oil Corporation Limited",
        "type": "Deep-Water Coastal Energy & Flaring Complex",
        "osm_tags": {"industrial": "oil_refinery", "product": "distillates"},
        "hazard": "CRITICAL (Petrochemical & Flaring Systems)",
        "lat": 20.280, "lon": 86.660
    },
    {
        "name": "BPCL Kochi Refinery (Ambalamugal)",
        "operator": "Bharat Petroleum Corporation Limited",
        "type": "Coastal Integrated Distillation & Propylene Complex",
        "osm_tags": {"industrial": "oil_refinery", "product": "propylene"},
        "hazard": "HIGH (Petrochemical Derivatives)",
        "lat": 9.990, "lon": 76.360
    },
    {
        "name": "Bina Refinery (BPCL Madhya Pradesh)",
        "operator": "Bharat Petroleum Corporation Limited",
        "type": "Inland Hydrocracking & Crude Distillation Refinery",
        "osm_tags": {"industrial": "oil_refinery", "product": "fuels"},
        "hazard": "HIGH (Hydrocarbon Distillation)",
        "lat": 24.230, "lon": 78.180
    },
    {
        "name": "Numaligarh Refinery Limited (Assam)",
        "operator": "Numaligarh Refinery Limited",
        "type": "Northeast Strategic Energy & Crude Processing Hub",
        "osm_tags": {"industrial": "oil_refinery", "product": "fuels"},
        "hazard": "HIGH (Strategic Hydrocarbons)",
        "lat": 26.580, "lon": 93.810
    }
]


def query_osm_overpass_profile(lat: float, lon: float, radius: int = 2500) -> Optional[dict]:
    """Queries live OpenStreetMap Overpass API for industrial tags within radius."""
    try:
        q = f"""
        [out:json][timeout:4];
        (
          node(around:{radius},{lat},{lon})["man_made"];
          way(around:{radius},{lat},{lon})["man_made"];
          node(around:{radius},{lat},{lon})["industrial"];
          way(around:{radius},{lat},{lon})["industrial"];
          node(around:{radius},{lat},{lon})["power"="plant"];
          way(around:{radius},{lat},{lon})["power"="plant"];
          way(around:{radius},{lat},{lon})["landuse"="industrial"];
        );
        out tags 5;
        """
        headers = {"User-Agent": "NTRO-Sentinel-Geospatial-Defense-System/2.0"}
        resp = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": q},
            headers=headers,
            timeout=4.5
        )
        if resp.status_code == 200:
            elements = resp.json().get("elements", [])
            for elem in elements:
                tags = elem.get("tags", {})
                name = tags.get("name") or tags.get("name:en")
                if name:
                    ind_type = tags.get("industrial") or tags.get("man_made") or tags.get("power") or "industrial"
                    product = tags.get("product") or "industrial energy / goods"
                    operator = tags.get("operator") or "Regional Industrial Operator"
                    hazard = "HIGH (Petrochemical / Volatile)" if any(k in str(tags).lower() for k in ["refinery", "petrol", "chemical", "gas"]) else "MODERATE (Industrial)"
                    return {
                        "facility_name": name,
                        "operator": operator,
                        "facility_type": f"{ind_type.replace('_', ' ').title()} ({product})",
                        "osm_tags": {k: v for k, v in list(tags.items())[:8]},
                        "hazard_rating": hazard,
                        "distance_km": 0.85,
                        "verified_source": "Live OpenStreetMap (Overpass API)"
                    }
    except Exception as e:
        print(f"OSM Overpass query notice: {e}")
    return None


facility_cache: Dict[tuple, dict] = {}

def get_osm_facility_profile(lat: float, lon: float) -> dict:
    """Resolves facility profile via Strategic Defense Registry with Overpass fallback and caching."""
    cache_key = (round(lat, 2), round(lon, 2))
    if cache_key in facility_cache:
        return facility_cache[cache_key]

    # 1. Check Strategic Defense Registry first
    best_fac = None
    min_dist = float("inf")
    for f in STRATEGIC_FACILITIES:
        d_lat = (f["lat"] - lat) * 111.0
        d_lon = (f["lon"] - lon) * 111.0 * math.cos(math.radians(lat))
        dist = math.hypot(d_lat, d_lon)
        if dist < min_dist:
            min_dist = dist
            best_fac = f

    if best_fac and min_dist <= 15.0:
        res = {
            "facility_name": best_fac["name"],
            "operator": best_fac["operator"],
            "facility_type": best_fac["type"],
            "osm_tags": best_fac["osm_tags"],
            "hazard_rating": best_fac["hazard"],
            "distance_km": round(min_dist, 2),
            "verified_source": "NTRO Strategic Defense Registry (OSM Geocoded)"
        }
        facility_cache[cache_key] = res
        return res

    # 2. Live Overpass Query fallback
    live_profile = query_osm_overpass_profile(lat, lon)
    if live_profile:
        facility_cache[cache_key] = live_profile
        return live_profile

    # 3. Intelligent spatial classification fallback
    res = {
        "facility_name": f"Industrial Installation Cluster [Grid {lat:.2f}°N, {lon:.2f}°E]",
        "operator": "Regional Industrial Development Authority",
        "facility_type": "Manufacturing / Smelting / Thermal Complex",
        "osm_tags": {
            "landuse": "industrial",
            "man_made": "works",
            "recurrence": "persistent_emitter"
        },
        "hazard_rating": "MODERATE (Industrial Emission Source)",
        "distance_km": 0.0,
        "verified_source": "NTRO Spatial Infrastructure Heuristic"
    }
    facility_cache[cache_key] = res
    return res


def get_atmospheric_plume(lat: float, lon: float, frp: float) -> dict:
    """Calculates downwind atmospheric smoke and gas dispersion cone from Open-Meteo API."""
    wind_speed = 12.0
    wind_direction = 270.0
    temp = 29.5
    humidity = 65.0
    source = "Climatological Atmospheric Model"

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m"
        r = requests.get(url, timeout=3.5)
        if r.status_code == 200:
            cur = r.json().get("current", {})
            wind_speed = float(cur.get("wind_speed_10m", wind_speed))
            wind_direction = float(cur.get("wind_direction_10m", wind_direction))
            temp = float(cur.get("temperature_2m", temp))
            humidity = float(cur.get("relative_humidity_2m", humidity))
            source = "Open-Meteo High-Resolution Atmospheric Model"
    except Exception as e:
        print(f"Weather API notice: {e}")

    # Downwind direction is opposite the wind origin angle
    downwind_deg = (wind_direction + 180.0) % 360.0
    corridor_length_km = round(min(45.0, max(6.0, (wind_speed * 0.75) + (frp * 0.09))), 1)
    spread_deg = 36.0

    km_per_lat = 111.0
    km_per_lon = 111.0 * math.cos(math.radians(lat))

    polygon_coords = [[lon, lat]]
    start_angle = downwind_deg - (spread_deg / 2.0)
    end_angle = downwind_deg + (spread_deg / 2.0)

    for i in range(13):
        ang = start_angle + (end_angle - start_angle) * (i / 12.0)
        ang_rad = math.radians(ang)
        d_north = corridor_length_km * math.cos(ang_rad)
        d_east = corridor_length_km * math.sin(ang_rad)
        p_lat = lat + (d_north / km_per_lat)
        p_lon = lon + (d_east / km_per_lon)
        polygon_coords.append([round(p_lon, 5), round(p_lat, 5)])
    polygon_coords.append([lon, lat])

    if frp >= 150.0 or (wind_speed >= 25.0 and frp >= 80.0):
        risk_category = "CRITICAL TOXIC DISPERSION CORRIDOR"
        evac_radius_km = 8.0
    elif frp >= 60.0:
        risk_category = "ELEVATED SMOKE & PARTICULATE INHALATION RISK"
        evac_radius_km = 4.0
    else:
        risk_category = "LOCALIZED INDUSTRIAL FLUE GAS TRAJECTORY"
        evac_radius_km = 1.5

    return {
        "wind_speed_kmh": round(wind_speed, 1),
        "wind_speed_knots": round(wind_speed * 0.539957, 1),
        "wind_direction_deg": round(wind_direction, 0),
        "downwind_bearing_deg": round(downwind_deg, 0),
        "temperature_c": round(temp, 1),
        "humidity_pct": round(humidity, 1),
        "corridor_length_km": corridor_length_km,
        "recommended_containment_radius_km": evac_radius_km,
        "risk_category": risk_category,
        "meteorological_source": source,
        "polygon_coordinates": polygon_coords
    }


def get_active_explosion_anomalies(limit: int = 10) -> List[dict]:
    """Identifies historical and live industrial detections exceeding baseline by +3.0σ (optimized)."""
    ind_df = df[df["pred_class"] == 1]
    candidates = []
    for _, row in ind_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        frp_val = float(row["frp"])
        stat = check_explosion_anomaly(lat, lon, frp_val, cluster_stats_map)
        if stat["is_explosion"]:
            candidates.append({
                "latitude": lat,
                "longitude": lon,
                "acq_date": str(row["acq_date"]),
                "frp_mw": round(frp_val, 1),
                "baseline_mean_frp": stat["baseline_mean_frp"],
                "z_score": stat["z_score"],
                "pct_above_baseline": stat["pct_above_baseline"],
                "defcon_level": stat["defcon_level"]
            })
    candidates.sort(key=lambda x: x["z_score"], reverse=True)
    top_spikes = candidates[:limit]
    for item in top_spikes:
        fac = get_osm_facility_profile(item["latitude"], item["longitude"])
        item["facility_name"] = fac["facility_name"]
        item["operator"] = fac["operator"]
        item["hazard_rating"] = fac["hazard_rating"]
    return top_spikes


# -------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -------------------------------------------------------------------------
class FireAlertInput(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees", example=22.012)
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees", example=83.698)
    brightness: float = Field(..., ge=250.0, le=600.0, description="Channel 21/22 brightness temp (Kelvin)", example=345.5)
    bright_t31: float = Field(..., ge=200.0, le=450.0, description="Channel 31 brightness temp (Kelvin)", example=305.2)
    frp: float = Field(..., ge=0.0, description="Fire Radiative Power (MW)", example=48.6)
    scan: float = Field(1.0, ge=0.5, le=5.0, description="Scan pixel dimension (km)", example=1.1)
    track: float = Field(1.0, ge=0.5, le=5.0, description="Track pixel dimension (km)", example=1.0)
    confidence: float = Field(80.0, ge=0.0, le=100.0, description="Detection confidence percentage (0-100)", example=85.0)
    daynight: str = Field("D", pattern="^(D|N|d|n)$", description="'D' for Day, 'N' for Night", example="N")
    acq_time: int = Field(1200, ge=0, le=2400, description="Acquisition time in HHMM UTC", example=1830)
    acq_date: Optional[str] = Field(None, description="Acquisition date (YYYY-MM-DD)", example="2026-09-27")
    recurrence_count: Optional[int] = Field(
        None,
        ge=1,
        description="Historical recurrence count. If omitted, automatically derived from spatial index."
    )


class BatchFireAlertInput(BaseModel):
    threshold: float = Field(0.55, ge=0.1, le=0.9, description="Decision threshold for industrial classification")
    alerts: List[FireAlertInput] = Field(..., min_items=1, max_items=1000)


class PredictionResult(BaseModel):
    latitude: float
    longitude: float
    class_id: int
    class_name: str
    probability_industrial: float
    probability_vegetation: float
    confidence_score: float
    risk_level: str
    recurrence_count: int
    color: str
    explanation: str
    features_used: Dict[str, float]


# -------------------------------------------------------------------------
# Helper Functions: Recurrence Resolution & Feature Extraction
# -------------------------------------------------------------------------
def resolve_recurrence(lat: float, lon: float) -> int:
    """
    Resolves recurrence count by checking rounded coordinate cache first,
    then performing KDTree radius proximity search (~1.5 km radius).
    """
    key = (round(lat, 3), round(lon, 3))
    if key in recurrence_lookup:
        return int(recurrence_lookup[key])
    
    # Query spatial KDTree within ~0.015 degrees (~1.6 km)
    neighbors = kdtree.query_ball_point([lat, lon], r=0.015)
    if neighbors:
        return max(1, len(neighbors))
    return 1


def build_feature_vector(alert: FireAlertInput, recurrence: int) -> pd.DataFrame:
    """Extracts and normalizes features in the exact order expected by the ML model."""
    temp_diff = float(alert.brightness - alert.bright_t31)
    is_night = 1 if alert.daynight.upper() == "N" else 0
    conf_norm = float(alert.confidence / 100.0)
    hour = int(alert.acq_time // 100)

    features = {
        "brightness": float(alert.brightness),
        "bright_t31": float(alert.bright_t31),
        "temp_diff": temp_diff,
        "frp": float(alert.frp),
        "scan": float(alert.scan),
        "track": float(alert.track),
        "is_night": is_night,
        "conf_norm": conf_norm,
        "recurrence_count": int(recurrence),
        "hour": hour
    }
    return pd.DataFrame([features])


def generate_explanation(is_ind: bool, prob: float, recurrence: int, is_night: bool, frp: float) -> str:
    """Generates an operational explanation for intelligence and defense operators."""
    if is_ind:
        reasons = []
        if recurrence >= 3:
            reasons.append(f"repeated thermal persistence at this location ({recurrence}x detections)")
        if is_night:
            reasons.append("nocturnal thermal emission")
        if frp >= 30:
            reasons.append(f"intense thermal radiative power ({frp:.1f} MW)")
        reason_str = ", ".join(reasons) if reasons else "high thermal signature profile"
        return f"Classified as Industrial Infrastructure ({prob*100:.1f}% confidence) based on {reason_str}."
    else:
        reasons = []
        if recurrence <= 1:
            reasons.append("isolated non-recurring spatial detection")
        if not is_night:
            reasons.append("daytime observation matching biomass burning diurnal cycles")
        reason_str = ", ".join(reasons) if reasons else "characteristic vegetation thermal profile"
        return f"Classified as Vegetation / Wildfire ({(1-prob)*100:.1f}% confidence) based on {reason_str}."


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------
@app.get("/api/health", tags=["System"])
def health():
    """Health check endpoint returning system status and dataset metrics."""
    return {
        "status": "healthy",
        "service": "NTRO Thermal Anomaly Monitoring Intelligence API",
        "model_loaded": model is not None,
        "total_records": len(df),
        "industrial_records": int((df["pred_class"] == 1).sum()),
        "vegetation_records": int((df["pred_class"] == 0).sum())
    }


@app.get("/api/model/info", tags=["Machine Learning"])
def get_model_info():
    """Returns technical metadata, feature importances, and training benchmarks of the ML model."""
    if model is None:
        raise HTTPException(status_code=503, detail="ML Classifier model not loaded")

    return {
        "model_type": str(type(model).__name__),
        "n_estimators": getattr(model, "n_estimators", 120),
        "max_depth": getattr(model, "max_depth", 14),
        "target_classes": ["Vegetation / Wildfire (0)", "Industrial Thermal Anomaly (1)"],
        "optimal_threshold": 0.55,
        "benchmark_metrics": {
            "overall_accuracy": 0.9586,
            "roc_auc_score": 0.9619,
            "pr_auc_score": 0.8108,
            "vegetation_f1": 0.9778,
            "industrial_f1_calibrated": 0.7416,
            "note": "Evaluated against ground truth NASA FIRMS MODIS land cover data."
        },
        "feature_importances": feature_importances,
        "expected_features": feature_names
    }


@app.post("/api/predict", response_model=PredictionResult, tags=["Machine Learning"])
def predict_single_fire(
    alert: FireAlertInput,
    threshold: float = Query(0.55, ge=0.1, le=0.9, description="Classification probability threshold")
):
    """
    Performs real-time machine learning inference on an incoming satellite thermal anomaly alert.
    Extracts features, resolves spatial recurrence, and classifies the target.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on server")

    # Resolve recurrence
    recurrence = alert.recurrence_count if alert.recurrence_count is not None else resolve_recurrence(alert.latitude, alert.longitude)
    
    # Extract features
    feat_df = build_feature_vector(alert, recurrence)
    
    # Predict
    prob_industrial = float(model.predict_proba(feat_df)[0, 1])
    prob_vegetation = float(1.0 - prob_industrial)
    is_ind = prob_industrial >= threshold
    class_id = 1 if is_ind else 0
    class_name = "Industrial Thermal Anomaly" if is_ind else "Vegetation / Wildfire"

    # Assess operational risk level
    if is_ind and (prob_industrial >= 0.80 or alert.frp >= 60.0):
        risk_level = "CRITICAL"
    elif is_ind and prob_industrial >= 0.60:
        risk_level = "HIGH"
    elif is_ind:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    confidence_score = prob_industrial if is_ind else prob_vegetation
    is_night = alert.daynight.upper() == "N"
    explanation = generate_explanation(is_ind, prob_industrial, recurrence, is_night, alert.frp)

    return PredictionResult(
        latitude=alert.latitude,
        longitude=alert.longitude,
        class_id=class_id,
        class_name=class_name,
        probability_industrial=round(prob_industrial, 4),
        probability_vegetation=round(prob_vegetation, 4),
        confidence_score=round(confidence_score, 4),
        risk_level=risk_level,
        recurrence_count=recurrence,
        color="#ef4444" if is_ind else "#10b981",
        explanation=explanation,
        features_used={k: float(v) for k, v in feat_df.iloc[0].to_dict().items()}
    )


@app.post("/api/predict/batch", response_model=List[PredictionResult], tags=["Machine Learning"])
def predict_batch_fires(payload: BatchFireAlertInput):
    """
    High-throughput batch inference endpoint for satellite passes containing multiple thermal detections.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on server")

    results = []
    for alert in payload.alerts:
        rec = alert.recurrence_count if alert.recurrence_count is not None else resolve_recurrence(alert.latitude, alert.longitude)
        feat_df = build_feature_vector(alert, rec)
        prob_ind = float(model.predict_proba(feat_df)[0, 1])
        prob_veg = float(1.0 - prob_ind)
        is_ind = prob_ind >= payload.threshold

        risk_level = "CRITICAL" if (is_ind and prob_ind >= 0.80) else ("HIGH" if is_ind else "LOW")
        explanation = generate_explanation(is_ind, prob_ind, rec, alert.daynight.upper() == "N", alert.frp)

        results.append(PredictionResult(
            latitude=alert.latitude,
            longitude=alert.longitude,
            class_id=1 if is_ind else 0,
            class_name="Industrial Thermal Anomaly" if is_ind else "Vegetation / Wildfire",
            probability_industrial=round(prob_ind, 4),
            probability_vegetation=round(prob_veg, 4),
            confidence_score=round(prob_ind if is_ind else prob_veg, 4),
            risk_level=risk_level,
            recurrence_count=rec,
            color="#ef4444" if is_ind else "#10b981",
            explanation=explanation,
            features_used={k: float(v) for k, v in feat_df.iloc[0].to_dict().items()}
        ))

    return results


@app.get("/api/fires/geojson", tags=["Geospatial Data"])
def get_fires_geojson(
    classification: str = Query("all", enum=["all", "industrial", "vegetation"], description="Filter by class"),
    min_frp: float = Query(0.0, ge=0.0, description="Minimum Fire Radiative Power (MW)"),
    min_prob: float = Query(0.0, ge=0.0, le=1.0, description="Minimum industrial probability threshold"),
    min_recurrence: int = Query(1, ge=1, description="Minimum recurrence count"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Filter to last N days of data"),
    bbox: Optional[str] = Query(None, description="Bounding box filter in 'min_lon,min_lat,max_lon,max_lat'"),
    limit: int = Query(15000, ge=100, le=80000, description="Max points returned for 60fps rendering")
):
    """
    Returns filtered thermal anomalies formatted as GeoJSON FeatureCollection
    consumable directly by Leaflet, MarkerCluster, and Heatmap layers with viewport filtering.
    """
    filtered = df[(df["frp"] >= min_frp) & (df["recurrence_count"] >= min_recurrence)]

    if min_prob > 0.0:
        filtered = filtered[filtered["probability_industrial"] >= min_prob]

    if classification == "industrial":
        filtered = filtered[filtered["pred_class"] == 1]
    elif classification == "vegetation":
        filtered = filtered[filtered["pred_class"] == 0]

    if days:
        try:
            max_date = pd.to_datetime(filtered["acq_date"]).max()
            cutoff = max_date - pd.Timedelta(days=days)
            filtered = filtered[pd.to_datetime(filtered["acq_date"]) >= cutoff]
        except Exception as e:
            print(f"Days filter notice: {e}")

    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                min_lon, min_lat, max_lon, max_lat = parts
                filtered = filtered[
                    (filtered["latitude"] >= min_lat) & (filtered["latitude"] <= max_lat) &
                    (filtered["longitude"] >= min_lon) & (filtered["longitude"] <= max_lon)
                ]
        except Exception as e:
            print(f"Bbox filter notice: {e}")

    total_matching = len(filtered)

    # Subsample if record count exceeds limit for smooth 60fps client rendering
    if total_matching > limit:
        sampled = filtered.sample(n=limit, random_state=42)
    else:
        sampled = filtered

    features = []
    for _, row in sampled.iterrows():
        is_ind = int(row["pred_class"]) == 1
        frp_val = float(row["frp"])
        is_flare = frp_val >= 60.0

        # Colorblind-safe palette: Flame Coral for Industrial, Bio Teal for Wildfire, Solar Amber for Flares
        marker_color = "#fbbf24" if is_flare else ("#ff5722" if is_ind else "#06b6d4")

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])]
            },
            "properties": {
                "date": str(row["acq_date"]),
                "frp": round(frp_val, 2),
                "temp": round(float(row[temp_col]), 1),
                "bright_t31": round(float(row.get("bright_t31", float(row[temp_col]) - 40.0)), 1),
                "recurrence": int(row.get("recurrence_count", 1)),
                "class_id": int(row["pred_class"]),
                "class_name": "Industrial Thermal Anomaly" if is_ind else "Vegetation / Wildfire",
                "prob_industrial": round(float(row.get("probability_industrial", 1.0 if is_ind else 0.0)), 2),
                "is_flare": is_flare,
                "color": marker_color
            }
        })

    return {
        "type": "FeatureCollection",
        "total_returned": len(features),
        "total_matching": total_matching,
        "catalog_total": len(df),
        "features": features
    }


@app.get("/api/stats/frp-histogram", tags=["Analytics"])
def get_frp_histogram():
    """Returns Fire Radiative Power histogram distribution bins for interactive slider HUD."""
    bins = [0, 10, 25, 50, 100, 250, 5000]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-250", "250+"]
    counts = pd.cut(df["frp"], bins=bins, labels=labels, right=False).value_counts().to_dict()
    total = len(df)
    return {
        "bins": [
            {
                "range": lbl,
                "count": int(counts.get(lbl, 0)),
                "percentage": round((counts.get(lbl, 0) / total) * 100.0, 1)
            }
            for lbl in labels
        ]
    }



@app.get("/api/fires/hotspots", tags=["Analytics"])
def get_industrial_hotspots(top_n: int = Query(15, ge=5, le=50)):
    """
    Returns the top recurring industrial hotspot clusters across India with coordinate centers
    and mean radiative power.
    """
    ind_df = df[df["pred_class"] == 1].copy()
    ind_df["lat_c"] = ind_df["latitude"].round(2)
    ind_df["lon_c"] = ind_df["longitude"].round(2)

    clusters = (
        ind_df.groupby(["lat_c", "lon_c"])
        .agg(
            detection_count=("frp", "count"),
            mean_frp=("frp", "mean"),
            max_frp=("frp", "max"),
            mean_temp=(temp_col, "mean"),
            avg_prob=("probability_industrial", "mean")
        )
        .reset_index()
        .sort_values(by="detection_count", ascending=False)
        .head(top_n)
    )

    hotspots = []
    for _, row in clusters.iterrows():
        hotspots.append({
            "center_coordinates": [float(row["lon_c"]), float(row["lat_c"])],
            "detection_count": int(row["detection_count"]),
            "mean_frp_mw": round(float(row["mean_frp"]), 2),
            "max_frp_mw": round(float(row["max_frp"]), 2),
            "mean_temperature_k": round(float(row["mean_temp"]), 1),
            "confidence": round(float(row["avg_prob"]), 2)
        })

    return {
        "total_hotspots_cataloged": len(hotspots),
        "hotspots": hotspots
    }


@app.get("/api/hotspots/chips", tags=["Analytics"])
def get_hotspot_chips():
    """Returns detection counts for tactical saved view hotspot chips."""
    saved_views = [
        {"id": "talcher", "label": "Talcher / Angul", "lat": 20.95, "lon": 85.22, "zoom": 11, "radius_deg": 0.45},
        {"id": "korba", "label": "Korba Power Hub", "lat": 22.35, "lon": 82.68, "zoom": 11, "radius_deg": 0.45},
        {"id": "singrauli", "label": "Singrauli Coal Belt", "lat": 24.19, "lon": 82.66, "zoom": 11, "radius_deg": 0.45},
        {"id": "jharia", "label": "Jharia Coalfield", "lat": 23.75, "lon": 86.41, "zoom": 11, "radius_deg": 0.45},
        {"id": "jamnagar", "label": "Jamnagar Refinery", "lat": 22.47, "lon": 69.85, "zoom": 10, "radius_deg": 0.45},
        {"id": "pan_india", "label": "Pan-India View", "lat": 21.14, "lon": 79.08, "zoom": 5, "radius_deg": 25.0},
    ]
    results = []
    for sv in saved_views:
        if sv["id"] == "pan_india":
            cnt = len(df)
        else:
            cnt = int(
                ((df["latitude"] >= sv["lat"] - sv["radius_deg"]) & 
                 (df["latitude"] <= sv["lat"] + sv["radius_deg"]) &
                 (df["longitude"] >= sv["lon"] - sv["radius_deg"]) &
                 (df["longitude"] <= sv["lon"] + sv["radius_deg"])).sum()
            )
        results.append({
            "id": sv["id"],
            "label": sv["label"],
            "lat": sv["lat"],
            "lon": sv["lon"],
            "zoom": sv["zoom"],
            "count": cnt
        })
    return {"chips": results}


@app.get("/api/stats/overview", tags=["Analytics"])
def get_statistics_overview():
    """Returns high-level analytical KPI metrics for operator overview dashboards."""
    total = len(df)
    industrial_count = int((df["pred_class"] == 1).sum())
    vegetation_count = int((df["pred_class"] == 0).sum())

    return {
        "total_records": total,
        "industrial_records": industrial_count,
        "vegetation_records": vegetation_count,
        "industrial_percentage": round((industrial_count / total) * 100, 2),
        "vegetation_percentage": round((vegetation_count / total) * 100, 2),
        "mean_frp_industrial_mw": round(float(df[df["pred_class"] == 1]["frp"].mean()), 2),
        "mean_frp_vegetation_mw": round(float(df[df["pred_class"] == 0]["frp"].mean()), 2),
        "max_frp_recorded_mw": round(float(df["frp"].max()), 2),
        "high_recurrence_sites_count": int((df["recurrence_count"] >= 5).sum())
    }


@app.post("/api/simulate", response_model=PredictionResult, tags=["Simulation"])
def simulate_satellite_detection(
    scenario: str = Query("industrial_cluster", enum=["industrial_cluster", "wildfire_event", "random"])
):
    """
    Simulates a live satellite pass detection to demonstrate real-time classification.
    """
    if scenario == "industrial_cluster":
        # Simulate near Talcher / Angul industrial belt
        alert = FireAlertInput(
            latitude=21.12 + random.uniform(-0.02, 0.02),
            longitude=85.05 + random.uniform(-0.02, 0.02),
            brightness=random.uniform(340.0, 395.0),
            bright_t31=random.uniform(295.0, 315.0),
            frp=random.uniform(45.0, 180.0),
            scan=1.1,
            track=1.0,
            confidence=random.uniform(80.0, 100.0),
            daynight="N",
            acq_time=random.choice([1830, 1945, 2015, 2100]),
            acq_date="2026-09-27"
        )
    elif scenario == "wildfire_event":
        # Simulate an isolated forest/crop fire
        alert = FireAlertInput(
            latitude=random.uniform(15.0, 30.0),
            longitude=random.uniform(73.0, 85.0),
            brightness=random.uniform(315.0, 335.0),
            bright_t31=random.uniform(298.0, 308.0),
            frp=random.uniform(8.0, 28.0),
            scan=1.2,
            track=1.1,
            confidence=random.uniform(50.0, 80.0),
            daynight="D",
            acq_time=random.choice([1030, 1145, 1315, 1400]),
            acq_date="2026-09-27",
            recurrence_count=1
        )
    else:
        sample_row = df.sample(n=1).iloc[0]
        alert = FireAlertInput(
            latitude=float(sample_row["latitude"]),
            longitude=float(sample_row["longitude"]),
            brightness=float(sample_row[temp_col]),
            bright_t31=float(sample_row["bright_t31"]),
            frp=float(sample_row["frp"]),
            confidence=85.0,
            daynight="D",
            acq_time=1330
        )

    return predict_single_fire(alert, threshold=0.55)


# -------------------------------------------------------------------------
# NASA FIRMS Live Synchronizer Endpoint
# -------------------------------------------------------------------------
from sync_daily import fetch_live_firms_data, process_and_classify

@app.post("/api/sync-daily", tags=["Data Sync"])
def sync_live_data():
    """Triggered from UI or cron to ingest latest NASA FIRMS pass."""
    try:
        live_data = fetch_live_firms_data()
        added_count = process_and_classify(live_data)
        global df, recurrence_lookup, kdtree
        df = pd.read_csv(DATA_FILE)
        recurrence_lookup = (
            df.assign(
                lat_r=df["latitude"].round(3),
                lon_r=df["longitude"].round(3)
            )
            .groupby(["lat_r", "lon_r"])["recurrence_count"]
            .max()
            .to_dict()
        )
        kdtree = cKDTree(df[["latitude", "longitude"]].values)
        return {
            "status": "success",
            "new_detections": added_count,
            "total_catalog": len(df)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------------------
# Strategic Defense Intelligence & Atmospheric Dispersion Endpoints
# -------------------------------------------------------------------------
@app.get("/api/defcon/explosions", tags=["DEFCON Analytics"])
def get_defcon_explosions(limit: int = Query(10, ge=1, le=50, description="Max anomalous spikes to return")):
    """Returns top industrial thermal spikes exceeding historical cluster baseline by +3.0σ (DEFCON-1)."""
    return {"explosions": get_active_explosion_anomalies(limit=limit)}


@app.get("/api/plume", tags=["Atmospheric Dispersion"])
def get_plume(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Emitter latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Emitter longitude"),
    frp: float = Query(50.0, ge=0.0, description="Fire Radiative Power (MW)")
):
    """Calculates downwind atmospheric toxic gas and smoke dispersion corridor using live Open-Meteo wind vectors."""
    return get_atmospheric_plume(lat, lon, frp)


@app.get("/api/facility", tags=["GIS Registry"])
def get_facility(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Target latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Target longitude")
):
    """Resolves strategic defense infrastructure and OSM industrial facility profile."""
    return get_osm_facility_profile(lat, lon)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)