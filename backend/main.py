import os
import random
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
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
else:
    print(f"Warning: Model file '{MODEL_FILE}' not found.")


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
    limit: int = Query(15000, ge=100, le=80000, description="Max points returned for 60fps WebGL rendering")
):
    """
    Returns filtered thermal anomalies formatted as GeoJSON FeatureCollection
    consumable directly by MapLibre GL, Mapbox, or Leaflet.
    """
    filtered = df[(df["frp"] >= min_frp) & (df["recurrence_count"] >= min_recurrence)]

    if min_prob > 0.0:
        filtered = filtered[filtered["probability_industrial"] >= min_prob]

    if classification == "industrial":
        filtered = filtered[filtered["pred_class"] == 1]
    elif classification == "vegetation":
        filtered = filtered[filtered["pred_class"] == 0]

    # Subsample if record count exceeds limit for smooth rendering
    if len(filtered) > limit:
        filtered = filtered.sample(n=limit, random_state=42)

    features = []
    for _, row in filtered.iterrows():
        is_ind = int(row["pred_class"]) == 1
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])]
            },
            "properties": {
                "date": str(row["acq_date"]),
                "frp": round(float(row["frp"]), 2),
                "temp": round(float(row[temp_col]), 1),
                "bright_t31": round(float(row.get("bright_t31", float(row[temp_col]) - 40.0)), 1),
                "recurrence": int(row.get("recurrence_count", 1)),
                "class_id": int(row["pred_class"]),
                "class_name": "Industrial Thermal Anomaly" if is_ind else "Vegetation / Wildfire",
                "prob_industrial": round(float(row.get("probability_industrial", 1.0 if is_ind else 0.0)), 2),
                "color": "#ef4444" if is_ind else "#10b981"
            }
        })

    return {
        "type": "FeatureCollection",
        "total_returned": len(features),
        "features": features
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)