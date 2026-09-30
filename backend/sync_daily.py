import io
import os
import warnings
import joblib
import requests
import pandas as pd
import numpy as np

# Suppress scikit-learn unpickling version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# NASA FIRMS API Configuration
MAP_KEY = "c6e44180f1d2ff237d5550da6318316d"
COUNTRY_CODE = "IND"       # India
SOURCE = "MODIS_NRT"      # Near-real-time daily pass
DAYS = 1                   # Rolling last 24 hours
AREA = "68,6,98,38"        # Geographic Bounding Box for India (west, south, east, north)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_FILE = os.path.join(BASE_DIR, "classified_fires.csv")
MODEL_FILE = os.path.join(BASE_DIR, "firms_industrial_classifier.pkl")


def fetch_live_firms_data() -> pd.DataFrame:
    """
    Fetches live near-real-time active thermal detections from NASA FIRMS.
    Primary query targets Area API with the MAP_KEY; falls back to Country API
    and the South Asia NRT 24h CSV stream if needed.
    """
    endpoints = [
        (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{DAYS}",
            "NASA FIRMS Area API"
        ),
        (
            f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{SOURCE}/{COUNTRY_CODE}/{DAYS}",
            "NASA FIRMS Country API"
        ),
        (
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_24h.csv",
            "NASA FIRMS South Asia 24h Public Stream"
        )
    ]

    last_error = None
    for url, desc in endpoints:
        try:
            print(f"[NASA SYNC] Querying {desc}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                raw_text = response.text.strip()
                if raw_text.startswith("Invalid") or raw_text.startswith("Error"):
                    print(f"[NASA SYNC] {desc} returned notice: {raw_text[:120]}")
                    continue

                live_df = pd.read_csv(io.StringIO(response.text))
                if live_df.empty:
                    print(f"[NASA SYNC] {desc} returned 0 records.")
                    return live_df

                # If retrieved from broader regional fallback, filter to India bounding box
                if "South_Asia" in url:
                    live_df = live_df[
                        (live_df["latitude"] >= 6.0) & (live_df["latitude"] <= 38.0) &
                        (live_df["longitude"] >= 68.0) & (live_df["longitude"] <= 98.0)
                    ].copy()

                print(f"[NASA SYNC] Ingested {len(live_df)} new thermal anomaly records via {desc}.")
                return live_df
            else:
                print(f"[NASA SYNC] {desc} returned HTTP {response.status_code}")
        except Exception as e:
            last_error = e
            print(f"[NASA SYNC] {desc} request exception: {e}")

    if last_error:
        raise ConnectionError(f"FIRMS API connection failed: {last_error}")
    raise ConnectionError("Unable to fetch live FIRMS satellite pass from any configured endpoint.")


def process_and_classify(live_df: pd.DataFrame) -> int:
    """
    Applies feature engineering, spatial recurrence resolution, and Random Forest
    inference on incoming satellite alerts, then appends deduplicated records to catalog.
    """
    if live_df is None or live_df.empty:
        print("[NASA SYNC] No new detections in current satellite swath.")
        return 0

    # Load existing catalog for baseline recurrence
    if os.path.exists(HISTORICAL_FILE):
        hist_df = pd.read_csv(HISTORICAL_FILE)
    else:
        hist_df = pd.DataFrame()

    # Harmonize column names
    temp_col = 'brightness' if 'brightness' in live_df.columns else 'bright_ti4'
    t31_col = 'bright_t31' if 'bright_t31' in live_df.columns else 'bright_ti5'

    # Ensure required columns exist with defaults
    if temp_col not in live_df.columns:
        live_df[temp_col] = 310.0
    if t31_col not in live_df.columns:
        live_df[t31_col] = live_df[temp_col] - 15.0
    if 'frp' not in live_df.columns:
        live_df['frp'] = 15.0
    if 'scan' not in live_df.columns:
        live_df['scan'] = 1.0
    if 'track' not in live_df.columns:
        live_df['track'] = 1.0
    if 'daynight' not in live_df.columns:
        live_df['daynight'] = 'D'
    if 'acq_time' not in live_df.columns:
        live_df['acq_time'] = 1200
    if 'confidence' not in live_df.columns:
        live_df['confidence'] = 75.0

    # Feature Engineering: 0.005 deg (~500m) spatial grid for recurrence
    live_df['lat_cell'] = live_df['latitude'].round(3)
    live_df['lon_cell'] = live_df['longitude'].round(3)

    if not hist_df.empty:
        hist_coords = hist_df[['latitude', 'longitude']].copy()
        hist_coords['lat_cell'] = hist_coords['latitude'].round(3)
        hist_coords['lon_cell'] = hist_coords['longitude'].round(3)
        all_cells = pd.concat([hist_coords[['lat_cell', 'lon_cell']], live_df[['lat_cell', 'lon_cell']]])
        recurrence_map = all_cells.groupby(['lat_cell', 'lon_cell']).size().to_dict()
        live_df['recurrence_count'] = live_df.apply(
            lambda r: recurrence_map.get((r['lat_cell'], r['lon_cell']), 1),
            axis=1
        )
    else:
        live_df['recurrence_count'] = live_df.groupby(['lat_cell', 'lon_cell'])['latitude'].transform('count')

    live_df['temp_diff'] = live_df[temp_col] - live_df[t31_col]
    live_df['is_night'] = (live_df['daynight'].astype(str).str.upper() == 'N').astype(int)
    live_df['hour'] = live_df['acq_time'].astype(str).str.zfill(4).str[:2].astype(int)

    if live_df['confidence'].dtype == object:
        live_df['conf_norm'] = live_df['confidence'].map({'l': 0.25, 'n': 0.65, 'h': 1.0}).fillna(0.65)
    else:
        live_df['conf_norm'] = (live_df['confidence'].astype(float) / 100.0).clip(0.0, 1.0)

    feature_cols = [
        'brightness', 'bright_t31', 'temp_diff', 'frp',
        'scan', 'track', 'is_night', 'conf_norm',
        'recurrence_count', 'hour'
    ]

    # Model Inference
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Classifier model missing at '{MODEL_FILE}'")

    model = joblib.load(MODEL_FILE)
    X = live_df[feature_cols].fillna(0)
    live_df['pred_class'] = model.predict(X)
    live_df['probability_industrial'] = model.predict_proba(X)[:, 1]

    export_cols = [
        'latitude', 'longitude', 'acq_date', 'frp',
        'brightness', 'bright_t31', 'recurrence_count',
        'pred_class', 'probability_industrial'
    ]
    new_formatted = live_df[export_cols].copy()

    initial_count = len(hist_df) if not hist_df.empty else 0
    if not hist_df.empty:
        merged = pd.concat([hist_df, new_formatted], ignore_index=True)
        updated = merged.drop_duplicates(subset=['latitude', 'longitude', 'acq_date']).copy()
    else:
        updated = new_formatted.drop_duplicates(subset=['latitude', 'longitude', 'acq_date']).copy()

    added_count = len(updated) - initial_count
    updated.to_csv(HISTORICAL_FILE, index=False)
    print(f"[NASA SYNC] Done. Added {added_count} distinct detections. Total points in catalog: {len(updated)}")
    return added_count


if __name__ == "__main__":
    print("=== NTRO Sentinel Daily FIRMS Sync Task Triggered ===")
    fresh_data = fetch_live_firms_data()
    count = process_and_classify(fresh_data)
    print(f"=== Execution Complete: {count} new anomalies integrated into classified_fires.csv ===")
