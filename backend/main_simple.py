#!/usr/bin/env python3
"""
FastAPI Backend for Olive Yield Forecasting - Simplified Version
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import os

app = FastAPI(title="Olive Yield Forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_simple_prediction():
    """Use the simplified model directly"""
    olives = pd.read_csv(os.path.join(DATA_DIR, "elies.csv"))
    olives.columns = ["idx", "year", "trees", "olives", "oil", "ratio", "price"]
    olives = olives[["year", "trees", "olives", "oil", "ratio"]].dropna()
    olives["year"] = olives["year"].astype(int)

    weather = pd.read_csv(os.path.join(DATA_DIR, "Nea_Zichni_scrapped_data_full.csv"))
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather["year"] = weather["Date"].dt.year
    weather["month"] = weather["Date"].dt.month

    # Simple model: predict harvest for year N using weather from year N-1
    # For 2026 harvest, use 2025 weather
    harvest_year = 2026
    weather_year = harvest_year - 1
    
    w = weather[weather["year"] == weather_year]
    aug_clouds = w[w["month"] == 8]["Clouds"].mean()
    dec_clouds = w[w["month"] == 12]["Clouds"].mean()
    jul_temp = w[w["month"] == 7]["Avg_Temp"].mean()
    
    # Use historical means if missing
    if pd.isna(aug_clouds):
        aug_clouds = weather[weather["month"] == 8]["Clouds"].mean()
    if pd.isna(dec_clouds):
        dec_clouds = weather[weather["month"] == 12]["Clouds"].mean()
    if pd.isna(jul_temp):
        jul_temp = weather[weather["month"] == 7]["Avg_Temp"].mean()
    
    # Load simple model
    model_path = os.path.join(DATA_DIR, "olive_model_simple.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            m = pickle.load(f)
        
        X = pd.DataFrame({
            "Clouds_Aug": [aug_clouds],
            "Clouds_Dec": [dec_clouds],
            "Temp_Jul": [jul_temp]
        }).fillna(0)
        
        X_scaled = m["scaler"].transform(X)
        pred = max(0, m["model"].predict(X_scaled)[0])
        cv_mae = m.get("cv_mae", 800)
    else:
        pred = 950  # fallback
        cv_mae = 800
    
    last = olives.iloc[-1] if len(olives) > 0 else None
    
    return {
        "year": harvest_year,
        "weather_year": weather_year,
        "olives_kg": round(pred),
        "oil_kg": round(pred * 0.23),
        "ci_kg": round(cv_mae * 2),
        "trees": int(last["trees"]) if last is not None else 130,
        "last_harvest": {
            "year": int(last["year"]) if last is not None else None,
            "olives": float(last["olives"]) if last is not None else None,
            "oil": float(last["oil"]) if last is not None else None,
            "trees": int(last["trees"]) if last is not None else None
        } if last is not None else None,
        "historical_avg": {
            "olives": round(olives["olives"].mean()),
            "oil": round(olives["oil"].mean())
        },
        "weather": {
            "aug_clouds": round(aug_clouds, 1),
            "dec_clouds": round(dec_clouds, 1),
            "jul_temp": round(jul_temp, 1)
        }
    }

@app.get("/")
def root():
    return {"name": "Olive Yield Forecasting API", "version": "1.0.0", "endpoints": ["/api/prediction", "/api/history"]}

@app.get("/api/prediction")
def get_prediction():
    return load_simple_prediction()

@app.get("/api/history")
def get_history():
    olives = pd.read_csv(os.path.join(DATA_DIR, "elies.csv"))
    olives.columns = ["idx", "year", "trees", "olives", "oil", "ratio", "price"]
    olives = olives[["year", "trees", "olives", "oil", "ratio"]].dropna()
    olives["year"] = olives["year"].astype(int)
    olives = olives.sort_values("year")
    
    return {
        "data": olives.to_dict("records"),
        "year_range": {"start": int(olives["year"].min()), "end": int(olives["year"].max())}
    }

@app.get("/api/dashboard")
def get_dashboard():
    pred = load_simple_prediction()
    return {
        "prediction": {
            "year": pred["year"],
            "ensemble_olives_kg": pred["olives_kg"],
            "estimated_oil_kg": pred["oil_kg"],
            "olives_per_tree_kg": round(pred["olives_kg"] / pred["trees"], 1),
            "trees": pred["trees"]
        },
        "last_harvest": pred.get("last_harvest"),
        "historical_average": pred.get("historical_avg"),
        "latest_weather": {
            "year": pred.get("weather_year"),
            "avg_clouds_pct": pred.get("weather", {}).get("aug_clouds", 0)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
