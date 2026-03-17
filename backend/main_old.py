#!/usr/bin/env python3
"""
FastAPI Backend for Olive Yield Forecasting
Serves predictions and historical data via REST API
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Olive Yield Forecasting API",
    description="API for predicting olive yields based on weather data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data files are in the parent directory
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_olive_data():
    """Load olive production data"""
    df = pd.read_csv(os.path.join(DATA_DIR, "elies.csv"))
    df.columns = ["index", "year", "trees", "olives", "oil", "ratio", "price"]
    df = df[["year", "trees", "olives", "oil", "ratio"]].dropna()
    df["year"] = df["year"].astype(int)
    return df


def load_weather_data():
    """Load combined weather data"""
    df = pd.read_csv(os.path.join(DATA_DIR, "Nea_Zichni_scrapped_data_full.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    return df


def load_models():
    """Load trained models - prefers simple model"""
    # Try simple model first
    simple_path = os.path.join(DATA_DIR, "olive_model_simple.pkl")
    if os.path.exists(simple_path):
        with open(simple_path, "rb") as f:
            return pickle.load(f)
    # Fallback to complex model
    model_path = os.path.join(DATA_DIR, "olive_models.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_prediction_data():
    """Get prediction data ready for API"""
    
    models_data = load_models()
    df_olives = load_olive_data()
    df_weather = load_weather_data()
    
    if models_data is None:
        return {"error": "Models not trained. Run olive_predictor.py first."}
    
    models = models_data["models"]
    scaler = models_data["scaler"]
    feature_cols = models_data["feature_cols"]
    
    # Get latest year
    latest_year = df_weather["year"].max()
    
    # Create features for latest year
    df_weather["season"] = df_weather["month"].apply(
        lambda m: "winter" if m in [12, 1, 2] 
        else "spring" if m in [3, 4, 5]
        else "summer" if m in [6, 7, 8]
        else "autumn"
    )
    
    # Yearly aggregates
    yearly = df_weather.groupby("year").agg({
        "Avg_Temp": "mean",
        "Max_Temp": "mean", 
        "Min_Temp": "mean",
        "Rain": "sum",
        "Pressure": "mean",
        "Snow": "sum",
        "Sun_Hours": "sum",
        "Max_Wind": "mean",
        "Avg_Wind": "mean",
        "Avg_Gust": "mean",
        "Clouds": "mean"
    }).reset_index()
    
    yearly.columns = ["year"] + [f"{c}_yearly" for c in yearly.columns[1:]]
    
    # Seasonal features
    seasonal = df_weather.groupby(["year", "season"]).agg({
        "Rain": "sum",
        "Avg_Temp": "mean"
    }).reset_index()
    
    for season in ["winter", "spring", "summer", "autumn"]:
        s = seasonal[seasonal["season"] == season][["year", "Rain", "Avg_Temp"]].copy()
        s.columns = ["year", f"Rain_{season}", f"Avg_Temp_{season}"]
        yearly = yearly.merge(s, on="year", how="left")
    
    # Key months
    for month, col in [(3, "Max_Wind"), (4, "Avg_Temp"), (6, "Max_Temp"), (8, "Clouds"), (9, "Avg_Temp")]:
        m = df_weather[df_weather["month"] == month][["year", col]].copy()
        m.columns = ["year", f"{col}_M{month}"]
        yearly = yearly.merge(m, on="year", how="left")
    
    yearly["Temp_Spread_yearly"] = yearly["Max_Temp_yearly"] - yearly["Min_Temp_yearly"]
    
    # Get latest year data
    latest = yearly[yearly["year"] == latest_year]
    
    if len(latest) == 0:
        return {"error": f"No weather data for year {latest_year}"}
    
    # Handle missing columns in features
    available_cols = [c for c in feature_cols if c in latest.columns]
    missing_cols = [c for c in feature_cols if c not in latest.columns]
    
    if missing_cols:
        print(f"Warning: Missing features in backend: {missing_cols}")
    
    # Create X with available columns, fill missing with mean
    X = latest[available_cols].copy()
    for col in missing_cols:
        X[col] = yearly[col].mean() if col in yearly.columns else 0
    
    # Ensure correct column order
    X = X[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    trees = df_olives[df_olives["year"] == latest_year]["trees"].values
    trees = int(trees[0]) if len(trees) > 0 else int(df_olives["trees"].mean())
    
    predictions = {}
    for name, model in models.items():
        pred = float(max(0, model.predict(X_scaled)[0]))
        predictions[name] = round(pred, 0)
    
    ensemble = float(np.mean(list(predictions.values())))
    avg_ratio = df_olives["ratio"].mean()
    
    return {
        "year": int(latest_year),
        "trees": trees,
        "predictions_olives_kg": predictions,
        "ensemble_olives_kg": round(ensemble, 0),
        "estimated_oil_kg": round(ensemble * avg_ratio, 0),
        "olives_per_tree_kg": round(ensemble / trees, 1),
        "avg_oil_ratio": round(avg_ratio, 3)
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "Olive Yield Forecasting API",
        "version": "1.0.0",
        "endpoints": [
            "/api/prediction",
            "/api/history",
            "/api/weather",
            "/api/dashboard"
        ]
    }


@app.get("/api/prediction")
def get_prediction():
    """Get current year's olive yield prediction"""
    return get_prediction_data()


@app.get("/api/history")
def get_history():
    """Get historical olive production data"""
    df = load_olive_data()
    df = df.sort_values("year")
    return {
        "data": df.to_dict("records"),
        "count": len(df),
        "year_range": {
            "start": int(df["year"].min()),
            "end": int(df["year"].max())
        }
    }


@app.get("/api/weather")
def get_weather(
    years: Optional[int] = Query(10, description="Number of years of weather data")
):
    """Get weather data summary"""
    df = load_weather_data()
    
    # Get yearly aggregates
    yearly = df.groupby("year").agg({
        "Avg_Temp": "mean",
        "Max_Temp": "mean",
        "Min_Temp": "mean",
        "Rain": "sum",
        "Clouds": "mean"
    }).reset_index()
    
    yearly = yearly.sort_values("year").tail(years)
    
    return {
        "data": yearly.to_dict("records"),
        "count": len(yearly),
        "year_range": {
            "start": int(yearly["year"].min()),
            "end": int(yearly["year"].max())
        }
    }


@app.get("/api/dashboard")
def get_dashboard():
    """Get combined dashboard data"""
    
    prediction = get_prediction_data()
    history = load_olive_data().sort_values("year")
    weather = load_weather_data()
    
    # Get recent weather
    latest_year = weather["year"].max()
    latest_weather = weather[weather["year"] == latest_year].agg({
        "Rain": "sum",
        "Avg_Temp": "mean",
        "Clouds": "mean"
    })
    
    # Get historical comparison
    last_year = history.iloc[-1] if len(history) > 0 else None
    avg_historical = {
        "olives": float(history["olives"].mean()) if len(history) > 0 else 0,
        "oil": float(history["oil"].mean()) if len(history) > 0 else 0
    }
    
    return {
        "prediction": prediction,
        "latest_weather": {
            "year": int(latest_year),
            "total_rain_mm": round(float(latest_weather["Rain"]), 1),
            "avg_temp_c": round(float(latest_weather["Avg_Temp"]), 1),
            "avg_clouds_pct": round(float(latest_weather["Clouds"]), 1)
        },
        "last_harvest": {
            "year": int(last_year["year"]) if last_year is not None else None,
            "olives_kg": float(last_year["olives"]) if last_year is not None else None,
            "oil_kg": float(last_year["oil"]) if last_year is not None else None,
            "trees": int(last_year["trees"]) if last_year is not None else None
        } if last_year is not None else None,
        "historical_average": {
            "olives_kg": round(avg_historical["olives"], 0),
            "oil_kg": round(avg_historical["oil"], 0)
        },
        "last_updated": datetime.now().isoformat()
    }


@app.get("/api/models")
def get_models_info():
    """Get information about trained models"""
    models_data = load_models()
    
    if models_data is None:
        return {"error": "Models not trained"}
    
    return {
        "feature_count": len(models_data["feature_cols"]),
        "features": models_data["feature_cols"],
        "models": list(models_data["models"].keys()),
        "top_features": models_data["feature_importance"].head(10).to_dict("records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
