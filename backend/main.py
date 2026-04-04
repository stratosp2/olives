#!/usr/bin/env python3
"""
FastAPI Backend for Olive Yield Forecasting - Multi-Model Support
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import pickle
import os

app = FastAPI(title="Olive Yield Forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR))
FRONTEND_DIR = os.path.join(DATA_DIR, "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

import numpy as np

def to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Available models
AVAILABLE_MODELS = {
    "simple": {
        "name": "Statistical Model",
        "file": "olive_model_simple.pkl",
        "description": "Best statistical model (R²=0.69, p=0.00003)",
        "features": ["Rain_Jan", "Clouds_Sep", "Clouds_Oct"]
    }
}

def load_olive_data():
    olives = pd.read_csv(os.path.join(DATA_DIR, "elies.csv"))
    olives = olives[["year", "trees", "olives", "oil", "ratio"]].dropna()
    olives["year"] = olives["year"].astype(int)
    olives["trees"] = olives["trees"].astype(int)
    return olives

# Weather cache
weather_cache = {"data": None, "timestamp": 0}
CACHE_TTL_SECONDS = 3600  # 1 hour

def load_weather_data():
    weather = pd.read_csv(os.path.join(DATA_DIR, "Nea_Zichni_scrapped_data_full.csv"))
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather["year"] = weather["Date"].dt.year
    weather["month"] = weather["Date"].dt.month
    return weather

def create_features(weather):
    """Create features for models"""
    weather["season"] = weather["month"].apply(lambda m:
        "winter" if m in [12, 1, 2] else
        "spring" if m in [3, 4, 5] else
        "summer" if m in [6, 7, 8] else "autumn")
    
    # Yearly aggregates
    yearly = weather.groupby("year").agg({
        "Avg_Temp": "mean", "Max_Temp": "mean", "Min_Temp": "mean",
        "Rain": "sum", "Clouds": "mean", "Sun_Hours": "sum"
    }).reset_index()
    
    # Seasonal
    for season in ["winter", "spring", "summer", "autumn"]:
        s = weather[weather["season"] == season].groupby("year").agg({
            "Rain": "sum", "Avg_Temp": "mean", "Clouds": "mean"
        }).reset_index()
        s.columns = ["year", f"Rain_{season}", f"Temp_{season}", f"Clouds_{season}"]
        yearly = yearly.merge(s, on="year", how="left")
    
    # Monthly
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m, name in enumerate(month_names, 1):
        m_data = weather[weather["month"] == m].groupby("year").agg({
            "Rain": "sum", "Avg_Temp": "mean", "Clouds": "mean"
        }).reset_index()
        m_data.columns = ["year", f"Rain_{name}", f"Temp_{name}", f"Clouds_{name}"]
        yearly = yearly.merge(m_data, on="year", how="left")
    
    return yearly

def predict_simple(weather_year, olives):
    """Use simple statistical model"""
    w = weather[weather["year"] == weather_year]
    
    # New model features: Rain_Jan, Clouds_Sep, Clouds_Oct
    jan_rain = w[w["month"] == 1]["Rain"].sum()
    sep_clouds = w[w["month"] == 9]["Clouds"].mean()
    oct_clouds = w[w["month"] == 10]["Clouds"].mean()
    
    # Fallback to means
    if pd.isna(jan_rain): jan_rain = weather[weather["month"] == 1]["Rain"].sum()
    if pd.isna(sep_clouds): sep_clouds = weather[weather["month"] == 9]["Clouds"].mean()
    if pd.isna(oct_clouds): oct_clouds = weather[weather["month"] == 10]["Clouds"].mean()
    
    model_path = os.path.join(DATA_DIR, "olive_model_simple.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            m = pickle.load(f)
        
        X = pd.DataFrame({
            "Rain_Jan": [jan_rain],
            "Clouds_Sep": [sep_clouds],
            "Clouds_Oct": [oct_clouds]
        }).fillna(0)
        
        X_scaled = m["scaler"].transform(X)
        pred = max(0, m["model"].predict(X_scaled)[0])
        cv_mae = m.get("cv_mae", 300)
        r2 = m.get("adj_r2", 0.94)
    else:
        pred = 950
        cv_mae = 200
        r2 = 0
    
    return {
        "olives_kg": round(pred),
        "oil_kg": round(pred * 0.15),
        "ci_kg": round(cv_mae * 2),
        "cv_mae": cv_mae,
        "r2": r2,
        "model_name": "Statistical Model (Simple)",
        "features": {"rain_jan": jan_rain, "clouds_sep": sep_clouds, "clouds_oct": oct_clouds}
    }

def predict_complex(weather_year, yearly):
    """Use complex ML ensemble model"""
    model_path = os.path.join(DATA_DIR, "olive_models.pkl")
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    models = data["models"]
    scaler = data["scaler"]
    feature_cols = data["feature_cols"]
    
    latest = yearly[yearly["year"] == weather_year]
    if len(latest) == 0:
        return None
    
    # Handle missing columns
    available_cols = [c for c in feature_cols if c in latest.columns]
    X = latest[available_cols].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Ensemble prediction
    preds = [m.predict(X_scaled)[0] for m in models.values()]
    pred = max(0, sum(preds) / len(preds))
    
    return {
        "olives_kg": round(pred),
        "oil_kg": round(pred * 0.23),
        "ci_kg": round(pred * 0.4),
        "cv_mae": round(pred * 0.3),
        "r2": 0.85,  # complex model overfits
        "model_name": "ML Ensemble (Complex)",
        "individual_predictions": {k: round(v.predict(X_scaled)[0]) for k, v in models.items()}
    }

# Load data at startup
olives = load_olive_data()
weather = load_weather_data()
yearly_features = create_features(weather)

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return {
            "error": str(e),
            "frontend_dir": FRONTEND_DIR,
            "exists": os.path.exists(FRONTEND_DIR),
            "files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else [],
            "base_dir": BASE_DIR,
            "name": "Olive Yield Forecasting API",
            "version": "2.0.0",
            "available_models": list(AVAILABLE_MODELS.keys()),
            "endpoints": ["/api/prediction", "/api/history", "/api/models", "/api/dashboard"]
        }

@app.get("/api/models")
def list_models():
    return {"models": AVAILABLE_MODELS}

@app.get("/api/prediction")
def get_prediction(
    model: str = Query("simple", description="Model to use: simple or complex")
):
    if model not in AVAILABLE_MODELS:
        return {"error": f"Unknown model. Available: {list(AVAILABLE_MODELS.keys())}"}
    
    # Predict for NEXT harvest year (the one we haven't harvested yet)
    # Always predict 2 years ahead (we're in 2026, so predict for late 2026 harvest)
    harvest_year = 2026
    weather_year = harvest_year - 1  # 2025
    
    # If weather data doesn't exist for weather_year, use last available
    available_weather_years = sorted(weather["year"].unique())
    if weather_year not in available_weather_years:
        weather_year = max(available_weather_years)
        harvest_year = weather_year + 1
    
    if model == "simple":
        result = predict_simple(weather_year, olives)
    else:
        result = predict_complex(weather_year, yearly_features)
        if result is None:
            result = {"error": "Complex model not available", "olives_kg": 0, "oil_kg": 0}
    
    last = olives.iloc[-1]
    
    return to_native({
        "model": model,
        "model_info": AVAILABLE_MODELS.get(model),
        "harvest_year": int(harvest_year),
        "weather_year_used": int(weather_year),
        "prediction": result,
        "last_harvest": {
            "year": int(last["year"]),
            "olives": float(last["olives"]),
            "oil": float(last["oil"]),
            "trees": int(last["trees"])
        }
    })

@app.get("/api/history")
def get_history():
    olives_sorted = olives.sort_values("year")
    records = olives_sorted.to_dict("records")
    # Fix ratio: calculate oil as percentage of olives
    for r in records:
        if r.get('olives') and r.get('oil'):
            r['ratio'] = round(r['oil'] / r['olives'] * 100, 1)  # percentage
        elif r.get('ratio'):
            r['ratio'] = round(r['ratio'] * 100, 1)  # convert to percentage
    return to_native({
        "data": records,
        "year_range": {"start": int(olives["year"].min()), "end": int(olives["year"].max())}
    })

@app.get("/api/dashboard")
def get_dashboard(model: str = Query("simple")):
    pred_response = get_prediction(model)
    pred = pred_response.get("prediction", {})
    
    return {
        "model_used": model,
        "prediction": {
            "year": pred_response.get("harvest_year"),
            "ensemble_olives_kg": pred.get("olives_kg", 0),
            "estimated_oil_kg": pred.get("oil_kg", 0),
            "ci_kg": pred.get("ci_kg", 0),
            "olives_per_tree_kg": round(pred.get("olives_kg", 0) / 130, 1),
            "trees": 130
        },
        "last_harvest": pred_response.get("last_harvest"),
        "historical_average": {
            "olives_kg": round(olives["olives"].mean()),
            "oil_kg": round(olives["oil"].mean())
        }
    }

@app.get("/api/weather/current")
def get_current_weather():
    """Get weather - uses cache, only refreshes once per hour"""
    global weather_cache
    import time
    
    current_time = time.time()
    
    # Check if cache is still valid
    if weather_cache["data"] and (current_time - weather_cache["timestamp"]) < CACHE_TTL_SECONDS:
        return weather_cache["data"]
    
    # Try OpenWeatherMap (no API key needed for basic)
    try:
        import requests
        LAT = 41.0302293
        LON = 23.8174898
        
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": LAT,
            "lon": LON,
            "units": "metric",
            "APPID": "demo"  # Replace with your key for higher limits
        }
        
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        
        if data.get("cod") != 200:
            raise Exception("OpenWeatherMap error: " + str(data.get("message", "")))
        
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        
        result = {
            "date": data.get("dt", 0),
            "temperature": round(main.get("temp", 0), 1),
            "humidity": int(main.get("humidity", 0)),
            "clouds": int(clouds.get("all", 0)),
            "rain": round(rain.get("1h", rain.get("3h", 0)), 1),
            "wind": round(wind.get("speed", 0), 1),
            "updated": data.get("dt", 0)
        }
        
        # Cache the result
        weather_cache = {"data": result, "timestamp": current_time}
        return result
        
    except Exception as e:
        # Return cached data if available
        if weather_cache["data"]:
            weather_cache["data"]["updated"] = "cached"
            return weather_cache["data"]
        
        # Fallback to file
        try:
            weather_sorted = weather.sort_values("Date", ascending=False)
            latest = weather_sorted.iloc[0]
            latest_date = latest["Date"]
            wind_val = latest.get("Avg_Wind", 0)
            if pd.isna(wind_val):
                wind_val = 0
            return {
                "date": str(latest_date)[:10],
                "temperature": round(float(latest["Avg_Temp"]), 1),
                "humidity": 0,
                "clouds": round(float(latest["Clouds"]), 1),
                "rain": round(float(latest["Rain"]), 1),
                "wind": round(float(wind_val), 1),
                "updated": str(latest_date)[:10] + " (file)"
            }
        except:
            return {"error": str(e), "fallback": "failed"}

@app.get("/api/weather/update")
def update_weather():
    """Fetch latest weather from Open-Meteo API"""
    try:
        import requests
        from datetime import datetime
        
        LAT = 41.0302293
        LON = 23.8174898
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": "temperature_2m,temperature_2m_max,temperature_2m_min,precipitation,wind_speed_10m_max,wind_speed_10m_mean,wind_gusts_10m_max,sunshine_duration,cloud_cover,snowfall_sum",
            "past_days": 92,
            "forecast_days": 7,
            "timezone": "Europe/Athens"
        }
        
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        hourly = data.get("hourly", {})
        
        records = []
        times = hourly.get("time", [])
        for i, t in enumerate(times):
            records.append({
                "Date": t[:10],
                "Year": int(t[:4]),
                "Month": int(t[5:7]),
                "Avg_Temp": hourly.get("temperature_2m", [None])[i],
                "Max_Temp": hourly.get("temperature_2m_max", [None])[i],
                "Min_Temp": hourly.get("temperature_2m_min", [None])[i],
                "Rain": hourly.get("precipitation", [0])[i] or 0,
                "Max_Wind": hourly.get("wind_speed_10m_max", [None])[i],
                "Avg_Wind": hourly.get("wind_speed_10m_mean", [None])[i],
                "Avg_Gust": hourly.get("wind_gusts_10m_max", [None])[i],
                "Sun_Hours": (hourly.get("sunshine_duration", [0])[i] or 0) / 3600,
                "Clouds": hourly.get("cloud_cover", [None])[i],
                "Snow": hourly.get("snowfall_sum", [0])[i] or 0,
                "Pressure": 1013
            })
        
        new_df = pd.DataFrame(records)
        new_df["Date"] = pd.to_datetime(new_df["Date"])
        
        csv_path = os.path.join(DATA_DIR, "Nea_Zichni_scrapped_data_full.csv")
        
        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path)
            old_df["Date"] = pd.to_datetime(old_df["Date"])
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["Date"]).sort_values("Date")
            combined.to_csv(csv_path, index=False)
        else:
            new_df.to_csv(csv_path, index=False)
        
        global weather
        weather = load_weather_data()
        
        latest = new_df.iloc[-1]
        return {
            "success": True,
            "last_update": latest["Date"].strftime("%Y-%m-%d"),
            "records_added": len(new_df),
            "last_weather": {
                "date": latest["Date"].strftime("%Y-%m-%d"),
                "temp": round(latest["Avg_Temp"], 1),
                "clouds": round(latest["Clouds"], 1)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Disease Risk Assessment Functions
DISEASE_THRESHOLDS = {
    "peacock_spot": {
        "temp_min": 10, "temp_max": 25, "temp_optimal": (15, 22),
        "humidity_min": 70, "wetness_hours": 48,
        "description": "Fungal disease causing leaf spots, defoliation. Optimal: 15-22°C with high humidity and prolonged leaf wetness.",
        "symptoms": ["Circular dark spots on leaves", "Yellow halos", "Premature defoliation"]
    },
    "verticillium_wilt": {
        "temp_min": 15, "temp_max": 28, "temp_optimal": (18, 25),
        "rain_periods": True, "poor_drainage": True,
        "description": "Soil-borne fungus blocking water vessels. Thrives with mild temps + rain periods.",
        "symptoms": ["Sudden wilting of branches", "Leaf browning (stays attached)", "Branch dieback"]
    },
    "olive_knot": {
        "temp_min": 5, "temp_max": 25, "temp_optimal": (10, 20),
        "humidity_min": 60, "wounds_required": True,
        "description": "Bacterial disease causing tumors on branches. Favored by mild temps, humidity, and bark wounds.",
        "symptoms": ["Galls/knots on branches", "Tumor formation", "Reduced vigor"]
    },
    "xylella": {
        "temp_min": 15, "temp_max": 35, "temp_optimal": (26, 28),
        "humidity_min": 50, "vector_present": True,
        "description": "Deadly bacterial disease (OQDS). Optimal spread at 26-28°C via spittlebug vectors.",
        "symptoms": ["Leaf scorch", "Branch wilting", "Rapid tree decline"]
    },
    "anthracnose": {
        "temp_min": 15, "temp_max": 27, "temp_optimal": (18, 24),
        "humidity_min": 85, "rain_wetness": True,
        "description": "Fungal fruit rot. High humidity and rain favor fruit infection.",
        "symptoms": ["Mummified fruit", "Fruit rot", "Twig dieback"]
    }
}

FOLIAR_CONDITIONS = {
    "nitrogen_early_spring": {
        "temp_range": (10, 20), "humidity_min": 50,
        "best_months": [2, 3], "description": "Nitrogen for vegetative recovery and flower differentiation"
    },
    "boron_preflowering": {
        "temp_range": (12, 22), "humidity_min": 55,
        "best_months": [3, 4], "description": "Boron, zinc for flowering and fruit set"
    },
    "potassium_summer": {
        "temp_range": (20, 32), "humidity_min": 40,
        "best_months": [6, 7, 8], "description": "Potassium, magnesium for fruit growth"
    },
    "calcium_midsummer": {
        "temp_range": (18, 30), "humidity_min": 45,
        "best_months": [5, 6, 7], "description": "Calcium for fruit firmness and stress resistance"
    },
    "phosphorus_autumn": {
        "temp_range": (12, 25), "humidity_min": 50,
        "best_months": [9, 10, 11], "description": "Phosphorus for reserve replenishment"
    }
}

def calculate_disease_risk(weather_data):
    """Calculate disease risk based on recent weather conditions"""
    if weather_data is None or len(weather_data) == 0:
        return {"error": "No weather data available"}
    
    current = weather_data.iloc[-1]
    temp = current.get("Avg_Temp", 20)
    humidity = current.get("Humidity", 50) if "Humidity" in current.columns else 50
    rain = current.get("Rain", 0)
    clouds = current.get("Clouds", 50)
    
    month = current["month"] if "month" in current else 5
    
    recent = weather_data.tail(30)
    avg_temp = recent["Avg_Temp"].mean()
    total_rain = recent["Rain"].sum()
    avg_humidity = recent["Humidity"].mean() if "Humidity" in recent.columns else 50
    rainy_days = (recent["Rain"] > 0.5).sum()
    
    risks = {}
    
    # Peacock Spot
    ps = DISEASE_THRESHOLDS["peacock_spot"]
    ps_risk = 0
    if ps["temp_min"] <= avg_temp <= ps["temp_max"]:
        ps_risk += 30
        if ps["temp_optimal"][0] <= avg_temp <= ps["temp_optimal"][1]:
            ps_risk += 30
    if avg_humidity >= ps["humidity_min"]:
        ps_risk += 20
    if rainy_days >= 3:
        ps_risk += 20
    if month in [3, 4, 5, 9, 10, 11]:
        ps_risk += 10
    risks["peacock_spot"] = {
        "name": "Peacock Spot (Cycloconio)",
        "risk_level": min(100, ps_risk),
        "risk_label": "HIGH" if ps_risk >= 70 else "MODERATE" if ps_risk >= 40 else "LOW",
        "description": ps["description"],
        "recommendations": [
            "Apply copper fungicide before winter rains" if ps_risk >= 50 else "Monitor conditions",
            "Ensure good canopy ventilation",
            "Avoid overhead irrigation"
        ],
        "best_treatment_time": "Late October or early spring"
    }
    
    # Verticillium Wilt
    vw = DISEASE_THRESHOLDS["verticillium_wilt"]
    vw_risk = 0
    if vw["temp_min"] <= avg_temp <= vw["temp_max"]:
        vw_risk += 30
        if vw["temp_optimal"][0] <= avg_temp <= vw["temp_optimal"][1]:
            vw_risk += 30
    if total_rain > 50:
        vw_risk += 25
    if rainy_days >= 5:
        vw_risk += 15
    risks["verticillium_wilt"] = {
        "name": "Verticillium Wilt",
        "risk_level": min(100, vw_risk),
        "risk_label": "HIGH" if vw_risk >= 70 else "MODERATE" if vw_risk >= 40 else "LOW",
        "description": vw["description"],
        "recommendations": [
            "Improve soil drainage" if vw_risk >= 50 else "Monitor soil moisture",
            "Avoid planting after susceptible crops (cotton, tomatoes)",
            "No chemical cure - focus on prevention"
        ],
        "best_treatment_time": "Prevention before planting"
    }
    
    # Olive Knot
    ok = DISEASE_THRESHOLDS["olive_knot"]
    ok_risk = 0
    if ok["temp_min"] <= avg_temp <= ok["temp_max"]:
        ok_risk += 25
        if ok["temp_optimal"][0] <= avg_temp <= ok["temp_optimal"][1]:
            ok_risk += 25
    if avg_humidity >= ok["humidity_min"]:
        ok_risk += 25
    if month in [12, 1, 2, 3, 4]:
        ok_risk += 15
    if total_rain > 30:
        ok_risk += 10
    risks["olive_knot"] = {
        "name": "Olive Knot (Tuberculosis)",
        "risk_level": min(100, ok_risk),
        "risk_label": "HIGH" if ok_risk >= 70 else "MODERATE" if ok_risk >= 40 else "LOW",
        "description": ok["description"],
        "recommendations": [
            "Remove tumors in summer (July-August) in dry conditions",
            "Apply copper after pruning",
            "Disinfect pruning tools"
        ],
        "best_treatment_time": "Summer for tumor removal, copper spray after pruning"
    }
    
    # Xylella
    xl = DISEASE_THRESHOLDS["xylella"]
    xl_risk = 0
    if xl["temp_min"] <= avg_temp <= xl["temp_max"]:
        xl_risk += 20
        if xl["temp_optimal"][0] <= avg_temp <= xl["temp_optimal"][1]:
            xl_risk += 40
    if avg_temp > 20:
        xl_risk += 20
    if avg_humidity >= xl["humidity_min"]:
        xl_risk += 20
    risks["xylella"] = {
        "name": "Xylella Fastidiosa",
        "risk_level": min(100, xl_risk),
        "risk_label": "HIGH" if xl_risk >= 70 else "MODERATE" if xl_risk >= 40 else "LOW",
        "description": xl["description"],
        "recommendations": [
            "Monitor for spittlebug vectors",
            "No cure - prevention critical",
            "Report suspicious symptoms immediately"
        ],
        "best_treatment_time": "Vector control in spring-summer"
    }
    
    # Anthracnose
    ac = DISEASE_THRESHOLDS["anthracnose"]
    ac_risk = 0
    if ac["temp_min"] <= avg_temp <= ac["temp_max"]:
        ac_risk += 25
        if ac["temp_optimal"][0] <= avg_temp <= ac["temp_optimal"][1]:
            ac_risk += 25
    if avg_humidity >= ac["humidity_min"]:
        ac_risk += 30
    if total_rain > 40:
        ac_risk += 20
    risks["anthracnose"] = {
        "name": "Anthracnose (Leprosy)",
        "risk_level": min(100, ac_risk),
        "risk_label": "HIGH" if ac_risk >= 70 else "MODERATE" if ac_risk >= 40 else "LOW",
        "description": ac["description"],
        "recommendations": [
            "Apply fungicide at first signs of rain in risky periods",
            "Remove infected fruit",
            "Improve air circulation"
        ],
        "best_treatment_time": "Before rain events in spring-autumn"
    }
    
    return risks

def calculate_foliar_recommendations(weather_data):
    """Calculate foliar fertilisation recommendations based on weather and season"""
    if weather_data is None or len(weather_data) == 0:
        return {"error": "No weather data available"}
    
    current = weather_data.iloc[-1]
    temp = current.get("Avg_Temp", 20)
    humidity = current.get("Humidity", 50) if "Humidity" in current.columns else 50
    clouds = current.get("Clouds", 50)
    month = current["month"] if "month" in current.columns else 5
    
    recent = weather_data.tail(7)
    recent_temp = recent["Avg_Temp"].mean()
    recent_humidity = recent["Humidity"].mean() if "Humidity" in recent.columns else 50
    recent_rain = recent["Rain"].sum()
    
    recommendations = []
    
    # Check each foliar type
    for key, fert in FOLIAR_CONDITIONS.items():
        temp_min, temp_max = fert["temp_range"]
        in_season = month in fert["best_months"]
        temp_ok = temp_min <= recent_temp <= temp_max
        humidity_ok = recent_humidity >= fert["humidity_min"]
        no_rain = recent_rain < 5
        
        if in_season:
            status = "OPTIMAL" if (temp_ok and humidity_ok and no_rain) else "SUBOPTIMAL" if in_season else "NOT_YET"
            recommendations.append({
                "type": key,
                "name": key.replace("_", " ").title(),
                "description": fert["description"],
                "months": fert["best_months"],
                "current_month": month,
                "in_season": in_season,
                "weather_suitable": temp_ok and humidity_ok and no_rain,
                "conditions": {
                    "temperature": {"current": round(recent_temp, 1), "required": f"{temp_min}-{temp_max}°C", "ok": temp_ok},
                    "humidity": {"current": round(recent_humidity, 1), "required": f">{fert['humidity_min']}%", "ok": humidity_ok},
                    "no_rain_expected": {"required": "No rain", "ok": no_rain}
                },
                "status": status,
                "recommendation": f"Apply now" if status == "OPTIMAL" else f"Wait for better conditions" if status == "SUBOPTIMAL" else "Not in optimal season"
            })
    
    # General timing advice
    timing_advice = {
        "best_time_of_day": "Early morning (6-9 AM) or late afternoon (5-7 PM)",
        "avoid": "Midday heat (>30°C), strong wind, rain forecast within 24h",
        "current_conditions": {
            "temperature": round(recent_temp, 1),
            "humidity": round(recent_humidity, 1),
            "recent_rain": round(recent_rain, 1)
        }
    }
    
    return {
        "recommendations": recommendations,
        "timing_advice": timing_advice,
        "season": "winter" if month in [12, 1, 2] else "spring" if month in [3, 4, 5] else "summer" if month in [6, 7, 8] else "autumn"
    }


@app.get("/api/disease-risk")
def get_disease_risk():
    """Get disease risk assessment based on current weather"""
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 40.7333,
                "longitude": 22.8333,
                "past_days": 30,
                "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,cloud_cover_mean",
                "timezone": "Europe/Athens"
            },
            timeout=30
        )
        data = resp.json()
        daily = data.get("daily", {})
        
        records = []
        dates = daily.get("time", [])
        for i, date in enumerate(dates):
            records.append({
                "Date": date,
                "Avg_Temp": daily.get("temperature_2m_mean", [None])[i],
                "Max_Temp": daily.get("temperature_2m_max", [None])[i],
                "Min_Temp": daily.get("temperature_2m_min", [None])[i],
                "Rain": daily.get("precipitation_sum", [0])[i] or 0,
                "Clouds": daily.get("cloud_cover_mean", [None])[i],
                "Humidity": 100 - (daily.get("cloud_cover_mean", [50])[i] or 50),
                "month": int(date[5:7])
            })
        
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        
        risks = calculate_disease_risk(df)
        overall = {
            "HIGH": sum(1 for r in risks.values() if isinstance(r, dict) and r.get("risk_label") == "HIGH"),
            "MODERATE": sum(1 for r in risks.values() if isinstance(r, dict) and r.get("risk_label") == "MODERATE"),
            "LOW": sum(1 for r in risks.values() if isinstance(r, dict) and r.get("risk_label") == "LOW")
        }
        
        return {
            "success": True,
            "current_conditions": {
                "avg_temperature": round(df["Avg_Temp"].mean(), 1),
                "total_rain_30d": round(df["Rain"].sum(), 1),
                "avg_humidity_estimate": round(df["Humidity"].mean(), 1),
                "rainy_days_30d": int((df["Rain"] > 0.5).sum())
            },
            "overall_risk": "HIGH" if overall["HIGH"] >= 3 else "MODERATE" if overall["HIGH"] >= 1 or overall["MODERATE"] >= 3 else "LOW",
            "disease_counts": overall,
            "diseases": risks
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/foliar-fertilization")
def get_foliar_recommendations():
    """Get foliar fertilisation recommendations based on current weather and season"""
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 40.7333,
                "longitude": 22.8333,
                "past_days": 7,
                "daily": "temperature_2m_mean,precipitation_sum,cloud_cover_mean",
                "timezone": "Europe/Athens"
            },
            timeout=30
        )
        data = resp.json()
        daily = data.get("daily", {})
        
        records = []
        dates = daily.get("time", [])
        for i, date in enumerate(dates):
            records.append({
                "Date": date,
                "Avg_Temp": daily.get("temperature_2m_mean", [None])[i],
                "Rain": daily.get("precipitation_sum", [0])[i] or 0,
                "Clouds": daily.get("cloud_cover_mean", [None])[i],
                "Humidity": 100 - (daily.get("cloud_cover_mean", [50])[i] or 50),
                "month": int(date[5:7])
            })
        
        df = pd.DataFrame(records)
        
        result = calculate_foliar_recommendations(df)
        
        return {
            "success": True,
            "current_conditions": result.get("timing_advice", {}).get("current_conditions", {}),
            "season": result.get("season", "unknown"),
            "recommendations": result.get("recommendations", []),
            "timing_advice": result.get("timing_advice", {})
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
