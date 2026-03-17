#!/usr/bin/env python3
"""
Olive Yield Prediction Model
Uses weather data and historical olive production to predict future yields.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_olive_data():
    """Load olive production data"""
    df = pd.read_csv("elies.csv")
    df.columns = ["index", "year", "trees", "olives", "oil", "ratio", "price"]
    df = df[["year", "trees", "olives", "oil", "ratio"]].dropna()
    df["year"] = df["year"].astype(int)
    print(f"Loaded olive data: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    return df


def load_weather_data():
    """Load and combine all weather data"""
    weather_files = {
        "rain": "rain_zichni.csv",
        "temperature": "temperature_zichni.csv",
        "pressure": "pressure_zichni.csv",
        "snow": "snow_zichni.csv",
        "sunhours": "sunhours_zichni.csv",
        "wind": "wind_zichni.csv",
        "clouds": "clouds_zichni.csv"
    }
    
    dfs = {}
    for name, file in weather_files.items():
        try:
            df = pd.read_csv(file)
            # Drop unnamed index columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            # Fix column names: replace spaces with underscores
            df.columns = [c.replace(" ", "_") for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"])
            dfs[name] = df
            print(f"Loaded {name}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Merge all weather data
    df_weather = dfs["temperature"].copy()
    for name, df in dfs.items():
        if name != "temperature":
            df_weather = df_weather.merge(df, on="Date", how="outer")
    
    df_weather = df_weather.sort_values("Date")
    df_weather["year"] = df_weather["Date"].dt.year
    df_weather["month"] = df_weather["Date"].dt.month
    
    return df_weather


def create_features(df_weather):
    """Create monthly features for each year (pivot by month)"""
    
    # Get yearly aggregates
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
    
    # Get seasonal features (critical periods for olive trees)
    seasons = {
        "winter": [12, 1, 2],  # Dormancy
        "spring": [3, 4, 5],   # Flowering
        "summer": [6, 7, 8],   # Fruit development
        "autumn": [9, 10, 11]   # Harvest preparation
    }
    
    df_weather["season"] = df_weather["month"].apply(
        lambda m: "winter" if m in [12, 1, 2] 
        else "spring" if m in [3, 4, 5]
        else "summer" if m in [6, 7, 8]
        else "autumn"
    )
    
    seasonal_features = df_weather.groupby(["year", "season"]).agg({
        "Rain": "sum",
        "Avg_Temp": "mean",
        "Max_Temp": "mean",
        "Min_Temp": "mean"
    }).reset_index()
    
    # Pivot seasons to columns
    for season in seasons.keys():
        season_data = seasonal_features[seasonal_features["season"] == season].copy()
        season_data = season_data.drop("season", axis=1)
        season_data.columns = ["year"] + [f"{c}_{season}" for c in season_data.columns[1:]]
        yearly = yearly.merge(season_data, on="year", how="left")
    
    # Key months for olives (based on R script analysis)
    key_months = {
        3: "Max_Wind",    # March wind
        4: "Avg_Temp",    # April temperature (flowering)
        6: "Max_Temp",    # June heat
        8: "Clouds",      # August cloud cover
        9: "Avg_Temp"     # September temperature
    }
    
    for month, col in key_months.items():
        month_data = df_weather[df_weather["month"] == month][["year", col]].copy()
        month_data.columns = ["year", f"{col}_M{month}"]
        yearly = yearly.merge(month_data, on="year", how="left")
    
    # Temperature spread (diurnal range)
    yearly["Temp_Spread_yearly"] = yearly["Max_Temp_yearly"] - yearly["Min_Temp_yearly"]
    
    return yearly


def prepare_training_data():
    """Prepare final dataset for model training"""
    
    # Load data
    df_olives = load_olive_data()
    df_weather = load_weather_data()
    
    # Create features
    df_features = create_features(df_weather)
    
    # Merge with olive data
    df = df_features.merge(df_olives, on="year", how="inner")
    
    print(f"\nTraining data: {len(df)} years")
    print(f"Features: {len(df.columns) - 4} (excluding year, olives, oil, ratio)")
    
    return df


# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(df):
    """Train and evaluate multiple models"""
    
    feature_cols = [c for c in df.columns if c not in ["year", "olives", "oil", "ratio"]]
    
    X = df[feature_cols].copy()
    y_olives = df["olives"].copy()
    y_oil = df["oil"].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models to try
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y_olives, cv=min(5, len(df)), scoring="neg_mean_absolute_error")
        
        # Fit on full data
        model.fit(X_scaled, y_olives)
        y_pred = model.predict(X_scaled)
        
        mae = mean_absolute_error(y_olives, y_pred)
        rmse = np.sqrt(mean_squared_error(y_olives, y_pred))
        r2 = r2_score(y_olives, y_pred)
        
        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_MAE": -cv_scores.mean()
        }
        
        trained_models[name] = model
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.1f} kg")
        print(f"  RMSE: {rmse:.1f} kg")
        print(f"  R²: {r2:.3f}")
        print(f"  CV MAE: {-cv_scores.mean():.1f} kg")
    
    # Feature importance (from Random Forest)
    rf_model = trained_models["Random Forest"]
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    print(importance.head(10).to_string(index=False))
    
    return trained_models, scaler, feature_cols, importance


def train_and_save_models():
    """Main training pipeline"""
    
    df = prepare_training_data()
    models, scaler, feature_cols, importance = train_models(df)
    
    # Save models
    with open("olive_models.pkl", "wb") as f:
        pickle.dump({
            "models": models,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "feature_importance": importance
        }, f)
    
    print("\n" + "="*60)
    print("Models saved to olive_models.pkl")
    print("="*60)
    
    return df, models, scaler, feature_cols


# ============================================================
# PREDICTION
# ============================================================

def load_models():
    """Load saved models"""
    with open("olive_models.pkl", "rb") as f:
        data = pickle.load(f)
    return data["models"], data["scaler"], data["feature_cols"], data["feature_importance"]


def predict_current_year():
    """Make prediction for the current/next year using latest weather data"""
    
    models, scaler, feature_cols, importance = load_models()
    df = prepare_training_data()
    
    # Get latest year weather data
    df_weather = load_weather_data()
    latest_year = df_weather["year"].max()
    
    # Create features for latest year
    yearly = create_features(df_weather)
    latest_data = yearly[yearly["year"] == latest_year]
    
    if len(latest_data) == 0:
        print(f"No weather data for year {latest_year}")
        return None
    
    # Prepare features - handle missing columns
    available_cols = [c for c in feature_cols if c in latest_data.columns]
    missing_cols = [c for c in feature_cols if c not in latest_data.columns]
    
    if missing_cols:
        print(f"Warning: Missing features: {missing_cols}")
    
    X_pred = latest_data[available_cols].copy()
    
    # Add missing columns with mean values from training data
    for col in missing_cols:
        X_pred[col] = df[feature_cols][col].mean() if col in df.columns else 0
    
    # Ensure correct column order
    X_pred = X_pred[feature_cols]
    X_pred = X_pred.fillna(0)
    X_scaled = scaler.transform(X_pred)
    
    # Get olive data for reference
    df_olives = load_olive_data()
    trees_current = df_olives[df_olives["year"] == latest_year]["trees"].values
    if len(trees_current) > 0:
        trees = trees_current[0]
    else:
        trees = df_olives["trees"].mean()
    
    # Make predictions with each model
    predictions = {}
    print("\n" + "="*60)
    print(f"PREDICTIONS FOR YEAR {latest_year}")
    print("="*60)
    print(f"Number of olive trees: {trees}")
    print()
    
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        pred = max(0, pred)  # Can't have negative yield
        predictions[name] = pred
        
        # Calculate oil estimate (using average ratio)
        avg_ratio = df_olives["ratio"].mean()
        oil_pred = pred * avg_ratio
        
        print(f"{name}:")
        print(f"  Olives: {pred:.0f} kg")
        print(f"  Oil (est): {oil_pred:.0f} kg")
        print()
    
    # Ensemble prediction (average of all models)
    ensemble_pred = np.mean(list(predictions.values()))
    avg_ratio = df_olives["ratio"].mean()
    ensemble_oil = ensemble_pred * avg_ratio
    
    print("="*60)
    print("ENSEMBLE PREDICTION (Average of all models)")
    print("="*60)
    print(f"Olives: {ensemble_pred:.0f} kg")
    print(f"Estimated Oil: {ensemble_oil:.0f} kg")
    print(f"Per tree: {ensemble_pred/trees:.1f} kg")
    
    return {
        "year": latest_year,
        "trees": trees,
        "predictions": predictions,
        "ensemble_olives": ensemble_pred,
        "ensemble_oil": ensemble_oil
    }


def get_historical_predictions():
    """Get predictions for all historical years (for validation)"""
    
    models, scaler, feature_cols, importance = load_models()
    df = prepare_training_data()
    
    feature_cols = [c for c in df.columns if c not in ["year", "olives", "oil", "ratio"]]
    X = df[feature_cols].fillna(df[feature_cols].mean())
    X_scaled = scaler.transform(X)
    
    results = []
    
    for name, model in models.items():
        df[f"pred_{name}"] = model.predict(X_scaled)
    
    # Ensemble
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    df["ensemble_pred"] = df[pred_cols].mean(axis=1)
    df["ensemble_pred"] = df["ensemble_pred"].clip(lower=0)
    
    # Calculate errors
    for col in pred_cols + ["ensemble_pred"]:
        mae = mean_absolute_error(df["olives"], df[col])
        r2 = r2_score(df["olives"], df[col])
        print(f"{col}: MAE={mae:.0f}kg, R²={r2:.3f}")
    
    return df[["year", "olives", "oil"] + pred_cols + ["ensemble_pred"]]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_and_save_models()
        elif sys.argv[1] == "predict":
            predict_current_year()
        elif sys.argv[1] == "history":
            get_historical_predictions()
    else:
        # Default: train and predict
        train_and_save_models()
        predict_current_year()
