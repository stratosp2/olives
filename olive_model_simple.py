#!/usr/bin/env python3
"""
Systematic Feature Selection for Olive Yield Prediction
Tests all possible feature combinations to find statistically significant model
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "."

def load_data():
    olives = pd.read_csv(f"{DATA_DIR}/elies.csv")
    olives.columns = ["idx", "year", "trees", "olives", "oil", "ratio", "price"]
    olives = olives[["year", "trees", "olives", "oil", "ratio"]].dropna()
    olives["year"] = olives["year"].astype(int)
    
    weather = pd.read_csv(f"{DATA_DIR}/Nea_Zichni_scrapped_data_full.csv")
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather["year"] = weather["Date"].dt.year
    weather["month"] = weather["Date"].dt.month
    weather["season"] = weather["month"].apply(lambda m:
        "winter" if m in [12, 1, 2] else
        "spring" if m in [3, 4, 5] else
        "summer" if m in [6, 7, 8] else "autumn")
    
    return olives, weather


def create_all_features(weather):
    """Create all possible features"""
    
    yearly = weather.groupby("year").agg({
        "Avg_Temp": "mean",
        "Max_Temp": "mean",
        "Min_Temp": "mean",
        "Rain": "sum",
        "Snow": "sum",
        "Sun_Hours": "sum",
        "Clouds": "mean",
        "Max_Wind": "mean",
        "Avg_Wind": "mean"
    }).reset_index()
    
    # Seasonal
    for season in ["winter", "spring", "summer", "autumn"]:
        s = weather[weather["season"] == season].groupby("year").agg({
            "Rain": "sum", "Avg_Temp": "mean", "Clouds": "mean"
        }).reset_index()
        s.columns = ["year", f"Rain_{season}", f"Temp_{season}", f"Clouds_{season}"]
        yearly = yearly.merge(s, on="year", how="left")
    
    # Monthly
    for month in range(1, 13):
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m = weather[weather["month"] == month].groupby("year").agg({
            "Rain": "sum", "Avg_Temp": "mean", "Clouds": "mean"
        }).reset_index()
        m.columns = ["year", f"Rain_{month_names[month-1]}", f"Temp_{month_names[month-1]}", f"Clouds_{month_names[month-1]}"]
        yearly = yearly.merge(m, on="year", how="left")
    
    return yearly


def test_model(df, features):
    """Test a model with given features"""
    
    X = df[features].copy().fillna(0)
    y = df["olives"].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Training metrics
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Adjusted R2
    n, p = len(y), X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0
    
    # Cross-validation
    loo = LeaveOneOut()
    cv_scores = cross_val_score(model, X_scaled, y, cv=loo, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    
    # F-test for overall significance
    if p > 0 and r2 > 0:
        f_stat = (r2 / p) / ((1 - r2) / (n - p - 1)) if (n - p - 1) > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1) if (n - p - 1) > 0 else 1
    else:
        f_stat, f_pvalue = 0, 1
    
    return {
        "features": features,
        "n_features": len(features),
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "cv_mae": cv_mae,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "model": model,
        "scaler": scaler,
        "coefficients": dict(zip(features, model.coef_))
    }


def main():
    print("="*70)
    print("SYSTEMATIC FEATURE SELECTION")
    print("="*70)
    
    olives, weather = load_data()
    yearly_weather = create_all_features(weather)
    df = yearly_weather.merge(olives, on="year", how="inner")
    
    print(f"\nData: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    
    # Get all numeric features
    exclude = ["year", "olives", "oil", "ratio", "trees"]
    all_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    print(f"Total features available: {len(all_features)}")
    
    # First: test each feature individually
    print("\n" + "="*70)
    print("SINGLE FEATURE ANALYSIS")
    print("="*70)
    print(f"\n{'Feature':<20} {'r':>8} {'p':>10} {'R²':>8} {'CV MAE':>10}")
    print("-"*60)
    
    single_results = []
    for feat in all_features:
        valid = df[[feat, "olives"]].dropna()
        if len(valid) > 5:
            r, p = stats.pearsonr(valid[feat], valid["olives"])
            result = test_model(df, [feat])
            print(f"{feat:<20} {r:>8.3f} {p:>10.4f} {result['r2']:>8.3f} {result['cv_mae']:>10.0f}")
            single_results.append({
                "feature": feat,
                "r": r,
                "p": p,
                "r2": result["r2"],
                "cv_mae": result["cv_mae"]
            })
    
    # Sort by p-value (most significant first)
    single_results = sorted(single_results, key=lambda x: x["p"])
    
    # Now test combinations of top features
    print("\n" + "="*70)
    print("TOP FEATURE COMBINATIONS")
    print("="*70)
    
    top_features = [r["feature"] for r in single_results[:10] if r["p"] < 0.3]
    
    print(f"\nTesting combinations of top features: {top_features[:6]}")
    
    # Test 1-feature models
    print("\n--- 1 Feature Models ---")
    results = []
    for feat in top_features[:6]:
        r = test_model(df, [feat])
        r["sig"] = r["f_pvalue"] < 0.1
        results.append(r)
        sig_mark = "***" if r["f_pvalue"] < 0.01 else "**" if r["f_pvalue"] < 0.05 else "*" if r["f_pvalue"] < 0.1 else ""
        print(f"{feat:<20} R²={r['r2']:.3f} adj={r['adj_r2']:.3f} CV MAE={r['cv_mae']:.0f} F p={r['f_pvalue']:.4f} {sig_mark}")
    
    # Test 2-feature combinations
    print("\n--- 2 Feature Models ---")
    for i, f1 in enumerate(top_features[:5]):
        for f2 in top_features[i+1:6]:
            r = test_model(df, [f1, f2])
            r["sig"] = r["f_pvalue"] < 0.1
            results.append(r)
            if r["adj_r2"] > 0.3:
                sig_mark = "***" if r["f_pvalue"] < 0.01 else "**" if r["f_pvalue"] < 0.05 else "*" if r["f_pvalue"] < 0.1 else ""
                print(f"{f1}+{f2}: R²={r['r2']:.3f} adj={r['adj_r2']:.3f} CV MAE={r['cv_mae']:.0f} {sig_mark}")
    
    # Test 3-feature combinations
    print("\n--- 3 Feature Models ---")
    for i, f1 in enumerate(top_features[:4]):
        for j, f2 in enumerate(top_features[i+1:5]):
            for f3 in top_features[j+1:6]:
                r = test_model(df, [f1, f2, f3])
                r["sig"] = r["f_pvalue"] < 0.1
                results.append(r)
                if r["adj_r2"] > 0.35:
                    sig_mark = "***" if r["f_pvalue"] < 0.01 else "**" if r["f_pvalue"] < 0.05 else "*" if r["f_pvalue"] < 0.1 else ""
                    print(f"{f1}+{f2}+{f3}: R²={r['r2']:.3f} adj={r['adj_r2']:.3f} CV MAE={r['cv_mae']:.0f} {sig_mark}")
    
    # Find best model by adjusted R2 (simplicity + performance)
    valid_results = [r for r in results if r["adj_r2"] > 0 and r["n_features"] <= 3]
    
    if valid_results:
        # Sort by: 1) significance (True first), 2) adjusted R2 (highest first)
        valid_results = sorted(valid_results, key=lambda x: (-int(bool(x["sig"])), -x["adj_r2"]))
        best = valid_results[0]
        
        print("\n" + "="*70)
        print("BEST MODEL SELECTED")
        print("="*70)
        print(f"\nFeatures: {best['features']}")
        print(f"R²: {best['r2']:.3f}")
        print(f"Adjusted R²: {best['adj_r2']:.3f}")
        print(f"CV MAE: {best['cv_mae']:.0f} kg")
        print(f"F-test p-value: {best['f_pvalue']:.4f} {'(SIGNIFICANT)' if best['sig'] else ''}")
        print(f"\nCoefficients:")
        for feat, coef in best["coefficients"].items():
            print(f"  {feat}: {coef:.2f}")
        
        # Save best model
        with open("olive_model_simple.pkl", "wb") as f:
            pickle.dump({
                "model": best["model"],
                "features": best["features"],
                "scaler": best["scaler"],
                "r2": best["r2"],
                "adj_r2": best["adj_r2"],
                "cv_mae": best["cv_mae"],
                "f_pvalue": best["f_pvalue"]
            }, f)
        
        print(f"\nModel saved!")
        
        # Make prediction
        latest_year = df["year"].max()
        latest = df[df["year"] == latest_year]
        X = latest[best["features"]].fillna(0)
        X_scaled = best["scaler"].transform(X)
        pred = best["model"].predict(X_scaled)[0]
        
        print(f"\n" + "="*70)
        print(f"PREDICTION FOR {latest_year}")
        print("="*70)
        print(f"Olives: {pred:.0f} kg")
        print(f"95% CI: {pred - 2*best['cv_mae']:.0f} - {pred + 2*best['cv_mae']:.0f} kg")
    else:
        print("\nNo statistically significant model found!")
        # Fall back to best single feature
        best_single = single_results[0]
        print(f"\nUsing best single feature: {best_single['feature']}")
        print(f"Correlation: {best_single['r']:.3f}, p-value: {best_single['p']:.4f}")


if __name__ == "__main__":
    main()
