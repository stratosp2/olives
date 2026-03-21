#!/usr/bin/env python3
"""
Weather data scraper for Nea Zichni using Open-Meteo API
Fetches historical data + recent data from forecast API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

LAT = 40.7333
LON = 22.8333

def fetch_historical(start_year=2009, end_year=2025):
    """Fetch historical weather data"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_speed_10m_mean,wind_gusts_10m_max,sunshine_duration,cloud_cover_mean,snowfall_sum",
        "timezone": "Europe/Athens"
    }
    
    print(f"Fetching historical data {start_year}-{end_year}...")
    resp = requests.get(url, params=params, timeout=120)
    data = resp.json()
    daily = data.get("daily", {})
    
    df = pd.DataFrame({
        "Date": daily.get("time", []),
        "Avg_Temp": daily.get("temperature_2m_mean", []),
        "Max_Temp": daily.get("temperature_2m_max", []),
        "Min_Temp": daily.get("temperature_2m_min", []),
        "Rain": daily.get("precipitation_sum", []),
        "Max_Wind": daily.get("wind_speed_10m_max", []),
        "Avg_Wind": daily.get("wind_speed_10m_mean", []),
        "Avg_Gust": daily.get("wind_gusts_10m_max", []),
        "Sun_Hours": [s/3600 for s in daily.get("sunshine_duration", [])],
        "Clouds": daily.get("cloud_cover_mean", []),
        "Snow": daily.get("snowfall_sum", [])
    })
    
    print(f"  Downloaded {len(df)} daily records")
    return df


def fetch_recent():
    """Fetch recent data from forecast API (includes past 92 days)"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,cloud_cover,sunshine_duration",
        "past_days": 92,
        "forecast_days": 7,
        "timezone": "Europe/Athens"
    }
    
    print("Fetching recent data...")
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    hourly = data.get("hourly", {})
    
    df = pd.DataFrame({
        "DateTime": hourly.get("time", []),
        "Temp": hourly.get("temperature_2m", []),
        "Rain": hourly.get("precipitation", []),
        "Wind": hourly.get("wind_speed_10m", []),
        "Clouds": hourly.get("cloud_cover", []),
        "Sun_Hours": [s/3600 if s else 0 for s in hourly.get("sunshine_duration", [])]
    })
    
    # Convert to daily
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.date
    
    daily = df.groupby("Date").agg({
        "Temp": "mean",
        "Rain": "sum",
        "Wind": "mean",
        "Clouds": "mean",
        "Sun_Hours": "sum"
    }).reset_index()
    
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.rename(columns={"Temp": "Avg_Temp"})
    
    # Estimate max/min temp (rough approximation)
    daily["Max_Temp"] = daily["Avg_Temp"] + 5
    daily["Min_Temp"] = daily["Avg_Temp"] - 5
    
    print(f"  Downloaded {len(daily)} recent daily records")
    return daily[["Date", "Avg_Temp", "Max_Temp", "Min_Temp", "Rain", "Wind", "Sun_Hours", "Clouds"]]


def main():
    print("="*60)
    print("Weather Data Scraper - Nea Zichni")
    print("="*60)
    
    # Fetch historical (2009-2025)
    df_hist = fetch_historical(2009, 2025)
    
    # Convert dates to datetime
    df_hist["Date"] = pd.to_datetime(df_hist["Date"])
    
    # Fetch recent data
    df_recent = fetch_recent()
    
    # Find the cutoff - where recent data starts
    cutoff = df_recent["Date"].min() - timedelta(days=1)
    print(f"  Using data before {cutoff} from historical")
    
    # Filter historical to only include data before recent
    df_hist = df_hist[df_hist["Date"] <= cutoff]
    
    # Combine
    df_all = pd.concat([df_hist, df_recent], ignore_index=True)
    df_all = df_all.sort_values("Date").drop_duplicates("Date")
    
    print(f"\nTotal records: {len(df_all)}")
    print(f"Date range: {df_all['Date'].min()} to {df_all['Date'].max()}")
    
    # Add year/month
    df_all["Year"] = df_all["Date"].dt.year
    df_all["Month"] = df_all["Date"].dt.month
    
    # Save individual files (matching original format)
    df_all[["Date", "Avg_Temp", "Max_Temp", "Min_Temp"]].to_csv("temperature_zichni.csv", index=False)
    df_all[["Date", "Rain"]].to_csv("rain_zichni.csv", index=False)
    df_all[["Date", "Max_Wind", "Avg_Wind", "Avg_Gust"]].rename(columns={"Max_Wind": "Max Wind", "Avg_Wind": "Avg Wind", "Avg_Gust": "Avg Gust"}).to_csv("wind_zichni.csv", index=False)
    df_all[["Date", "Sun_Hours"]].to_csv("sunhours_zichni.csv", index=False)
    df_all[["Date", "Clouds"]].to_csv("clouds_zichni.csv", index=False)
    df_all[["Date", "Snow"]].to_csv("snow_zichni.csv", index=False)
    
    # Pressure not available, use placeholder
    df_all[["Date"]].assign(Pressure=None).to_csv("pressure_zichni.csv", index=False)
    
    # Save combined monthly
    monthly = df_all.groupby(["Year", "Month"]).agg({
        "Avg_Temp": "mean",
        "Max_Temp": "mean",
        "Min_Temp": "mean",
        "Rain": "sum",
        "Sun_Hours": "sum",
        "Clouds": "mean",
        "Snow": "sum"
    }).reset_index()
    
    monthly["Date"] = pd.to_datetime(monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01")
    
    # Add wind columns (estimate from daily)
    wind_daily = df_all.copy()
    wind_daily["Year"] = wind_daily["Date"].dt.year
    wind_daily["Month"] = wind_daily["Date"].dt.month
    wind_monthly = wind_daily.groupby(["Year", "Month"]).agg({
        "Max_Wind": "mean",
        "Avg_Wind": "mean",
        "Avg_Gust": "mean"
    }).reset_index()
    
    monthly = monthly.merge(wind_monthly, on=["Year", "Month"], how="left")
    
    # Add Pressure placeholder
    monthly["Pressure"] = 1013
    
    monthly = monthly[["Date", "Year", "Month", "Avg_Temp", "Max_Temp", "Min_Temp", "Rain", 
                       "Max_Wind", "Avg_Wind", "Avg_Gust", "Sun_Hours", "Clouds", "Pressure", "Snow"]]
    
    monthly.to_csv("Nea_Zichni_scrapped_data_full.csv", index=False)
    
    print("\nSaved all files:")
    print("  - temperature_zichni.csv")
    print("  - rain_zichni.csv")
    print("  - wind_zichni.csv")
    print("  - sunhours_zichni.csv")
    print("  - clouds_zichni.csv")
    print("  - snow_zichni.csv")
    print("  - pressure_zichni.csv")
    print("  - Nea_Zichni_scrapped_data_full.csv")
    
    print("\nData years:", sorted(monthly["Year"].unique()))


if __name__ == "__main__":
    main()
