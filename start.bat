@echo off
chcp 65001 >nul
title Ελιές - Olive Forecasting

echo ========================================
echo  🫒 Εκκίνηση Συστήματος Πρόβλεψης Ελιών
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Ξεκινώντας Backend (port 8001)...
start "Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001"

echo [2/2] Ξεκινώντας Frontend (port 8002)...
start "Frontend" cmd /k "python -m http.server 8002 --directory frontend"

echo.
echo ========================================
echo  ✅ Υπηρεσίες σε λειτουργία!
echo ========================================
echo.
echo   Backend API:  http://localhost:8001
echo   Frontend:     http://localhost:8002
echo.
echo  Κλείστε τα παράθυρα για να σταματήσετε.
echo.

start http://localhost:8002
