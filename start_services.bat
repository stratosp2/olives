@echo off
chcp 65001 >nul
echo ================================================
echo   Εκκίνηση Συστήματος Πρόβλεψης Ελιών
echo ================================================
echo.

cd /d "%~dp0"

REM Install dependencies if needed
echo Checking dependencies...
python -c "import fastapi" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install -q -r requirements.txt
)

REM Start backend
echo.
echo [1/2] Starting Backend API...
start "Olive Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo [2/2] Starting Frontend Dashboard...
start "Olive Frontend" cmd /k "python -m http.server 8002 --directory frontend"

echo.
echo ================================================
echo   Υπηρεσίες σε λειτουργία:
echo ================================================
echo   Dashboard: http://localhost:8002
echo   Backend API: http://localhost:8001
echo   API Docs: http://localhost:8001/docs
echo ================================================
echo.
echo Πατήστε οποιοδήποτε πλήκτρο για έξοδο...
pause >nul
