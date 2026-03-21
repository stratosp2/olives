#!/usr/bin/env python3
"""
Olive Yield Forecasting - Complete Startup Script
Starts: Backend API + Frontend (simple HTML dashboard)
"""

import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

processes = []

def print_banner():
    print("=" * 60)
    print("  OLIVE YIELD FORECASTING SYSTEM")
    print("  Weather Data + ML Predictions")
    print("=" * 60)
    print()

def start_backend():
    """Start the FastAPI backend server"""
    print("[1/2] Starting Backend API on http://localhost:8001 ...")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"],
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("backend", proc))
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if proc.poll() is not None:
            print("  ERROR: Backend failed to start!")
            return False
            
        print("  ✓ Backend running at http://localhost:8001")
        print("  ✓ API docs at http://localhost:8001/docs")
        return True
    except Exception as e:
        print(f"  ERROR starting backend: {e}")
        return False

def start_frontend():
    """Start simple frontend server"""
    print("[2/2] Starting Frontend Dashboard on http://localhost:8002 ...")
    
    # Create a simple HTML frontend if it doesn't exist
    frontend_html = os.path.join(SCRIPT_DIR, "frontend", "index.html")
    os.makedirs(os.path.join(SCRIPT_DIR, "frontend"), exist_ok=True)
    
    # Generate the frontend HTML
    html_content = """<!DOCTYPE html>
<html lang="el">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ελιές - Πρόβλεψη Παραγωγής</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            color: white;
            padding: 40px 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 0 0 30px 30px;
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { opacity: 0.9; font-size: 1.1em; }
        .cards { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
        .card h2 {
            color: #1a472a;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid #4a7c59;
            padding-bottom: 10px;
        }
        .prediction {
            text-align: center;
            padding: 20px;
        }
        .big-number {
            font-size: 3em;
            font-weight: bold;
            color: #2d5a3d;
        }
        .unit { font-size: 1em; color: #666; }
        .label { color: #888; margin-top: 5px; }
        .historical-table {
            width: 100%;
            border-collapse: collapse;
        }
        .historical-table th, .historical-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .historical-table th { background: #f5f5f5; color: #1a472a; }
        .weather-summary {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .weather-item {
            background: #f9f9f9;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .weather-value { font-size: 1.5em; font-weight: bold; color: #2d5a3d; }
        .weather-label { font-size: 0.85em; color: #666; }
        .loading { 
            text-align: center; 
            padding: 50px; 
            color: #666;
        }
        .error { color: #c0392b; padding: 20px; background: #ffebee; border-radius: 8px; }
        .refresh-btn {
            background: #4a7c59;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 20px;
        }
        .refresh-btn:hover { background: #3d6349; }
        footer {
            text-align: center;
            color: rgba(255,255,255,0.7);
            padding: 20px;
            margin-top: 40px;
        }
        .api-link {
            color: #4a7c59;
            text-decoration: none;
        }
        .api-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <header>
        <h1>🌿 Ελιές - Πρόβλεψη Παραγωγής</h1>
        <p class="subtitle">Olive Yield Forecasting System</p>
    </header>
    
    <div class="container">
        <div class="cards">
            <div class="card">
                <h2>📊 Πρόβλεψη Παραγωγής</h2>
                <div id="prediction" class="prediction">
                    <div class="loading">Φόρτωση...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>🌤️ Τελευταία Καιρικά Δεδομένα</h2>
                <div id="weather" class="weather-summary">
                    <div class="loading">Φόρτωση...</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 Ιστορικά Δεδομένα</h2>
            <div id="history">
                <div class="loading">Φόρτωση...</div>
            </div>
        </div>
        
        <div style="text-align: center;">
            <button class="refresh-btn" onclick="loadData()">🔄 Ενημέρωση</button>
        </div>
    </div>
    
    <footer>
        <p>API: <a href="http://localhost:8001" class="api-link">http://localhost:8001</a> | 
           Docs: <a href="http://localhost:8001/docs" class="api-link">/docs</a></p>
    </footer>

    <script>
        const API_BASE = 'http://localhost:8001';
        
        async function loadData() {
            await Promise.all([
                loadPrediction(),
                loadDashboard()
            ]);
        }
        
        async function loadPrediction() {
            try {
                const resp = await fetch(API_BASE + '/api/prediction');
                const data = await resp.json();
                
                if (data.error) {
                    document.getElementById('prediction').innerHTML = 
                        '<div class="error">Σφάλμα: ' + data.error + '</div>';
                    return;
                }
                
                document.getElementById('prediction').innerHTML = `
                    <div class="big-number">${data.ensemble_olives_kg || 'N/A'}</div>
                    <div class="unit">κιλά ελιές</div>
                    <div class="label">για ${data.year}</div>
                    <hr style="margin: 15px 0; border: none; border-top: 1px solid #eee;">
                    <div class="weather-item">
                        <div class="weather-value">${data.estimated_oil_kg || 'N/A'}</div>
                        <div class="weather-label">κιλά λάδι (εκτίμηση)</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        ${data.olives_per_tree_kg} κιλά/δέντρο | ${data.trees} δέντρα
                    </div>
                `;
            } catch (e) {
                document.getElementById('prediction').innerHTML = 
                    '<div class="error">Δεν ήταν δυνατή η σύνδεση με το API</div>';
            }
        }
        
        async function loadDashboard() {
            try {
                const resp = await fetch(API_BASE + '/api/dashboard');
                const data = await resp.json();
                
                // Weather
                if (data.latest_weather) {
                    const w = data.latest_weather;
                    document.getElementById('weather').innerHTML = `
                        <div class="weather-item">
                            <div class="weather-value">${w.total_rain_mm}</div>
                            <div class="weather-label">mm βροχή</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">${w.avg_temp_c}°C</div>
                            <div class="weather-label">μέση θερμοκρασία</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">${w.avg_clouds_pct}%</div>
                            <div class="weather-label">συννεφιά</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">${w.year}</div>
                            <div class="weather-label">έτος</div>
                        </div>
                    `;
                }
                
                // History
                const resp2 = await fetch(API_BASE + '/api/history');
                const historyData = await resp2.json();
                
                if (historyData.data && historyData.data.length > 0) {
                    const rows = historyData.data.slice(-10).reverse().map(d => 
                        `<tr>
                            <td>${d.year}</td>
                            <td>${d.trees}</td>
                            <td>${d.olives || '-'}</td>
                            <td>${d.oil || '-'}</td>
                            <td>${d.ratio ? (d.ratio * 100).toFixed(1) + '%' : '-'}</td>
                        </tr>`
                    ).join('');
                    
                    document.getElementById('history').innerHTML = `
                        <table class="historical-table">
                            <thead>
                                <tr>
                                    <th>Έτος</th>
                                    <th>Δέντρα</th>
                                    <th>Ελιές (kg)</th>
                                    <th>Λάδι (kg)</th>
                                    <th>Απόδοση</th>
                                </tr>
                            </thead>
                            <tbody>${rows}</tbody>
                        </table>
                    `;
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        // Load on start
        loadData();
        
        // Auto-refresh every 5 minutes
        setInterval(loadData, 5 * 60 * 1000);
    </script>
</body>
</html>
"""
    
    with open(frontend_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Start simple HTTP server for frontend
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8002", "--directory", os.path.join(SCRIPT_DIR, "frontend")],
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append(("frontend", proc))
        print("  ✓ Frontend running at http://localhost:8002")
        return True
    except Exception as e:
        print(f"  ERROR starting frontend: {e}")
        return False

def cleanup(signum=None, frame=None):
    """Kill all started processes"""
    print("\n\nShutting down services...")
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            try:
                proc.kill()
            except:
                pass
    print("All services stopped.")
    sys.exit(0)

def main():
    print_banner()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Install dependencies if needed
    print("Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import pandas
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    
    # Start services
    if not start_backend():
        print("\nFailed to start backend. Exiting.")
        sys.exit(1)
    
    time.sleep(1)
    
    if not start_frontend():
        print("\nWarning: Frontend failed to start, but backend is running.")
    
    print("\n" + "=" * 60)
    print("ALL SERVICES RUNNING!")
    print("=" * 60)
    print("  Dashboard: http://localhost:8002")
    print("  Backend API: http://localhost:8001")
    print("  API Docs: http://localhost:8001/docs")
    print("=" * 60)
    print("\nPress Ctrl+C to stop all services.\n")
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8002")
    except:
        pass
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
