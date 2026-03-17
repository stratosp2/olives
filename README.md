# Ελιές - Olive Yield Forecasting

Olive yield prediction system for Nea Zichni, Greece.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001

# Start frontend (in another terminal)
python -m http.server 8002 --directory frontend
```

Open http://localhost:8002

## API Endpoints

- `GET /api/prediction` - Get yield prediction
- `GET /api/dashboard` - Dashboard data
- `GET /api/history` - Historical harvest data

## Model

Statistical model using:
- August cloud cover
- December cloud cover  
- July temperature

R² = 0.62, p-value = 0.002
