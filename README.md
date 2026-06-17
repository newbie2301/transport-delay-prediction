# Intelligent Transport Delay Prediction and Route Optimization System

A machine-learning system that predicts commute delays from time-of-day,
peak-hour, and live traffic-density signals, then ranks real alternative
routes (via OpenRouteService) by *predicted* total travel time rather than
distance alone. Live weather (via OpenWeather) is shown alongside each
prediction, since rain and visibility are well-known delay drivers.

The model is built around **TabPFN**, a transformer-based tabular model that
fits quickly without manual hyperparameter tuning. Because TabPFN requires a
PyTorch install and isn't always available (e.g. offline grading
environments, CI), the system automatically falls back to a
`GradientBoostingRegressor` with an identical interface, so the project runs
end-to-end either way. The same fallback pattern applies to SHAP-based
explainability — if SHAP can't run, a sensitivity-based approximation is used
instead.

## Features

- **Delay prediction** from hour, peak-hour flag, and traffic density
- **Risk classification**: Low / Medium / High, based on predicted delay
- **Route optimization**: fetches alternative routes from OpenRouteService
  and recommends the one with the lowest base-duration + predicted-delay
- **Live weather context** from OpenWeather
- **Explainability**: SHAP-based feature contribution breakdown per
  prediction (with a dependency-free fallback)
- **Interactive dashboard**: map view of route alternatives, a delay gauge,
  risk badge, and a feature-contribution chart

## Project structure

```
transport-delay-prediction/
├── backend/                 FastAPI application
│   ├── main.py               API routes, app wiring, static file serving
│   ├── model.py               TabPFN model wrapper + GradientBoosting fallback
│   ├── feature_engineering.py Raw input -> model feature vector
│   ├── risk.py                 Delay -> risk-level classification
│   ├── explainability.py       SHAP-based per-prediction explanations
│   ├── schemas.py              Pydantic request/response models
│   ├── config.py               Environment-driven settings
│   └── services/                OpenRouteService + OpenWeather integrations
├── ml/                       Dataset generation, training, evaluation scripts
├── frontend/                  Static dashboard (HTML/CSS/JS + Leaflet map)
├── tests/                     Pytest suite (API tests mock external services)
├── data/                       Place a `traffic_data.csv` here to train on
│                                real data instead of the synthetic generator
└── models/                     Trained model + evaluation plots are written here
```

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`tabpfn` and `shap` are optional — if either fails to install or import, the
app automatically uses its fallback path. The rest of the requirements are
required.

### 2. Configure API keys

```bash
cp .env.example .env
```

Then fill in:

- `ORS_API_KEY` — free key from [openrouteservice.org](https://openrouteservice.org/dev/#/signup)
- `OPENWEATHER_API_KEY` — free key from [openweathermap.org](https://home.openweathermap.org/api_keys)

### 3. (Optional) Train the model explicitly

The API will train a model automatically on first run if none exists yet,
but you can also train (and compare candidate models) directly:

```bash
python -m ml.train_model
python -m ml.evaluate     # writes plots to models/plots/
```

### 4. Run the API + dashboard

```bash
uvicorn backend.main:app --reload
```

Then open <http://localhost:8000> for the dashboard, or
<http://localhost:8000/docs> for the interactive API documentation.

### 5. Run tests

```bash
pytest -v
```

API tests mock the external services, so the suite runs offline without
needing real API keys.

## API reference

### `GET /api/health`

```json
{ "status": "ok", "model_loaded": true, "model_type": "GradientBoostingRegressor" }
```

### `POST /api/predict`

Request:

```json
{
  "hour": 9,
  "minute": 15,
  "traffic_density": 72,
  "origin": "Koramangala, Bengaluru",
  "destination": "Whitefield, Bengaluru",
  "city": "Bengaluru"
}
```

Response (truncated):

```json
{
  "predicted_delay_minutes": 8.4,
  "risk_level": "Medium",
  "expected_arrival_time": "09:52",
  "weather": { "city": "Bengaluru", "description": "light rain", "temperature_celsius": 24.1, "humidity_percent": 81, "wind_speed_ms": 3.4 },
  "routes": [ { "route_index": 0, "distance_km": 18.4, "base_duration_minutes": 35.0, "predicted_delay_minutes": 8.4, "total_time_minutes": 43.4, "is_best": true, "geometry": [[12.93, 77.61], ...] } ],
  "best_route_index": 0,
  "model_mae_minutes": 1.8,
  "model_rmse_minutes": 2.3,
  "explanation": [ { "feature": "traffic_density", "impact": 4.1, "percent": 62.3 } ]
}
```

## Using real historical data

By default the model trains on a synthetic dataset that encodes realistic
peak-hour and traffic-density relationships (see `ml/generate_dataset.py`).
To train on real data instead, place a CSV with columns
`hour, peak_hour, traffic_density, delay` at `data/traffic_data.csv` — both
`ml/train_model.py` and the API's auto-train path will pick it up
automatically.

## License

MIT — see [LICENSE](LICENSE).
