# Phase 5 — Deployment & Operations Runbook
**Hospital Risk Intelligence API — v1.0.0**

---

## Overview

This runbook covers how to run, test, and maintain the Hospital Risk Intelligence API. The API serves two ML models built in Phase 3:

| Endpoint | Model | Purpose |
|---|---|---|
| `POST /predict/risk` | Model A (Random Forest) | Predict visit risk: Low / Medium / High |
| `POST /predict/claim` | Model B (Random Forest + SMOTE) | Predict claim outcome: Paid / Pending / Rejected |

---

## Prerequisites

Before running the API, ensure the following files exist (created by Phases 3 & 4):

```
project_root/
├── models/
│   ├── model_a_rf.joblib       ← Visit risk model
│   └── model_b_rf.joblib       ← Claim outcome model
├── data_outputs/
│   ├── feature_schema.json     ← Model metadata & encodings
│   └── model_card.json         ← Model governance doc
└── api/
    ├── main.py                 ← FastAPI application
    ├── requirements.txt
    └── logs/                   ← Auto-created on first prediction
```

---

## Option 1 — Run Locally (Development)

### Step 1: Create and activate a virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r api/requirements.txt
```

### Step 3: Start the server

```bash
# Run from the project root (not from inside api/)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` restarts the server automatically when you edit `main.py`. Remove it for production.

### Step 4: Verify startup

Open your browser and check:

- **Swagger UI (interactive docs):** http://localhost:8000/docs
- **ReDoc (reference docs):** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

Expected `/health` response:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "models_loaded": {
    "model_a_risk": true,
    "model_b_claim": true
  }
}
```

---

## Option 2 — Run with Docker (Production)

### Step 1: Build the Docker image

```bash
# Run from the project root (where Dockerfile lives)
docker build -t hospital-risk-api:1.0.0 .
```

Build time: ~2–3 minutes (downloads Python packages). Subsequent builds are faster due to layer caching.

### Step 2: Run the container

```bash
docker run -d \
  --name hospital-risk-api \
  -p 8000:8000 \
  hospital-risk-api:1.0.0
```

Flags:
- `-d` — run in background (detached)
- `--name` — gives the container a name for easy reference
- `-p 8000:8000` — maps host port 8000 to container port 8000

### Step 3: Check container health

```bash
# View logs
docker logs hospital-risk-api

# Check running status
docker ps

# Check Docker health status (HEALTHY / UNHEALTHY)
docker inspect --format='{{.State.Health.Status}}' hospital-risk-api
```

### Step 4: Stop and remove the container

```bash
docker stop hospital-risk-api
docker rm hospital-risk-api
```

### Persisting logs outside the container

Prediction logs are written to `api/logs/predictions.jsonl` inside the container. To persist them on your host machine:

```bash
docker run -d \
  --name hospital-risk-api \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/api/logs \
  hospital-risk-api:1.0.0
```

---

## Testing the Running API

### Using curl (command line)

**Health check:**
```bash
curl http://localhost:8000/health
```

**Risk prediction:**
```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62,
    "gender": "M",
    "city": "Mumbai",
    "chronic_flag": 1,
    "department": "ICU",
    "visit_type": "ER",
    "length_of_stay_hours": 48.0,
    "patient_visit_frequency": 5,
    "avg_los_per_patient": 30.0,
    "days_since_registration": 720,
    "visit_month": 6,
    "visit_day_of_week": 0
  }'
```

**Claim prediction:**
```bash
curl -X POST http://localhost:8000/predict/claim \
  -H "Content-Type: application/json" \
  -d '{
    "billed_amount": 52000.0,
    "approved_amount": 0.0,
    "approval_ratio": 0.02,
    "length_of_stay_hours": 36.0,
    "department": "Cardiology",
    "visit_type": "ER",
    "age": 58,
    "gender": "F",
    "chronic_flag": 1,
    "insurance_provider": "HealthPlus",
    "patient_visit_frequency": 4,
    "days_since_registration": 550,
    "visit_month": 9
  }'
```

### Using Python (requests library)

```python
import requests

BASE = "http://localhost:8000"

# Risk prediction
payload = {
    "age": 62, "gender": "M", "city": "Mumbai", "chronic_flag": 1,
    "department": "ICU", "visit_type": "ER", "length_of_stay_hours": 48.0,
    "patient_visit_frequency": 5, "avg_los_per_patient": 30.0,
    "days_since_registration": 720, "visit_month": 6, "visit_day_of_week": 0
}
response = requests.post(f"{BASE}/predict/risk", json=payload)
print(response.json())
```

### Using Swagger UI (easiest for beginners)

1. Open http://localhost:8000/docs
2. Click on `POST /predict/risk`
3. Click "Try it out"
4. Edit the example JSON payload
5. Click "Execute"
6. See the response directly in the browser

---

## Viewing Prediction Logs

Logs are written in JSONL format (one JSON object per line):

```bash
# Tail the log file in real time
tail -f api/logs/predictions.jsonl

# View last 10 predictions via API
curl http://localhost:8000/logs/recent?n=10

# Count total predictions logged
wc -l api/logs/predictions.jsonl
```

Each log entry contains:
```json
{
  "timestamp": "2026-01-15T10:30:00+00:00",
  "model_version": "1.0.0",
  "endpoint": "predict/risk",
  "input_hash": "a3f92b1c4d8e5f7a",
  "prediction": "High",
  "probabilities": {"High": 0.54, "Medium": 0.31, "Low": 0.15}
}
```

---

## Allowed Field Values

Both endpoints reject requests with unknown categorical values (HTTP 422):

| Field | Allowed Values |
|---|---|
| `gender` | `F`, `M` |
| `department` | `Cardiology`, `ER`, `General`, `ICU`, `Neurology`, `Orthopedics` |
| `visit_type` | `ER`, `ICU`, `OPD` |
| `city` | `Bangalore`, `Chennai`, `Delhi`, `Hyderabad`, `Mumbai`, `Pune` |
| `insurance_provider` | `CareOne`, `HealthPlus`, `MediCareX`, `SecureLife` |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError` on startup | Model .joblib files missing | Run Phase 3 notebook first |
| `HTTP 422` on prediction | Invalid field value or type | Check allowed values table above |
| `HTTP 500` on prediction | Feature shape mismatch | Verify feature_schema.json matches main.py |
| Docker build fails | Missing libgomp1 | Already in Dockerfile; check network connectivity |
| Port 8000 already in use | Another service running | Use `--port 8001` or kill the other process |

---

## Updating the Models

When new models are trained (retraining after performance drift):

1. Re-run `Phase3_Modeling.ipynb` — produces new `model_a_rf.joblib` and `model_b_rf.joblib`
2. Re-run `Phase4_Evaluation.ipynb` — updates `model_card.json`
3. Increment `MODEL_VERSION` in `api/main.py` (e.g. `"1.0.0"` → `"1.1.0"`)
4. Rebuild Docker image: `docker build -t hospital-risk-api:1.1.0 .`
5. Deploy the new image with the updated version tag

---

## API Endpoints Summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health & model load status |
| `GET` | `/models/info` | Model metadata, features, performance |
| `POST` | `/predict/risk` | Visit risk prediction (Model A) |
| `POST` | `/predict/claim` | Claim outcome prediction (Model B) |
| `GET` | `/logs/recent` | Recent prediction log entries |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | Reference documentation |
