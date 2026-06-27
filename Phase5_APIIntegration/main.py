import sys
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UTILS_DIR = BASE_DIR / "Utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

# Import utilities after adding path
from phase1_utils import ENCODING_MAPS

# ============================================================
# MODEL PATHS
# ============================================================

MODEL_A_PATH = BASE_DIR / "models" / "model_a_rf.joblib"
MODEL_B_PATH = BASE_DIR / "models" / "model_b_rf.joblib"

LOG_PATH = BASE_DIR / "Phase5_APIIntegration" / "logs"
LOG_PATH.mkdir(exist_ok=True)

MODEL_VERSION = "1.0.0"

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

MODEL_A = joblib.load(MODEL_A_PATH)
MODEL_B = joblib.load(MODEL_B_PATH)

print("Models loaded successfully")


print("\nModel A expects:")
print(MODEL_A.feature_names_in_)

print("\nModel B expects:")
print(MODEL_B.feature_names_in_)

# ============================================================
# LOGGER
# ============================================================

logging.basicConfig(
    filename=LOG_PATH / "predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hospital Risk Intelligence API",
    version=MODEL_VERSION
)

# ============================================================
# HELPERS
# ============================================================

def encode_category(value, field):

    mapping = ENCODING_MAPS[field]

    if value not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"{value} not allowed for {field}"
        )

    return mapping[value]


def log_prediction(endpoint, request_data, prediction):

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "input": request_data,
        "prediction": str(prediction)
    }

    logging.info(json.dumps(record))


# ============================================================
# REQUEST MODELS
# ============================================================

class RiskRequest(BaseModel):

    age: int = Field(
        ...,
        ge=0,
        le=120,
        example=45,
        description="Patient age in years"
    )

    gender: str = Field(
        ...,
        example="F",
        description="Allowed values: F, M"
    )

    city: str = Field(
        ...,
        example="Bangalore",
        description="Allowed values: Bangalore, Chennai, Delhi, Hyderabad, Mumbai, Pune"
    )

    chronic_flag: int = Field(
        ...,
        ge=0,
        le=1,
        example=1,
        description="0 = No chronic condition, 1 = Chronic condition present"
    )

    department: str = Field(
        ...,
        example="Cardiology",
        description="Allowed values: Cardiology, ER, General, ICU, Neurology, Orthopedics"
    )

    visit_type: str = Field(
        ...,
        example="OPD",
        description="Allowed values: ER, ICU, OPD"
    )

    length_of_stay_hours: float = Field(
        ...,
        ge=0,
        example=12.0,
        description="Length of stay in hours"
    )

    patient_visit_frequency: int = Field(
        ...,
        ge=1,
        example=3,
        description="Number of historical patient visits"
    )

    avg_los_per_patient: float = Field(
        ...,
        ge=0,
        example=8.5,
        description="Average historical LOS for patient"
    )

    days_since_registration: int = Field(
        ...,
        ge=0,
        example=365,
        description="Days since patient registration"
    )

    visit_month: int = Field(
        ...,
        ge=1,
        le=12,
        example=6,
        description="Month of visit (1-12)"
    )

    visit_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        example=2,
        description="0 = Monday, 6 = Sunday"
    )


class ClaimRequest(BaseModel):

    billed_amount: float = Field(
        ...,
        ge=0,
        example=10000.0,
        description="Total billed amount"
    )

    approved_amount: float = Field(
        ...,
        ge=0,
        example=6000.0,
        description="Approved amount by insurer"
    )

    approval_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        example=0.60,
        description="approved_amount / billed_amount"
    )

    length_of_stay_hours: float = Field(
        ...,
        ge=0,
        example=24.0,
        description="Length of stay in hours"
    )

    department: str = Field(
        ...,
        example="Cardiology",
        description="Allowed values: Cardiology, ER, General, ICU, Neurology, Orthopedics"
    )

    visit_type: str = Field(
        ...,
        example="OPD",
        description="Allowed values: ER, ICU, OPD"
    )

    age: int = Field(
        ...,
        ge=0,
        le=120,
        example=45,
        description="Patient age"
    )

    gender: str = Field(
        ...,
        example="F",
        description="Allowed values: F, M"
    )

    chronic_flag: int = Field(
        ...,
        ge=0,
        le=1,
        example=1,
        description="0 = No chronic condition, 1 = Chronic condition present"
    )

    insurance_provider: str = Field(
        ...,
        example="MediCareX",
        description="Allowed values: CareOne, HealthPlus, MediCareX, SecureLife"
    )

    patient_visit_frequency: int = Field(
        ...,
        ge=1,
        example=4,
        description="Total historical patient visits"
    )

    days_since_registration: int = Field(
        ...,
        ge=0,
        example=550,
        description="Days since patient registration"
    )

    visit_month: int = Field(
        ...,
        ge=1,
        le=12,
        example=9,
        description="Month of visit (1-12)"
    )
# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }


# ============================================================
# RISK PREDICTION
# ============================================================

@app.post("/predict/risk")
def predict_risk(req: RiskRequest):

    try:

        feature_row = {

            "age": req.age,

            "chronic_flag": req.chronic_flag,

            "length_of_stay_hours":
                req.length_of_stay_hours,

            "patient_visit_frequency":
                req.patient_visit_frequency,

            "avg_los_per_patient":
                req.avg_los_per_patient,

            "days_since_registration":
                req.days_since_registration,

            "visit_month":
                req.visit_month,

            "visit_day_of_week":
                req.visit_day_of_week,

            "department_enc":
                encode_category(
                    req.department,
                    "department"
                ),

            "visit_type_enc":
                encode_category(
                    req.visit_type,
                    "visit_type"
                ),

            "gender_enc":
                encode_category(
                    req.gender,
                    "gender"
                ),

            "city_enc":
                encode_category(
                    req.city,
                    "city"
                )
        }

        X = pd.DataFrame([feature_row])

        # Exact order expected by model
        X = X[MODEL_A.feature_names_in_]

        prediction = MODEL_A.predict(X)[0]

        probabilities = MODEL_A.predict_proba(X)[0]

        response = {
            "prediction": str(prediction),

            "probabilities": {
                str(cls): float(prob)
                for cls, prob in zip(
                    MODEL_A.classes_,
                    probabilities
                )
            },

            "model_version": MODEL_VERSION
        }

        log_prediction(
            "/predict/risk",
            req.model_dump(),
            prediction
        )

        return response

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/models/info")
def model_info():

    return {

        "model_version": MODEL_VERSION,

        "model_a_features":
            MODEL_A.feature_names_in_.tolist(),

        "model_b_features":
            MODEL_B.feature_names_in_.tolist(),

        "model_a_classes":
            [str(c) for c in MODEL_A.classes_],

        "model_b_classes":
            [str(c) for c in MODEL_B.classes_]
    }

@app.post("/predict/claim")
def predict_claim(req: ClaimRequest):

    try:

        feature_row = {

            "billed_amount":
                req.billed_amount,

            "approved_amount":
                req.approved_amount,

            "approval_ratio":
                req.approval_ratio,

            "length_of_stay_hours":
                req.length_of_stay_hours,

            "age":
                req.age,

            "chronic_flag":
                req.chronic_flag,

            "patient_visit_frequency":
                req.patient_visit_frequency,

            "days_since_registration":
                req.days_since_registration,

            "visit_month":
                req.visit_month,

            "department_enc":
                encode_category(
                    req.department,
                    "department"
                ),

            "visit_type_enc":
                encode_category(
                    req.visit_type,
                    "visit_type"
                ),

            "insurance_provider_enc":
                encode_category(
                    req.insurance_provider,
                    "insurance_provider"
                ),

            "gender_enc":
                encode_category(
                    req.gender,
                    "gender"
                )
        }

        X = pd.DataFrame([feature_row])

        X = X[MODEL_B.feature_names_in_]

        prediction = MODEL_B.predict(X)[0]

        probabilities = MODEL_B.predict_proba(X)[0]

        response = {

            "prediction":
                str(prediction),

            "probabilities": {

                str(cls): float(prob)

                for cls, prob in zip(
                    MODEL_B.classes_,
                    probabilities
                )
            },

            "model_version":
                MODEL_VERSION
        }

        log_prediction(
            "/predict/claim",
            req.model_dump(),
            prediction
        )

        return response

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================================
# RECENT PREDICTION LOGS
# ============================================================

@app.get("/logs/recent", tags=["Logs"])
def get_recent_logs(n: int = 10):

    logfile = LOG_PATH / "predictions.log"

    if not logfile.exists():

        return {
            "message": "No prediction logs found"
        }

    with open(logfile, "r") as f:

        logs = f.readlines()

    n = min(n, 100)

    return {

        "total_logged": len(logs),

        "returned": min(n, len(logs)),

        "logs": logs[-n:]
    }


# ============================================================
# START MESSAGE
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["Home"])
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hospital Risk Intelligence API</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f9;
                margin: 14;
                padding: 40px;
            }

            .container {
                max-width: 1000px;
                margin: auto;
            }

            .header {
                text-align: center;
                background: linear-gradient(135deg,#1e3c72,#2a5298);
                color: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            }

            .header h1 {
                margin: 0;
                font-size: 42px;
            }

            .header p {
                font-size: 18px;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit,minmax(280px,1fr));
                gap: 20px;
                margin-top: 30px;
            }

            .card {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                transition: 0.3s;
            }

            .card:hover {
                transform: translateY(-5px);
            }

            .card h3 {
                color: #2a5298;
            }

            .btn {
                display: inline-block;
                margin-top: 15px;
                padding: 10px 20px;
                background: #2a5298;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }

            .btn:hover {
                background: #1e3c72;
            }

            footer {
                text-align: center;
                margin-top: 40px;
                color: gray;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <div class="header">
                <h1>🏥 Hospital Risk Intelligence API</h1>
                <p>Machine Learning Platform for Clinical Risk and Claim Prediction</p>
            </div>

            <div class="grid">

                <div class="card">
                    <h3>📘 Swagger Documentation</h3>
                    <p>Interactive API testing interface.</p>
                    <a class="btn" href="/docs">Open Docs</a>
                </div>

                <div class="card">
                    <h3>❤️ Health Check</h3>
                    <p>Verify service and model status.</p>
                    <a class="btn" href="/health">Check Health</a>
                </div>

                <div class="card">
                    <h3>🧠 Model Information</h3>
                    <p>View metadata and model details.</p>
                    <a class="btn" href="/models/info">View Models</a>
                </div>

                <div class="card">
                    <h3>📋 Prediction Logs</h3>
                    <p>Inspect recent prediction history.</p>
                    <a class="btn" href="/logs/recent">View Logs</a></li>
                </div>

            </div>

            <footer>
                <p>Hospital Risk Intelligence API v1.0.0</p>
            </footer>

        </div>

    </body>
    </html>
    """
