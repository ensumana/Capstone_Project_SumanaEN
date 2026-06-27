"""
phase5_utils.py
===============
Hospital Operations & Revenue Risk Intelligence Platform
Phase 5 Utility Module — Deployment & API Integration

Imported by: Phase 5 (deployment notebook), Phase 6 (monitoring)

Contents
--------
1.  Pydantic schemas     — request/response validation for both models
2.  validate_input()     — check input against feature schema
3.  build_input_df()     — turn API request dict into model-ready DataFrame
4.  predict_visit_risk() — end-to-end prediction pipeline for Model A
5.  predict_claim_outcome() — end-to-end prediction pipeline for Model B
6.  PredictionLogger     — log predictions to CSV with timestamp + input hash
7.  hash_input()         — SHA256 hash of input features for audit trail
8.  load_models()        — load both models + schema in one call
"""

import os
import csv
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

import numpy  as np
import pandas as pd

# ── Pydantic for schema validation (FastAPI dependency) ───────────────────
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  pydantic not installed. Run: pip install pydantic")

# ── Import shared constants ────────────────────────────────────────────────
from phase1_utils import (
    engineer_features,
    encode_categoricals,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    MODEL_A_TARGET,
    MODEL_B_TARGET,
    ENCODING_MAPS,
    DECODING_MAPS,
    PROVIDER_REJECTION_RATES,
)
from phase3_utils import (
    load_model,
    load_feature_schema,
    get_model_version,
    MODEL_A_PATH,
    MODEL_B_PATH,
    SCHEMA_PATH,
    MODEL_A_CLASSES,
    MODEL_B_CLASSES,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PYDANTIC SCHEMAS
# These define the expected input and output format for the API endpoints.
# FastAPI uses these to auto-validate requests and generate docs.
# ══════════════════════════════════════════════════════════════════════════════

if PYDANTIC_AVAILABLE:
    class VisitRiskRequest(BaseModel):
        """Input schema for POST /predict/visit-risk"""
        visit_id            : int
        patient_id          : int
        visit_date          : str   # YYYY-MM-DD
        age                 : int   = Field(..., ge=0, le=120)
        chronic_flag        : int   = Field(..., ge=0, le=1)
        department          : str
        visit_type          : str
        gender              : str
        city                : str
        insurance_provider  : str
        length_of_stay_hours: float = Field(..., ge=0)
        billed_amount       : float = Field(..., ge=0)
        approved_amount     : Optional[float] = None
        payment_days        : Optional[float] = None
        registration_date   : str   # YYYY-MM-DD

        class Config:
            schema_extra = {
                "example": {
                    "visit_id"            : 1001,
                    "patient_id"          : 42,
                    "visit_date"          : "2025-12-01",
                    "age"                 : 65,
                    "chronic_flag"        : 1,
                    "department"          : "Cardiology",
                    "visit_type"          : "ER",
                    "gender"              : "M",
                    "city"                : "Mumbai",
                    "insurance_provider"  : "HealthPlus",
                    "length_of_stay_hours": 24.5,
                    "billed_amount"       : 35000.0,
                    "approved_amount"     : 35000.0,
                    "payment_days"        : 15.0,
                    "registration_date"   : "2025-06-01",
                }
            }

    class VisitRiskResponse(BaseModel):
        """Output schema for POST /predict/visit-risk"""
        visit_id        : int
        predicted_class : str           # 'Low', 'Medium', or 'High'
        confidence      : float         # probability of predicted class
        probabilities   : Dict[str, float]  # {'Low': 0.1, 'Medium': 0.3, 'High': 0.6}
        model_version   : str
        timestamp       : str
        input_hash      : str

    class ClaimOutcomeRequest(BaseModel):
        """Input schema for POST /predict/claim-outcome"""
        visit_id            : int
        patient_id          : int
        visit_date          : str
        age                 : int   = Field(..., ge=0, le=120)
        chronic_flag        : int   = Field(..., ge=0, le=1)
        department          : str
        visit_type          : str
        gender              : str
        city                : str
        insurance_provider  : str
        length_of_stay_hours: float = Field(..., ge=0)
        billed_amount       : float = Field(..., ge=0)
        approved_amount     : Optional[float] = None
        payment_days        : Optional[float] = None
        registration_date   : str

        class Config:
            schema_extra = {
                "example": {
                    "visit_id"            : 2001,
                    "patient_id"          : 88,
                    "visit_date"          : "2025-12-05",
                    "age"                 : 45,
                    "chronic_flag"        : 0,
                    "department"          : "Orthopedics",
                    "visit_type"          : "OPD",
                    "gender"              : "F",
                    "city"                : "Delhi",
                    "insurance_provider"  : "SecureLife",
                    "length_of_stay_hours": 8.0,
                    "billed_amount"       : 22000.0,
                    "approved_amount"     : None,
                    "payment_days"        : None,
                    "registration_date"   : "2025-03-15",
                }
            }

    class ClaimOutcomeResponse(BaseModel):
        """Output schema for POST /predict/claim-outcome"""
        visit_id        : int
        predicted_class : str           # 'Paid', 'Pending', or 'Rejected'
        confidence      : float
        probabilities   : Dict[str, float]
        model_version   : str
        timestamp       : str
        input_hash      : str

    class HealthResponse(BaseModel):
        """Output schema for GET /health"""
        status        : str   # 'healthy'
        model_version : str
        models_loaded : Dict[str, bool]
        timestamp     : str


# ══════════════════════════════════════════════════════════════════════════════
# 2. INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_input(
    input_dict: dict,
    schema: dict,
    verbose: bool = True,
) -> dict:
    """
    Validate an incoming API request dict against the saved feature schema.
    Checks for missing fields, out-of-range numerics, and unseen categories.

    Parameters
    ----------
    input_dict : dict  Raw input from API request
    schema     : dict  Output of load_feature_schema()
    verbose    : bool  Print validation results

    Returns
    -------
    dict  {'valid': bool, 'warnings': list, 'errors': list}

    Usage
    -----
    from phase5_utils import validate_input, load_models
    _, _, schema = load_models()
    result = validate_input(request_dict, schema)
    """
    warnings_list = []
    errors_list   = []

    for feat, stats in schema['features'].items():
        if feat not in input_dict:
            warnings_list.append(f"Feature '{feat}' missing from input — will be imputed.")
            continue

        val = input_dict[feat]
        if val is None:
            continue

        # Range check for numerics
        try:
            val_float = float(val)
            if val_float < stats['min'] * 0.5 or val_float > stats['max'] * 1.5:
                warnings_list.append(
                    f"'{feat}' value {val_float:.2f} is outside expected range "
                    f"[{stats['min']:.2f}, {stats['max']:.2f}]"
                )
        except (TypeError, ValueError):
            pass

    # Category checks
    cat_fields = ['department', 'visit_type', 'gender', 'city', 'insurance_provider']
    for field in cat_fields:
        if field in input_dict and input_dict[field] not in ENCODING_MAPS.get(field, {}):
            errors_list.append(
                f"Unseen category '{input_dict[field]}' in field '{field}'. "
                f"Valid values: {list(ENCODING_MAPS[field].keys())}"
            )

    result = {
        'valid'   : len(errors_list) == 0,
        'warnings': warnings_list,
        'errors'  : errors_list,
    }

    if verbose:
        if result['valid']:
            print(f"  ✅ Input valid. Warnings: {len(warnings_list)}")
        else:
            print(f"  ❌ Input invalid. Errors: {len(errors_list)}")
        for w in warnings_list: print(f"     ⚠️  {w}")
        for e in errors_list:   print(f"     ❌ {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD INPUT DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
def build_input_df(input_dict: dict) -> pd.DataFrame:
    """
    Convert a raw API request dictionary into a model-ready single-row DataFrame.
    Applies feature engineering and encoding identical to Phase 2/3 training.

    Parameters
    ----------
    input_dict : dict  Raw API request fields

    Returns
    -------
    pd.DataFrame  Single-row model-ready input (same columns as training)

    Usage
    -----
    from phase5_utils import build_input_df
    row = build_input_df(request.dict())
    """
    row = pd.DataFrame([input_dict])

    # Parse dates
    for date_col in ['visit_date', 'registration_date', 'billing_date']:
        if date_col in row.columns:
            row[date_col] = pd.to_datetime(row[date_col])

    # Rename billing fields if they come from the visit record
    if 'length_of_stay_hours' not in row.columns and 'los' in row.columns:
        row['length_of_stay_hours'] = row['los']

    # Handle missing fields that engineer_features() needs
    if 'claim_status' not in row.columns:
        row['claim_status'] = 'Pending'
    if 'approved_amount' not in row.columns or row['approved_amount'].isnull().all():
        row['approved_amount'] = None

    # Apply same feature engineering as Phase 2 training
    # visit_frequency & avg_los not available at inference time for a single row
    # → use neutral defaults
    row['visit_frequency']      = 5      # dataset mean
    row['avg_los_per_patient']  = 19.55  # dataset mean

    row['provider_rejection_rate'] = (
        row['insurance_provider']
        .map(PROVIDER_REJECTION_RATES)
        .fillna(0.152)
    )

    row['days_since_registration'] = (
        (row['visit_date'] - row['registration_date']).dt.days.clip(lower=0)
    )

    billed = row['billed_amount'].iloc[0]
    approved = row['approved_amount'].iloc[0]
    row['approval_ratio'] = (approved / billed) if (approved is not None and billed > 0) else 0.0

    # Capping
    row['billed_amount_capped'] = min(billed, 53621.49)
    pd_val = row.get('payment_days', pd.Series([None])).iloc[0]
    row['payment_days_capped']  = min(float(pd_val), 30.5) if pd_val is not None else 0.0
    los_val = row['length_of_stay_hours'].iloc[0]
    row['los_capped']           = min(float(los_val), 53.34)

    # Time features
    row['visit_month']     = row['visit_date'].dt.month
    row['visit_dayofweek'] = row['visit_date'].dt.dayofweek
    row['visit_quarter']   = row['visit_date'].dt.quarter

    # Encode categoricals
    row = encode_categoricals(row)

    return row


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICT VISIT RISK (Model A)
# ══════════════════════════════════════════════════════════════════════════════
def predict_visit_risk(
    input_dict: dict,
    model_a,
    model_version: str = None,
) -> dict:
    """
    End-to-end prediction pipeline for Model A — Visit Risk.

    Parameters
    ----------
    input_dict    : dict  Raw request fields
    model_a       : fitted sklearn estimator
    model_version : str   Model version string

    Returns
    -------
    dict  Prediction response (matches VisitRiskResponse schema)

    Usage
    -----
    from phase5_utils import predict_visit_risk
    response = predict_visit_risk(request.dict(), model_a)
    """
    row       = build_input_df(input_dict)
    X         = row[MODEL_A_FEATURES]
    proba     = model_a.predict_proba(X)[0]
    pred_idx  = int(np.argmax(proba))
    pred_cls  = MODEL_A_CLASSES[pred_idx]
    confidence= float(proba[pred_idx])

    return {
        'visit_id'       : input_dict.get('visit_id'),
        'predicted_class': pred_cls,
        'confidence'     : round(confidence, 4),
        'probabilities'  : {cls: round(float(p), 4) for cls, p in zip(MODEL_A_CLASSES, proba)},
        'model_version'  : model_version or get_model_version(),
        'timestamp'      : datetime.now().isoformat(),
        'input_hash'     : hash_input(input_dict),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICT CLAIM OUTCOME (Model B)
# ══════════════════════════════════════════════════════════════════════════════
def predict_claim_outcome(
    input_dict: dict,
    model_b,
    model_version: str = None,
) -> dict:
    """
    End-to-end prediction pipeline for Model B — Claim Outcome.

    Parameters
    ----------
    input_dict    : dict  Raw request fields
    model_b       : fitted sklearn estimator
    model_version : str   Model version string

    Returns
    -------
    dict  Prediction response (matches ClaimOutcomeResponse schema)

    Usage
    -----
    from phase5_utils import predict_claim_outcome
    response = predict_claim_outcome(request.dict(), model_b)
    """
    row       = build_input_df(input_dict)
    X         = row[MODEL_B_FEATURES]
    proba     = model_b.predict_proba(X)[0]
    pred_idx  = int(np.argmax(proba))
    pred_cls  = MODEL_B_CLASSES[pred_idx]
    confidence= float(proba[pred_idx])

    return {
        'visit_id'       : input_dict.get('visit_id'),
        'predicted_class': pred_cls,
        'confidence'     : round(confidence, 4),
        'probabilities'  : {cls: round(float(p), 4) for cls, p in zip(MODEL_B_CLASSES, proba)},
        'model_version'  : model_version or get_model_version(),
        'timestamp'      : datetime.now().isoformat(),
        'input_hash'     : hash_input(input_dict),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICTION LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class PredictionLogger:
    """
    Append-only CSV logger for all model predictions.
    Each row records: timestamp, model, visit_id, prediction, confidence, input_hash.
    Used for audit trails in Phase 6.

    Usage
    -----
    from phase5_utils import PredictionLogger
    logger = PredictionLogger('Output_Phase5/prediction_log.csv')
    logger.log('model_a', response)
    df = logger.load()
    """

    COLUMNS = [
        'timestamp', 'model', 'visit_id',
        'predicted_class', 'confidence', 'model_version', 'input_hash',
    ]

    def __init__(self, path: str = 'Output_Phase5/prediction_log.csv'):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Write header if file doesn't exist
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, model_name: str, response: dict):
        """Append one prediction to the log file."""
        row = {
            'timestamp'      : response.get('timestamp', datetime.now().isoformat()),
            'model'          : model_name,
            'visit_id'       : response.get('visit_id'),
            'predicted_class': response.get('predicted_class'),
            'confidence'     : response.get('confidence'),
            'model_version'  : response.get('model_version'),
            'input_hash'     : response.get('input_hash'),
        }
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

    def load(self) -> pd.DataFrame:
        """Load the full prediction log as a DataFrame."""
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.path)

    def summary(self) -> dict:
        """Quick summary of logged predictions."""
        df = self.load()
        if df.empty:
            return {'total': 0}
        return {
            'total'         : len(df),
            'by_model'      : df['model'].value_counts().to_dict(),
            'by_prediction' : df['predicted_class'].value_counts().to_dict(),
            'date_range'    : f"{df['timestamp'].min()} → {df['timestamp'].max()}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. INPUT HASH
# ══════════════════════════════════════════════════════════════════════════════
def hash_input(input_dict: dict) -> str:
    """
    Compute a SHA256 hash of the input features for audit trail.
    The same input always produces the same hash — useful for deduplication
    and tracing which input produced which prediction.

    Parameters
    ----------
    input_dict : dict  Raw API request fields

    Returns
    -------
    str  First 16 characters of SHA256 hex digest

    Usage
    -----
    from phase5_utils import hash_input
    h = hash_input({'age': 45, 'department': 'ICU', ...})
    """
    canonical = json.dumps(input_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# 8. LOAD BOTH MODELS IN ONE CALL
# ══════════════════════════════════════════════════════════════════════════════
def load_models(
    model_a_path: str = MODEL_A_PATH,
    model_b_path: str = MODEL_B_PATH,
    schema_path:  str = SCHEMA_PATH,
    verbose:      bool = True,
):
    """
    Load both trained models and the feature schema in one call.
    Used at the top of Phase 5 notebook and in the FastAPI startup event.

    Parameters
    ----------
    model_a_path : str  Path to model_a_risk.pkl
    model_b_path : str  Path to model_b_claim.pkl
    schema_path  : str  Path to feature_schema.json
    verbose      : bool Print confirmation

    Returns
    -------
    model_a : fitted sklearn estimator
    model_b : fitted sklearn estimator
    schema  : dict  Feature schema

    Usage
    -----
    from phase5_utils import load_models
    model_a, model_b, schema = load_models()
    """
    model_a = load_model(model_a_path, verbose=verbose)
    model_b = load_model(model_b_path, verbose=verbose)
    schema  = load_feature_schema(schema_path, verbose=verbose)
    return model_a, model_b, schema


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from phase1_utils import notebook_setup, build_model_table, MODEL_A_FEATURES, MODEL_A_TARGET
    from phase3_utils  import time_split, train_model, save_model, save_feature_schema

    print("Running phase5_utils self-test...\n")

    ctx = notebook_setup(verbose=False)
    df  = build_model_table(ctx['df'])
    train, test = time_split(df, verbose=False)

    model_a, _, _ = train_model(train, MODEL_A_FEATURES, MODEL_A_TARGET, verbose=False)
    model_b, _, _ = train_model(train, MODEL_B_FEATURES, MODEL_B_TARGET, verbose=False)
    save_model(model_a, MODEL_A_PATH)
    save_model(model_b, MODEL_B_PATH)
    save_feature_schema(train, MODEL_A_FEATURES)

    # Test prediction pipeline
    sample = {
        'visit_id': 9999, 'patient_id': 42, 'visit_date': '2025-12-01',
        'registration_date': '2025-06-01', 'age': 65, 'chronic_flag': 1,
        'department': 'Cardiology', 'visit_type': 'ER', 'gender': 'M',
        'city': 'Mumbai', 'insurance_provider': 'HealthPlus',
        'length_of_stay_hours': 24.5, 'billed_amount': 35000.0,
        'approved_amount': 35000.0, 'payment_days': 15.0,
    }

    ma, mb, schema = load_models()
    resp_a = predict_visit_risk(sample, ma)
    resp_b = predict_claim_outcome(sample, mb)

    print(f"\nModel A prediction: {resp_a['predicted_class']} (confidence: {resp_a['confidence']:.3f})")
    print(f"Model B prediction: {resp_b['predicted_class']} (confidence: {resp_b['confidence']:.3f})")

    logger = PredictionLogger('_test_log.csv')
    logger.log('model_a', resp_a)
    logger.log('model_b', resp_b)
    print(f"\nLogger summary: {logger.summary()}")

    h = hash_input(sample)
    print(f"Input hash: {h}")

    import shutil, os
    os.remove('_test_log.csv')
    shutil.rmtree('models', ignore_errors=True)

    print("\n✅ phase5_utils self-test passed.")
