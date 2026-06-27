# Phase 5 — API Request & Response Documentation
**Hospital Risk Intelligence API — v1.0.0**

---

## Base URL

```
http://localhost:8000
```

For Docker deployments, replace `localhost` with the server IP or hostname.

---

## Authentication

The current version does not require authentication (suitable for internal hospital network deployment). For production internet-facing deployment, add an API key header or JWT token — consult the Phase 6 governance document.

---

## Endpoint Reference

---

### `GET /health`

Returns the current health status of the API and confirms both models are loaded.

**Request:**
```
GET /health
```

**Response — 200 OK:**
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "timestamp": "2026-06-25T10:00:00+00:00",
  "models_loaded": {
    "model_a_risk": true,
    "model_b_claim": true
  }
}
```

**Use this to:** Verify the service is running before sending prediction requests. Integrate with monitoring tools (Prometheus, Grafana, CloudWatch).

---

### `GET /models/info`

Returns metadata for both loaded models, including features, training dates, and performance metrics from Phase 4.

**Request:**
```
GET /models/info
```

**Response — 200 OK (abbreviated):**
```json
{
  "model_version": "1.0.0",
  "model_a": {
    "description": "Classifies hospital visits as Low / Medium / High operational risk",
    "target": "risk_score",
    "classes": ["Low", "Medium", "High"],
    "best_model": "RandomForest",
    "train_cutoff_date": "2025-11-08",
    "test_accuracy": 0.4108,
    "test_macro_f1": 0.3421
  },
  "model_b": {
    "description": "Predicts insurance claim outcome: Paid / Pending / Rejected",
    "target": "claim_status",
    "classes": ["Paid", "Pending", "Rejected"],
    "best_model": "RandomForest_SMOTE",
    "train_cutoff_date": "2025-11-08",
    "test_accuracy": 0.9546,
    "test_macro_f1": 0.9373
  },
  "label_encodings": {
    "department": ["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"],
    "visit_type": ["ER", "ICU", "OPD"],
    "gender": ["F", "M"],
    "city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"],
    "insurance_provider": ["CareOne", "HealthPlus", "MediCareX", "SecureLife"]
  }
}
```

---

### `POST /predict/risk`

**Model A — Visit Risk Classification**

Predicts whether a hospital visit is `Low`, `Medium`, or `High` operational risk based on patient and visit information available at admission time.

---

#### Request Schema

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `age` | integer | ✅ | 0–120 | Patient age in years |
| `gender` | string | ✅ | `"F"` or `"M"` | Patient gender |
| `city` | string | ✅ | See allowed values | Patient's home city |
| `chronic_flag` | integer | ✅ | 0 or 1 | 1 if patient has a chronic condition |
| `department` | string | ✅ | See allowed values | Hospital department |
| `visit_type` | string | ✅ | `"ER"`, `"ICU"`, `"OPD"` | Type of admission |
| `length_of_stay_hours` | float | ✅ | ≥ 0 | Actual or expected length of stay |
| `patient_visit_frequency` | integer | ✅ | ≥ 1 | Total historical visits by this patient |
| `avg_los_per_patient` | float | ✅ | ≥ 0 | Patient's average length of stay historically |
| `days_since_registration` | integer | ✅ | ≥ 0 | Days between registration date and visit date |
| `visit_month` | integer | ✅ | 1–12 | Month of visit |
| `visit_day_of_week` | integer | ✅ | 0–6 | 0 = Monday, 6 = Sunday |

---

#### Example 1 — High-Risk ICU Patient

**Request:**
```json
{
  "age": 75,
  "gender": "M",
  "city": "Mumbai",
  "chronic_flag": 1,
  "department": "ICU",
  "visit_type": "ICU",
  "length_of_stay_hours": 72.0,
  "patient_visit_frequency": 8,
  "avg_los_per_patient": 55.0,
  "days_since_registration": 1200,
  "visit_month": 1,
  "visit_day_of_week": 0
}
```

**Response — 200 OK:**
```json
{
  "prediction": "High",
  "probabilities": {
    "High": 0.54,
    "Medium": 0.31,
    "Low": 0.15
  },
  "model": "model_a_visit_risk",
  "model_version": "1.0.0",
  "timestamp": "2026-06-25T10:15:42+00:00",
  "business_note": "High Risk: consider immediate specialist review and resource pre-allocation. Medium Risk: schedule follow-up within 24 hours. Low Risk: standard care pathway."
}
```

**Clinical Action:** Trigger specialist assignment workflow and pre-allocate ICU bed.

---

#### Example 2 — Low-Risk OPD Visit

**Request:**
```json
{
  "age": 28,
  "gender": "F",
  "city": "Bangalore",
  "chronic_flag": 0,
  "department": "General",
  "visit_type": "OPD",
  "length_of_stay_hours": 1.5,
  "patient_visit_frequency": 1,
  "avg_los_per_patient": 1.5,
  "days_since_registration": 30,
  "visit_month": 6,
  "visit_day_of_week": 2
}
```

**Response — 200 OK:**
```json
{
  "prediction": "Low",
  "probabilities": {
    "High": 0.09,
    "Medium": 0.21,
    "Low": 0.70
  },
  "model": "model_a_visit_risk",
  "model_version": "1.0.0",
  "timestamp": "2026-06-25T10:16:00+00:00",
  "business_note": "High Risk: consider immediate specialist review and resource pre-allocation. Medium Risk: schedule follow-up within 24 hours. Low Risk: standard care pathway."
}
```

**Clinical Action:** Route to standard OPD pathway.

---

#### Error Response — Invalid Category (422)

**Request (invalid city):**
```json
{
  "age": 45,
  "gender": "M",
  "city": "Tokyo",
  ...
}
```

**Response — 422 Unprocessable Entity:**
```json
{
  "detail": "Invalid value 'Tokyo' for field 'city'. Allowed values: ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai', 'Pune']"
}
```

---

#### Error Response — Field Out of Range (422)

**Request (age = 200):**
```json
{
  "age": 200,
  ...
}
```

**Response — 422 Unprocessable Entity:**
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be less than or equal to 120",
      "input": 200
    }
  ]
}
```

---

### `POST /predict/claim`

**Model B — Insurance Claim Outcome Classification**

Predicts whether an insurance claim will be `Paid`, `Pending`, or `Rejected` before submission to the insurer.

> **Production note:** The `approved_amount` and `approval_ratio` fields should use historical averages for the provider/department combination, not post-submission actual values. See the model card (Phase 4) for full details.

---

#### Request Schema

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `billed_amount` | float | ✅ | ≥ 0 | Total amount billed to the insurer |
| `approved_amount` | float | ✅ | ≥ 0 | Use historical average for provider/dept |
| `approval_ratio` | float | ✅ | 0.0–1.0 | Historical approval rate (approved ÷ billed) |
| `length_of_stay_hours` | float | ✅ | ≥ 0 | Length of stay in hours |
| `department` | string | ✅ | See allowed values | Hospital department |
| `visit_type` | string | ✅ | `"ER"`, `"ICU"`, `"OPD"` | Type of admission |
| `age` | integer | ✅ | 0–120 | Patient age |
| `gender` | string | ✅ | `"F"` or `"M"` | Patient gender |
| `chronic_flag` | integer | ✅ | 0 or 1 | Chronic condition flag |
| `insurance_provider` | string | ✅ | See allowed values | Insurance company |
| `patient_visit_frequency` | integer | ✅ | ≥ 1 | Total visits by this patient |
| `days_since_registration` | integer | ✅ | ≥ 0 | Days since registration |
| `visit_month` | integer | ✅ | 1–12 | Month of visit |

---

#### Example 1 — Likely Rejected Claim

**Request:**
```json
{
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
}
```

**Response — 200 OK:**
```json
{
  "prediction": "Rejected",
  "probabilities": {
    "Paid": 0.08,
    "Pending": 0.11,
    "Rejected": 0.81
  },
  "rejection_risk": "81.0%",
  "recommended_action": "HIGH REJECTION RISK — review and correct before submission",
  "model": "model_b_claim_outcome",
  "model_version": "1.0.0",
  "timestamp": "2026-06-25T10:20:00+00:00"
}
```

**Finance Action:** Hold claim, review documentation, verify coverage eligibility, contact insurer before submission.

---

#### Example 2 — Likely Paid Claim

**Request:**
```json
{
  "billed_amount": 15000.0,
  "approved_amount": 14900.0,
  "approval_ratio": 0.993,
  "length_of_stay_hours": 6.0,
  "department": "General",
  "visit_type": "OPD",
  "age": 35,
  "gender": "M",
  "chronic_flag": 0,
  "insurance_provider": "SecureLife",
  "patient_visit_frequency": 2,
  "days_since_registration": 200,
  "visit_month": 8
}
```

**Response — 200 OK:**
```json
{
  "prediction": "Paid",
  "probabilities": {
    "Paid": 0.93,
    "Pending": 0.06,
    "Rejected": 0.01
  },
  "rejection_risk": "1.0%",
  "recommended_action": "LOW REJECTION RISK — proceed with standard submission",
  "model": "model_b_claim_outcome",
  "model_version": "1.0.0",
  "timestamp": "2026-06-25T10:21:00+00:00"
}
```

**Finance Action:** Proceed with standard claim submission.

---

#### Rejection Risk Thresholds

The `recommended_action` field uses these thresholds:

| `rejection_risk` | `recommended_action` |
|---|---|
| ≥ 70% | HIGH REJECTION RISK — review and correct before submission |
| 40–69% | MODERATE REJECTION RISK — verify supporting documentation |
| < 40% | LOW REJECTION RISK — proceed with standard submission |

---

### `GET /logs/recent`

Returns the most recent prediction log entries.

**Query Parameters:**

| Parameter | Type | Default | Max | Description |
|---|---|---|---|---|
| `n` | integer | 10 | 100 | Number of recent records to return |

**Request:**
```
GET /logs/recent?n=3
```

**Response — 200 OK:**
```json
{
  "total_logged": 47,
  "returned": 3,
  "logs": [
    {
      "timestamp": "2026-06-25T10:21:00+00:00",
      "model_version": "1.0.0",
      "endpoint": "predict/claim",
      "input_hash": "b7f3a1d2c4e5f8a9",
      "prediction": "Paid",
      "probabilities": {"Paid": 0.93, "Pending": 0.06, "Rejected": 0.01}
    },
    {
      "timestamp": "2026-06-25T10:20:00+00:00",
      "model_version": "1.0.0",
      "endpoint": "predict/claim",
      "input_hash": "a3c9e2f1b4d7e8f0",
      "prediction": "Rejected",
      "probabilities": {"Paid": 0.08, "Pending": 0.11, "Rejected": 0.81}
    },
    {
      "timestamp": "2026-06-25T10:15:42+00:00",
      "model_version": "1.0.0",
      "endpoint": "predict/risk",
      "input_hash": "d1e8f5a2c3b9f4e7",
      "prediction": "High",
      "probabilities": {"High": 0.54, "Medium": 0.31, "Low": 0.15}
    }
  ]
}
```

---

## HTTP Status Codes

| Code | Meaning |
|---|---|
| `200 OK` | Successful prediction or data retrieval |
| `422 Unprocessable Entity` | Invalid input — field value out of range or unknown category |
| `500 Internal Server Error` | Unexpected model or server error — check server logs |

---

## Allowed Field Values (Quick Reference)

```
gender:             F, M
department:         Cardiology, ER, General, ICU, Neurology, Orthopedics
visit_type:         ER, ICU, OPD
city:               Bangalore, Chennai, Delhi, Hyderabad, Mumbai, Pune
insurance_provider: CareOne, HealthPlus, MediCareX, SecureLife
```

All string values are **case-sensitive**.
