"""
FastAPI application for Telco Customer Churn prediction.

Endpoints:
    GET  /         → Health check
    POST /predict  → Predict churn from customer features
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model artifacts at startup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "adaboost_model.joblib")
encoder = joblib.load(MODEL_DIR / "onehot_encoder.joblib")

with open(MODEL_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predict whether a telecom customer will churn based on their account features.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — configurable via ALLOWED_ORIGINS env var
# ---------------------------------------------------------------------------
default_origins = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080,http://127.0.0.1:8080"
allowed_origins = os.getenv("ALLOWED_ORIGINS", default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic input model — accepts raw (human-readable) feature values
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    """Input schema matching the raw dataset columns (before encoding)."""

    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, le=72, description="Months with the company (0–72)")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)",
    )
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Preprocessing — exact replication of notebook encoding
# ---------------------------------------------------------------------------

BINARY_MAP = {"Yes": 1, "No": 0}
SERVICE_MAP = {"Yes": 1, "No": 0, "No internet service": 2, "No phone service": 2}
CONTRACT_MAP = {"Month-to-month": 1, "One year": 2, "Two year": 3}

SERVICE_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

VALID_PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
VALID_INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]


def preprocess(data: CustomerInput) -> pd.DataFrame:
    """Convert raw customer input into the 22-feature DataFrame the model expects."""

    row = data.model_dump()

    # Validate categorical values
    if row["Partner"] not in BINARY_MAP:
        raise HTTPException(status_code=422, detail="Partner must be 'Yes' or 'No'")
    if row["Dependents"] not in BINARY_MAP:
        raise HTTPException(status_code=422, detail="Dependents must be 'Yes' or 'No'")
    if row["PaperlessBilling"] not in BINARY_MAP:
        raise HTTPException(status_code=422, detail="PaperlessBilling must be 'Yes' or 'No'")
    if row["Contract"] not in CONTRACT_MAP:
        raise HTTPException(
            status_code=422,
            detail="Contract must be 'Month-to-month', 'One year', or 'Two year'",
        )
    for col in SERVICE_COLS:
        if row[col] not in SERVICE_MAP:
            raise HTTPException(
                status_code=422,
                detail=f"{col} must be 'Yes', 'No', 'No internet service', or 'No phone service'",
            )
    if row["PaymentMethod"] not in VALID_PAYMENT_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"PaymentMethod must be one of: {VALID_PAYMENT_METHODS}",
        )
    if row["InternetService"] not in VALID_INTERNET_SERVICES:
        raise HTTPException(
            status_code=422,
            detail=f"InternetService must be one of: {VALID_INTERNET_SERVICES}",
        )

    # --- Apply encoding (same as notebook) ---
    # Binary columns
    row["Partner"] = BINARY_MAP[row["Partner"]]
    row["Dependents"] = BINARY_MAP[row["Dependents"]]
    row["PaperlessBilling"] = BINARY_MAP[row["PaperlessBilling"]]

    # Service columns (ordinal)
    for col in SERVICE_COLS:
        row[col] = SERVICE_MAP[row[col]]

    # Contract (ordinal)
    row["Contract"] = CONTRACT_MAP[row["Contract"]]

    # One-hot encode PaymentMethod & InternetService using the saved encoder
    nominal_df = pd.DataFrame(
        [[row["PaymentMethod"], row["InternetService"]]],
        columns=["PaymentMethod", "InternetService"],
    )
    encoded_array = encoder.transform(nominal_df)
    encoded_cols = encoder.get_feature_names_out(["PaymentMethod", "InternetService"])

    # Build feature dict (without the nominal columns that were one-hot encoded)
    feature_dict = {
        "SeniorCitizen": row["SeniorCitizen"],
        "Partner": row["Partner"],
        "Dependents": row["Dependents"],
        "tenure": row["tenure"],
        "MultipleLines": row["MultipleLines"],
        "OnlineSecurity": row["OnlineSecurity"],
        "OnlineBackup": row["OnlineBackup"],
        "DeviceProtection": row["DeviceProtection"],
        "TechSupport": row["TechSupport"],
        "StreamingTV": row["StreamingTV"],
        "StreamingMovies": row["StreamingMovies"],
        "Contract": row["Contract"],
        "PaperlessBilling": row["PaperlessBilling"],
        "MonthlyCharges": row["MonthlyCharges"],
        "TotalCharges": row["TotalCharges"],
    }

    # Add one-hot encoded columns
    for col_name, val in zip(encoded_cols, encoded_array[0]):
        feature_dict[col_name] = val

    # Build DataFrame in the exact column order the model expects
    df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health check — confirms the API is running."""
    return {
        "status": "running",
        "message": "Telco Customer Churn Prediction API is live.",
        "docs": "/docs",
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    """Predict whether a customer will churn."""
    try:
        features_df = preprocess(customer)
        prediction = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------------------------
# Run with: uvicorn app.main:app --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
