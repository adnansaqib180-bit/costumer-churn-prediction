from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = next(
    (
        directory
        for directory in (ROOT_DIR / "models", ROOT_DIR / "MODELS")
        if directory.exists()
    ),
    ROOT_DIR / "models",
)
ML_MODEL_PATH = MODEL_DIR / "trained_model.pkl"
ANN_MODEL_PATH = next(
    (
        model_path
        for model_path in (
            MODEL_DIR / "final_ann.keras",
            MODEL_DIR / "trained_model.keras",
            MODEL_DIR / "churn_ann.keras",
        )
        if model_path.exists()
    ),
    MODEL_DIR / "final_ann.keras",
)

app = FastAPI(title="Customer Churn Prediction API")


class CustomerChurnInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customerID: str | None = None
    gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: int | bool = 0
    Partner: Literal["No", "Yes"] = "No"
    Dependents: Literal["No", "Yes"] = "No"
    tenure: int = 1
    PhoneService: Literal["No", "Yes"] = "Yes"
    MultipleLines: Literal["No", "Yes", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "DSL"
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = "No"
    OnlineBackup: Literal["No", "Yes", "No internet service"] = "No"
    DeviceProtection: Literal["No", "Yes", "No internet service"] = "No"
    TechSupport: Literal["No", "Yes", "No internet service"] = "No"
    StreamingTV: Literal["No", "Yes", "No internet service"] = "No"
    StreamingMovies: Literal["No", "Yes", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["No", "Yes"] = "No"
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float | str = 1000.0

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def normalize_senior_citizen(cls, value):
        if isinstance(value, bool):
            return int(value)
        return value


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    label: str
    model: str


def _prepare_ml_dataframe(payload: CustomerChurnInput) -> pd.DataFrame:
    record = payload.model_dump(exclude_none=True)
    df = pd.DataFrame([record])
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes", False: "No", True: "Yes"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    ml_columns = [
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    return df[ml_columns]


@lru_cache(maxsize=1)
def load_ml_model():
    if not ML_MODEL_PATH.exists():
        raise FileNotFoundError(f"ML model not found at {ML_MODEL_PATH}")
    return joblib.load(ML_MODEL_PATH)


@lru_cache(maxsize=1)
def load_ann_model():
    if not ANN_MODEL_PATH.exists():
        raise FileNotFoundError(f"ANN model not found at {ANN_MODEL_PATH}")

    try:
        from keras.models import load_model
    except Exception:
        try:
            from tensorflow.keras.models import load_model
        except Exception as exc:  
            raise RuntimeError("TensorFlow/Keras is not installed. Install it to use the ANN endpoint.") from exc

    return load_model(ANN_MODEL_PATH)


def _prepare_ann_dataframe(payload: CustomerChurnInput) -> pd.DataFrame:
    record = payload.model_dump(exclude_none=True)
    df = pd.DataFrame([record])
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"], errors="ignore")
    df = df.drop(columns=["PaperlessBilling", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection"], errors="ignore")

    df["MultipleLines"] = df["MultipleLines"].map({"No internet service": "No", "No": "No", "Yes": "Yes", "No phone service": "No"})
    for col in ["Partner", "Dependents", "PhoneService", "MultipleLines"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

    df = pd.get_dummies(data=df, columns=["PaymentMethod"], drop_first=True, dtype=int)
    for col in ["TechSupport", "StreamingMovies", "StreamingTV"]:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})

    expected_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "TechSupport",
        "StreamingMovies",
        "StreamingTV",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df[expected_columns]


@app.get("/")
def welcome():
    return {
        "message": "Welcome to the Customer Churn Prediction API",
        "models": [
            "machine-learning-model",
            "deep-learning-ann",
        ],
    }


@app.post("/ml-model", response_model=PredictionResponse)
def predict_ml(payload: CustomerChurnInput):
    try:
        model = load_ml_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load ML model: {exc}") from exc

    try:
        df = _prepare_ml_dataframe(payload)
        prediction = int(model.predict(df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0][1])
        label = "Churn" if prediction == 1 else "No Churn"
        return PredictionResponse(prediction=prediction, probability=probability, label=label, model="machine-learning-model")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {exc}") from exc


@app.post("/deep-learning-ann", response_model=PredictionResponse)
def predict_ann(payload: CustomerChurnInput):
    try:
        model = load_ann_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load ANN model: {exc}") from exc

    try:
        df = _prepare_ann_dataframe(payload)
        prediction_probability = float(model.predict(df, verbose=0).reshape(-1)[0])
        prediction = int(prediction_probability >= 0.5)
        label = "Churn" if prediction == 1 else "No Churn"
        return PredictionResponse(prediction=prediction, probability=prediction_probability, label=label, model="deep-learning-ann")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ANN prediction failed: {exc}") from exc
