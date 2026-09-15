from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


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
