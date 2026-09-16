# Churn Intelligence

<div align="center">

### Customer churn prediction with machine learning, deep learning, and a polished Streamlit experience

Predict retention risk from customer account details and compare two trained prediction engines through one simple interface.

<br />

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## Overview

**Churn Intelligence** is an end-to-end customer churn prediction project for the telecom domain. It combines:

- A tuned scikit-learn machine learning pipeline
- A trained Keras artificial neural network
- A FastAPI prediction service
- A dark, orange-accented Streamlit dashboard
- Docker support for reproducible deployment

The application turns customer profile and service information into a clear result:

> **Churn** or **No Churn**, together with an estimated churn probability.

## Highlights

| Capability | Description |
| --- | --- |
| Two prediction engines | Switch between the machine learning pipeline and deep learning ANN from the sidebar |
| Streamlit dashboard | Attractive orange/black interface designed for quick retention decisions |
| FastAPI backend | Typed request validation and separate endpoints for both models |
| Probability scoring | Shows the model's estimated churn probability when available |
| Reusable preprocessing | Input transformations match the training pipelines |
| Docker-ready | Run the dashboard or API in a consistent Python 3.11 environment |

## Project structure

```text
costumer-churn-prediction/
├── api/
│   ├── main.py              # FastAPI application and prediction endpoints
│   └── schemas.py           # Shared request/response schemas
├── models/
│   ├── trained_model.pkl    # Serialized scikit-learn pipeline
│   └── final_ann.keras      # Trained Keras ANN with built-in Normalization layer
├── training/
│   ├── ML/train_ml.py       # Machine learning training script
│   └── DL/
│       ├── train_dl.py      # ANN training script
│       └── test_dl.py       # ANN evaluation script
├── Data.csv                # Training data
├── user_interface.py       # Streamlit dashboard
├── Dockerfile              # Container image definition
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick start

### 1. Create an environment

Windows PowerShell:

```powershell
cd "C:\Users\USER\OneDrive\Desktop\Transaction Fraud Detection\rough\costumer-churn-prediction"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch the Streamlit dashboard

```powershell
python -m streamlit run user_interface.py
```

Open [http://localhost:8501](http://localhost:8501).

### 3. Launch the API

In a second terminal:

```powershell
uvicorn api.main:app --reload
```

Open the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

The deployed API is available at
[https://discerning-emotion-production-4278.up.railway.app](https://discerning-emotion-production-4278.up.railway.app).
The Streamlit dashboard sends prediction requests to this public service.

## Docker

The default container command launches the Streamlit dashboard:

```powershell
docker build -t churn-intelligence .
docker run --rm -p 8501:8501 churn-intelligence
```

Open [http://localhost:8501](http://localhost:8501).

To run the FastAPI service from the same image instead:

```powershell
docker run --rm -p 8000:8000 churn-intelligence `
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs).

For the deployed service, use
[https://discerning-emotion-production-4278.up.railway.app/docs](https://discerning-emotion-production-4278.up.railway.app/docs).

> **Note:** TensorFlow is included in `requirements.txt` so the ANN endpoint and ANN option are available in Docker. The image is therefore larger than a machine-learning-only image.

## API reference

### Welcome

```http
GET /
```

### Machine learning prediction

```http
POST /ml-model
Content-Type: application/json
```

### Deep learning ANN prediction

```http
POST /deep-learning-ann
Content-Type: application/json
```

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
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
  "MonthlyCharges": 59.99,
  "TotalCharges": 500.0
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.5817,
  "label": "Churn",
  "model": "machine-learning-model"
}
```

## Model inputs

The models use customer demographics, tenure, services, contract information, payment method, and billing amounts. The API validates categorical values and normalizes fields such as `TotalCharges` and `SeniorCitizen` before inference.

The machine learning pipeline handles its own preprocessing. The ANN model contains its own Keras `Normalization` layer, so the API only performs categorical feature preparation and passes the resulting features directly to the saved model. It does not need to load the training CSV or fit a second scaler at inference time.

## Training

The training scripts expect the project dataset to be available at the paths used by the scripts. From the project root, review and run the scripts after adjusting any local dataset path if needed:

```powershell
python training\ML\train_ml.py
python training\DL\train_dl.py
python training\DL\test_dl.py
```

The resulting model artifacts should be placed in `models/`:

- `models/trained_model.pkl`
- `models/final_ann.keras`

## Troubleshooting

### ### The ANN endpoint cannot load

Install the dependencies from `requirements.txt`, especially `tensorflow-cpu`, then restart the API. The ANN model is a Keras artifact and requires a TensorFlow/Keras runtime. Confirm that `models/final_ann.keras` is included in the deployment.

### The dashboard cannot find a model

Run the application from the project root so the `models/` directory is available:

```powershell
cd costumer-churn-prediction
python -m streamlit run user_interface.py
```

### Port already in use

Use another port:

```powershell
python -m streamlit run user_interface.py --server.port 8502
uvicorn api.main:app --port 8001
```

## Responsible use

Model predictions are decision-support signals, not definitive judgments about individual customers. Validate performance on current business data, monitor drift, and combine predictions with customer context before taking retention action.

## License

Add the project's license and data usage terms here before public distribution.
