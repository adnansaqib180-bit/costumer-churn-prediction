import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load trained pipeline (preprocessing + model)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('trained_model.pkl')

model = load_model()

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")
st.write("Customer ki details daalo, model batayega ke wo churn karega ya nahi.")

st.divider()

# -------------------------------------------------
# Input form
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])

with col2:
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)

st.divider()

# -------------------------------------------------
# Predict button
# -------------------------------------------------
if st.button("Predict Churn", type="primary"):

    input_df = pd.DataFrame([{
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }])

    prediction = model.predict(input_df)[0]

    # agar model predict_proba support karta ho (LogisticRegression karta hai)
    try:
        proba = model.predict_proba(input_df)[0][1]
    except AttributeError:
        proba = None

    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ Customer CHURN kar sakta hai")
    else:
        st.success("✅ Customer STAY karega")

    if proba is not None:
        st.write(f"Churn Probability: **{proba:.2%}**")
        st.progress(float(proba))