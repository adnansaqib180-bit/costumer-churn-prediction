from __future__ import annotations

import requests
import streamlit as st

API_BASE_URL = "https://discerning-emotion-production-4278.up.railway.app"


st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --orange: #ff7a18;
        --orange-light: #ffb347;
        --ink: #090909;
        --panel: #151515;
        --muted: #aaa39b;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(255, 122, 24, .18), transparent 28rem),
            linear-gradient(135deg, #090909 0%, #11100f 55%, #1b110b 100%);
        color: #f7f4ef;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #28221d;
    }

    [data-testid="stSidebar"] * { color: #f7f4ef; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    h1 { font-size: clamp(2.4rem, 5vw, 4.8rem) !important; line-height: 1 !important; }
    h2 { color: var(--orange-light); }

    .eyebrow {
        color: var(--orange);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.08rem;
        max-width: 42rem;
        line-height: 1.65;
    }

    .model-card {
        background: linear-gradient(135deg, rgba(255, 122, 24, .18), rgba(255, 122, 24, .04));
        border: 1px solid rgba(255, 145, 60, .38);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: .5rem 0 1.5rem;
    }

    .model-card strong { color: var(--orange-light); }
    .result-card {
        background: #151515;
        border: 1px solid #33271f;
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #ff7412, #ff9d3e);
        border: 0;
        border-radius: 10px;
        color: #120b06;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        min-height: 3rem;
        width: 100%;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #ff8a32, #ffc06d);
        color: #120b06;
    }

    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #171717;
        border-radius: 8px;
    }

    [data-testid="stMetricValue"] { color: var(--orange-light); }
    hr { border-color: #33271f; }
    </style>
    """,
    unsafe_allow_html=True,
)


def input_form() -> dict | None:
    """Render the customer form and return a validated API input object."""
    with st.form("customer_details"):
        st.markdown("### Customer profile")
        left, right = st.columns(2, gap="large")

        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone service", ["No", "Yes"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )

        with right:
            online_backup = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly charges", min_value=0.0, value=70.0, step=1.0
            )
            total_charges = st.number_input(
                "Total charges", min_value=0.0, value=1000.0, step=10.0
            )

        submitted = st.form_submit_button("Analyze churn risk", type="primary")

    if not submitted:
        return None

    return {
        "gender": gender,
        "SeniorCitizen": senior_citizen == "Yes",
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


def request_prediction(endpoint: str, customer_input: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=customer_input,
        timeout=90,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned {response.status_code}: {detail}")
    return response.json()


st.sidebar.markdown("## CHURN INTELLIGENCE")
st.sidebar.caption("Customer retention decision workspace")
st.sidebar.caption(f"Connected to: {API_BASE_URL}")
selected_model = st.sidebar.radio(
    "Choose prediction engine",
    ["Machine learning", "Deep learning ANN"],
    help="Select the trained model you want to use for this prediction.",
)
st.sidebar.divider()
st.sidebar.markdown(
    "**How it works**\n\n"
    "Enter a customer's service details, select a model, and get an instant "
    "churn probability."
)

st.markdown('<div class="eyebrow">Retention analytics / 01</div>', unsafe_allow_html=True)
st.title("Know who is about to leave.")
st.markdown(
    '<p class="hero-copy">A focused customer churn workspace for turning account '
    "signals into a clear retention decision.</p>",
    unsafe_allow_html=True,
)

model_name = "Logistic regression pipeline" if selected_model == "Machine learning" else "Artificial neural network"
st.markdown(
    f'<div class="model-card">ACTIVE ENGINE &nbsp; <strong>{model_name}</strong></div>',
    unsafe_allow_html=True,
)

customer_input = input_form()
if customer_input is not None:
    with st.spinner(f"Running {selected_model.lower()} analysis..."):
        try:
            result = request_prediction(
                "/ml-model"
                if selected_model == "Machine learning"
                else "/deep-learning-ann",
                customer_input,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            prediction = result["prediction"]
            probability = result.get("probability")
            risk = "HIGH RISK" if prediction else "LOW RISK"
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### Prediction result")
            metric_col, probability_col = st.columns(2)
            with metric_col:
                st.metric("Churn status", result["label"].upper())
                st.caption(f"{risk}  |  {result['model']}")
            with probability_col:
                if probability is not None:
                    st.metric("Churn probability", f"{probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
            if prediction:
                st.warning("This customer shows signals associated with churn. Consider a retention offer.")
            else:
                st.success("This customer currently shows a healthy retention profile.")
            st.markdown("</div>", unsafe_allow_html=True)
