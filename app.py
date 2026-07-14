import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", layout="wide")

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    📊 Customer Churn Prediction Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- SIDEBAR ----------
st.sidebar.title("📌 About")
st.sidebar.info(
    "This app predicts whether a customer will churn using XGBoost.\n\n"
    "You can either:\n"
    "- Enter data manually\n"
    "- Upload CSV file"
)

# ---------- INPUT FORM ----------
st.subheader("🧾 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72, 12)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

col4, col5 = st.columns(2)

with col4:
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col5:
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# ---------- ENCODING ----------
def encode(val):
    return hash(val) % 10

input_data = np.array([[ 
    encode(gender), SeniorCitizen, encode(Partner), encode(Dependents),
    tenure, encode(PhoneService), encode(MultipleLines),
    encode(InternetService), encode(OnlineSecurity), encode(OnlineBackup),
    encode(DeviceProtection), encode(TechSupport),
    encode(StreamingTV), encode(StreamingMovies),
    encode(Contract), encode(PaperlessBilling),
    encode(PaymentMethod), MonthlyCharges, TotalCharges
]])

# ---------- PREDICTION ----------
st.markdown("---")

if st.button("🔮 Predict Churn", use_container_width=True):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.error("⚠️ High Risk: Customer Will Leave")
        else:
            st.success("✅ Low Risk: Customer Will Stay")

    with colB:
        st.write("### 📊 Prediction Probability")

        fig, ax = plt.subplots()
        ax.bar(["Stay", "Churn"], prob)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

# ---------- FILE UPLOAD ----------
st.markdown("---")
st.subheader("📂 Bulk Prediction (Upload CSV)")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    try:
        # Simple encoding
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: hash(x) % 10)

        predictions = model.predict(df)

        df["Prediction"] = predictions

        st.success("✅ Predictions Completed!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "predictions.csv")

    except Exception as e:
        st.error(f"Error: {e}")