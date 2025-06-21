# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload transaction data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:", data.head())

    # Drop columns not present during model training
    for col in ["Time", "Class"]:
     if col in data.columns:
        data = data.drop(col, axis=1)


    # Handle scaling
    scaled_data = scaler.transform(data)

    # Predict
    predictions = model.predict(scaled_data)
    probs = model.predict_proba(scaled_data)[:, 1]

    # Show results
    data["Fraud Probability"] = probs
    data["Prediction"] = predictions
    st.write("🔍 Prediction Results:", data[["Fraud Probability", "Prediction"]])
