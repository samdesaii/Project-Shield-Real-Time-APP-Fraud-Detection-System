import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# --- CONFIGURATION ---
st.set_page_config(page_title="Project Shield: Fraud Detection", layout="wide")

# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, model_features

model, model_features = load_artifacts()

# --- SIDEBAR: SIMULATION PANEL ---
st.sidebar.header("📝 Transaction Simulator")
st.sidebar.markdown("Adjust values to test the Risk Engine.")

# Input fields (matching our original data logic)
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=1200.0, step=10.0)
balance = st.sidebar.number_input("Current Account Balance ($)", min_value=0.0, value=5000.0, step=100.0)
time_on_page = st.sidebar.slider("Time on Page (Seconds)", 5, 180, 20)
is_call_active = st.sidebar.checkbox("Is Customer on Active Call?", value=True)
is_new_beneficiary = st.sidebar.checkbox("Is New Beneficiary?", value=True)
bank_type = st.sidebar.selectbox("Beneficiary Bank Type", ["Traditional", "NeoBank", "Crypto_Exchange", "International"])

# Derived Feature Calculation
percent_balance = (amount / balance * 100) if balance > 0 else 0

# --- MAIN DASHBOARD ---
st.title("🛡️ Project Shield: Risk Analyst Console")

# 1. PREPROCESS INPUT (The Tricky Part)
def preprocess_input(amount, percent_balance, time_on_page, is_call, is_new, bank, features):
    # Create a dict with ALL model features initialized to 0
    input_data = {col: 0 for col in features}
    
    # Fill in the numerical/boolean values
    input_data['amount'] = amount
    input_data['percent_balance_transferred'] = percent_balance
    input_data['time_on_page_sec'] = time_on_page
    input_data['is_call_active'] = int(is_call)
    input_data['is_new_beneficiary'] = int(is_new)
    
    # Handle One-Hot Encoding for Bank Type
    # If model expects 'beneficiary_bank_type_NeoBank', set it to 1
    col_name = f"beneficiary_bank_type_{bank}"
    if col_name in input_data:
        input_data[col_name] = 1
        
    # Add dummy velocity features (In a real app, these come from DB)
    # We set them to 'High' to simulate a risky scenario for demo purposes
    input_data['count_last_1h'] = 2 
    input_data['count_last_24h'] = 5
    input_data['avg_amt_last_30d'] = 500
    input_data['ratio_to_avg_amt'] = amount / 500 if 500 > 0 else 0
    
    return pd.DataFrame([input_data])

# 2. GET PREDICTION
input_df = preprocess_input(amount, percent_balance, time_on_page, is_call_active, is_new_beneficiary, bank_type, model_features)
prob = model.predict_proba(input_df)[0][1] # Probability of Fraud (0.0 to 1.0)
prob_score = round(prob * 100, 1)

# 3. APPLY RISK POLICY (From Phase 1)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Probability Score", f"{prob_score}/100")

with col2:
    if prob_score < 40:
        risk_level = "LOW"
        color = "green"
        action = "✅ AUTO-APPROVE"
    elif prob_score < 80:
        risk_level = "MEDIUM"
        color = "orange"
        action = "⚠️ STEP-UP AUTH (2FA)"
    else:
        risk_level = "HIGH"
        color = "red"
        action = "⛔ AUTO-DECLINE"
        
    st.markdown(f"### Risk Level: :{color}[{risk_level}]")

with col3:
    st.markdown(f"### Engine Decision:")
    st.subheader(action)

# 4. EXPLAINABILITY (Visualizing the "Why")
st.divider()
st.subheader("🔍 Real-Time Factor Analysis")
st.write("Why did the model make this decision?")

# Simple bar chart of key drivers
# (In a real app, we would use SHAP here, but for speed, we visualize inputs)
factors = {
    "Amount ($)": amount,
    "% of Balance": percent_balance,
    "Time on Page (s)": time_on_page,
    "Active Call?": 100 if is_call_active else 0
}
st.bar_chart(factors)

# 5. RAW DATA VIEW
with st.expander("View Raw Model Input Vector"):
    st.dataframe(input_df)