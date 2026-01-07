# Project Title: End-to-End Real-Time Payment Fraud Detection System

Executive Summary: Developed a machine learning risk engine to detect Authorized Push Payment (APP) fraud. The system reduces financial loss by utilizing behavioral biometrics (active call detection) and transaction velocity, balancing risk suppression with a <0.5% False Positive Rate target.

### Business Value:

Problem: Traditional rules engines miss "social engineering" fraud where the user authorizes the payment.

Solution: A gradient-boosted classifier (XGBoost) that detects coercion patterns (e.g., user on a call, draining 90% of balance, rapid app navigation).

Outcome: Enforcing a dynamic risk policy (Auto-Decline vs. Step-Up Auth) to minimize customer friction.

### Tech Stack:

Data Engineering: DuckDB (SQL Window Functions for feature engineering).

Modeling: XGBoost, SHAP (for model explainability).

App/Deployment: Streamlit (Investigator Dashboard).

Language: Python 3.9.

Key Features:

✅ Velocity Checks: Tracks transaction frequency in 1h/24h windows using SQL.

✅ Context Awareness: flags "Panic Transfers" (High value + Low time-on-page + Active Call).

✅ Human-in-the-Loop: Streamlit dashboard for Analysts to review borderline "Grey Area" cases.


Site Link: https://projectshield.streamlit.app/
