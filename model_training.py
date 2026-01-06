import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import shap
import matplotlib.pyplot as plt
import joblib # To save the model

# 1. Load the "Enriched" Data (Result of Phase 3)
df = pd.read_csv("transactions_enriched.csv")

# 2. Preprocessing
# Drop columns that are unique identifiers (Noise for the model)
# We keep 'is_fraud' as our Target (Y)
cols_to_drop = ['transaction_id', 'customer_id', 'beneficiary_id', 'device_id', 'timestamp', 'location']
df_clean = df.drop(columns=cols_to_drop)

# Convert Categorical Variables (Strings) to Numbers (One-Hot Encoding)
# e.g. 'beneficiary_bank_type' -> 'beneficiary_bank_type_Crypto', 'beneficiary_bank_type_NeoBank'
df_model = pd.get_dummies(df_clean, drop_first=True)

# Define X (Features) and y (Target)
X = df_model.drop('is_fraud', axis=1)
y = df_model['is_fraud']

# 3. Train-Test Split
# We use 'stratify=y' to ensure the test set has the same % of fraud as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} transactions...")

# 4. Train the XGBoost Model
# scale_pos_weight is CRITICAL for fraud. 
# It tells the model: "Pay more attention to the minority class (Fraud)."
fraud_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    scale_pos_weight=fraud_ratio, # Handling Imbalance
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 5. Evaluation (The "Business Acumen" Metrics)
# Accuracy is useless in fraud (95% accuracy could mean you missed ALL fraud).
# We care about Recall (Catching the fraud) and Precision (Not annoying real users).
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probability score (Risk Score)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f} (Don't trust this!)")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f} (Global ranking quality)")
print(f"Precision: {precision_score(y_test, y_pred):.4f} (Hit rate)")
print(f"Recall:    {recall_score(y_test, y_pred):.4f} (Capture rate)")
print(f"AUPRC:     {average_precision_score(y_test, y_prob):.4f} (Best metric for imbalanced data)")

# 6. Explainability (SHAP)
# This explains: "Why did the model flag THIS specific transaction?"
print("\nGenerating SHAP Summary Plot...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Save the plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
print("SHAP Summary Plot saved as 'shap_summary.png'")

# 7. Save Model for Phase 5 (Deployment)
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl') # Save column names to ensure match later
print("Model saved to 'fraud_model.pkl'")