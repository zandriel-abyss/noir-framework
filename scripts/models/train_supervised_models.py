import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import os

# === Paths ===
INPUT_FEATURE_FILE = 'datasource/processed/features_final_all_layers.csv'
OUTPUT_DIR = 'output/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_FEATURE_FILE)
X = df.drop(columns=['wallet_address', 'label', 'xai_reason_code'])  # keep only numerical features
y = df['label']

# Encode labels to 0/1/2
label_mapping = {'normal': 0, 'fraud': 1, 'suspicious': 2}
y_encoded = y.map(label_mapping)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === Model Training ===
print("\n📦 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

print("📦 Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

# === Evaluation ===
def evaluate_model(model, name):
    print(f"\n📊 {name} Classification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

evaluate_model(rf, "Random Forest")
evaluate_model(xgb, "XGBoost")

y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
pd.DataFrame({'wallet_address': df.loc[X_test.index, 'wallet_address'], 'true_label': y_test, 'rf_pred': y_pred_rf, 'xgb_pred': y_pred_xgb}).to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
print(f"✅ Predictions saved to {OUTPUT_DIR}/predictions.csv")

# === SHAP Explainability (Optional, CPU-safe) ===
print("\n🧠 Running SHAP (KernelExplainer)... this may take a minute")

# Use first 50 test samples for SHAP preview
X_sample = X_test[:50]
explainer = shap.KernelExplainer(rf.predict, X_train[:100], link="identity")
shap_values = explainer.shap_values(X_sample)

# Plot SHAP summary (fraud class if multiclass)
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_sample, feature_names=X.columns, show=False)
else:
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns, show=False)

plt.savefig(f"{OUTPUT_DIR}/shap_rf_fraud_summary.png", bbox_inches='tight')
print(f"✅ SHAP summary plot saved to {OUTPUT_DIR}/shap_rf_fraud_summary.png")