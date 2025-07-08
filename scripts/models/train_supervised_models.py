import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="Train and evaluate fraud detection models.")
parser.add_argument('--input', type=str, default='datasource/processed/features_for_training.csv', help='Path to input CSV with features.')
parser.add_argument('--output', type=str, default='output/models', help='Directory to save outputs.')
args = parser.parse_args()

INPUT_FEATURE_FILE = args.input
OUTPUT_DIR = args.output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_FEATURE_FILE)
# Keep wallet_address for tracking, but don't use it for modeling
if 'wallet_address' not in df.columns:
    raise ValueError("Input data must contain a 'wallet_address' column for tracking.")
wallet_addresses = df['wallet_address']
X = df.drop(columns=['label', 'wallet_address'], errors='ignore')
y_encoded = df['label']

# === Train-Test Split ===
X_train, X_test, y_train, y_test, wa_train, wa_test = train_test_split(
    X, y_encoded, wallet_addresses, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Model Training ===
print("\n Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

rf_importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
rf_importances.sort_values(by='importance', ascending=False).to_csv(f"{OUTPUT_DIR}/rf_feature_importances.csv", index=False)

print(" Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

# === Evaluation ===
def evaluate_model(model, name):
    print(f"\n {name} Classification Report:")
    y_pred = model.predict(X_test)
    target_names = [str(cls) for cls in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    report = classification_report(y_test, model.predict(X_test), target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_classification_report.csv")

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_confusion_matrix.csv")

evaluate_model(rf, "Random Forest")
evaluate_model(xgb, "XGBoost")

y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
pd.DataFrame({
    'wallet_address': wa_test.reset_index(drop=True),
    'true_label': y_test.reset_index(drop=True),
    'rf_pred': pd.Series(y_pred_rf),
    'xgb_pred': pd.Series(y_pred_xgb)
}).to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
print(f" Predictions saved to {OUTPUT_DIR}/predictions.csv")

# === SHAP Explainability (Optional, CPU-safe) ===
print("\n Running SHAP (TreeExplainer)...")
# Ensure SHAP sampling reflects enough features
shap_sample_size = min(100, len(X_test))
X_sample = X_test.sample(n=shap_sample_size, random_state=42)
print("SHAP sample columns:", X_sample.columns.tolist())
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    for i, class_shap in enumerate(shap_values):
        shap.summary_plot(class_shap, X_sample, show=False)
        plt.savefig(f"{OUTPUT_DIR}/shap_rf_class{i}_beeswarm.png", bbox_inches='tight')
        plt.clf()
        print(f" SHAP beeswarm plot for class {i} saved to {OUTPUT_DIR}/shap_rf_class{i}_beeswarm.png")

        shap.summary_plot(class_shap, X_sample, plot_type="bar", show=False)
        plt.savefig(f"{OUTPUT_DIR}/shap_rf_class{i}_bar.png", bbox_inches='tight')
        plt.clf()
        print(f" SHAP bar plot for class {i} saved to {OUTPUT_DIR}/shap_rf_class{i}_bar.png")
else:
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(f"{OUTPUT_DIR}/shap_rf_summary.png", bbox_inches='tight')
    plt.clf()
    print(f" SHAP summary plot saved to {OUTPUT_DIR}/shap_rf_summary.png")