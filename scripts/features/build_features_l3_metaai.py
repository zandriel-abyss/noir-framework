"""
L3 - Meta-AI Features
  - Anomaly detection
  - Clustering
  - Combined risk tag
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib # For saving/loading models and scaler
from pathlib import Path
from scripts.features.feature_utils import calculate_l3_metaai_features

# File paths
INPUT_FILE = 'datasource/processed/features_l2_riskflags.csv'
OUTPUT_FILE = 'datasource/processed/features_l3_metaai.csv'
MODELS_DIR = Path('output/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load L2 features
df = pd.read_csv(INPUT_FILE)

# Create default columns if missing (handled in feature_utils for single wallet, but good for batch consistency)
if 'smart_contract_failures' not in df.columns:
    df['smart_contract_failures'] = 0
if 'layer_hopping_count' not in df.columns:
    df['layer_hopping_count'] = 0
if 'circular_tx_ratio' not in df.columns:
    df['circular_tx_ratio'] = 0.0
if 'avg_gas_fee' not in df.columns:
    df['avg_gas_fee'] = 0.0

# Set index and drop non-numeric cols for modeling
# Ensure all relevant L0-L2 features are included in X for training the meta-AI models
features_for_metaai = df.drop(columns=['wallet_address'] + (['label'] if 'label' in df.columns else []), errors='ignore')
features_for_metaai = features_for_metaai.fillna(0)

if features_for_metaai.empty:
    print("[WARNING L3 MetaAI] features_for_metaai DataFrame is empty. Skipping model training and saving.")
else:
    # Normalize data
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(features_for_metaai)
    # Save the list of feature columns used for fitting the scaler directly from the scaler object
    joblib.dump(scaler.feature_names_in_.tolist(), MODELS_DIR / 'l3_metaai_feature_columns.joblib')
    print(f"L3 Meta-AI feature columns saved to: {MODELS_DIR / 'l3_metaai_feature_columns.joblib'}")

    # --- Isolation Forest ---
    iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_model.fit(X_scaled_full) # Fit on full data
    df['anomaly_iso'] = iso_model.predict(X_scaled_full)
    df['anomaly_iso'] = df['anomaly_iso'].map({1: 0, -1: 1})  # 1 = anomaly

    # --- DBSCAN (density-based outlier detection) ---
    db_model = DBSCAN(eps=2.0, min_samples=5)
    db_model.fit(X_scaled_full) # Fit on full data
    df['anomaly_dbscan'] = (db_model.labels_ == -1).astype(int) # DBSCAN's labels_ directly give -1 for noise

    print(f"Attempting to save L3 Meta-AI models to {MODELS_DIR.resolve()}")
    # Save the trained models and scaler for real-time inference
    joblib.dump(scaler, MODELS_DIR / 'scaler_l3_metaai.joblib')
    joblib.dump(iso_model, MODELS_DIR / 'iso_forest_l3_metaai.joblib')
    joblib.dump(db_model, MODELS_DIR / 'dbscan_l3_metaai.joblib')
    print(f"L3 Meta-AI models and scaler saved to: {MODELS_DIR.resolve()}")

# Compute combined risk tag (can be done with the utility function or kept here for batch)
# For batch, it's efficient to do this vectorized.
# The utility function calculates for a single wallet, so we'll do it here for the batch process.
df['combined_risk_tag'] = (
    (df['num_fraud_counterparties'] > 2) |
    (df['num_suspicious_counterparties'] > 2) |
    (df['anomaly_iso'] == 1) |
    (df['anomaly_dbscan'] == 1)
).astype(int)

# Save output
df.to_csv(OUTPUT_FILE, index=False)
print(f"L3 meta-AI features saved to: {OUTPUT_FILE}")