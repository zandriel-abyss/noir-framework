import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# File paths
INPUT_FILE = 'datasource/processed/features_l2_riskflags.csv'
OUTPUT_FILE = 'datasource/processed/features_l3_metaai.csv'

# Load L2 features
df = pd.read_csv(INPUT_FILE)

# Set index and drop non-numeric cols for modeling
wallets = df['wallet_address']
X = df.drop(columns=['wallet_address'] + (['label'] if 'label' in df.columns else []))

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest ---
iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['anomaly_iso'] = iso_model.fit_predict(X_scaled)
df['anomaly_iso'] = df['anomaly_iso'].map({1: 0, -1: 1})  # 1 = anomaly

# --- DBSCAN (density-based outlier detection) ---
db_model = DBSCAN(eps=2.0, min_samples=5)
db_clusters = db_model.fit_predict(X_scaled)
df['anomaly_dbscan'] = (db_clusters == -1).astype(int)

# --- Combined Behavioral Tag ---
df['combined_risk_tag'] = (
    (df['num_fraud_counterparties'] > 2) |
    (df['num_suspicious_counterparties'] > 2) |
    (df['anomaly_iso'] == 1) |
    (df['anomaly_dbscan'] == 1)
).astype(int)

# Reattach wallet and label
if 'wallet_address' not in df.columns:
    df.insert(0, 'wallet_address', wallets)
if 'label' in df.columns:
    df = df[['wallet_address', 'label'] + [col for col in df.columns if col not in ['wallet_address', 'label']]]
else:
    df = df[['wallet_address'] + [col for col in df.columns if col != 'wallet_address']]

# Save output
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ L3 meta-AI features saved to: {OUTPUT_FILE}")