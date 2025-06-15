import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

INPUT_PATH = "datasource/processed/features_final_all_layers.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

# Drop non-numeric + non-features
drop_cols = ['wallet_address', 'label', 'xai_reason_code']
features = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Check missing values
missing = df.isnull().sum()
missing = missing[missing > 0]

# Near-zero variance
low_variance = features.loc[:, features.std() < 1e-5].columns.tolist()

# Correlation matrix
corr = features.corr()
high_corr_pairs = [(i, j) for i in corr.columns for j in corr.columns 
                   if i != j and abs(corr.loc[i, j]) > 0.95]

# Plot correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("docs/plots/correlation_heatmap.png")
plt.close()

# KDE plots by label (for select features)
plot_cols = ['total_transactions', 'wallet_age_days', 'dormant_awaken_count', 
             'burst_tx_ratio', 'num_fraud_counterparties', 'combined_risk_tag']

for col in plot_cols:
    if col in df.columns:
        plt.figure()
        sns.kdeplot(data=df, x=col, hue='label', fill=True)
        plt.title(f"{col} distribution by label")
        plt.savefig(f"docs/plots/distribution_{col}.png")
        plt.close()

# Summary
print("✅ Feature analysis complete.")
print(f"Missing values in: {list(missing.index)}")
print(f"Low variance features: {low_variance}")
print(f"Highly correlated pairs (>0.95): {high_corr_pairs}")