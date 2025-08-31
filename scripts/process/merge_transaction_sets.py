import pandas as pd
from pathlib import Path
import joblib

# Input paths
DATA_DIR = Path("datasource/raw")
fraud_path = DATA_DIR / "fraud_transactions.csv"
normal_path = DATA_DIR / "normal_transactions.csv"
mixer_path = DATA_DIR / "mixer_interactions.csv"

# Output path
output_path = DATA_DIR / "all_transactions_labeled.csv"

def load_and_label(path, label):
    df = pd.read_csv(path)
    df["label"] = label
    return df

def main():
    print("Merging datasets...")

    df_fraud = load_and_label(fraud_path, "fraud")
    df_normal = load_and_label(normal_path, "normal")
    df_mixer = load_and_label(mixer_path, "suspicious")

    combined = pd.concat([df_fraud, df_normal, df_mixer], ignore_index=True)

    # Ensure 'from' and 'to' addresses are strings and handle potential NaN values
    combined['from'] = combined['from'].astype(str).str.lower()
    combined['to'] = combined['to'].astype(str).str.lower()

    # Deduplicate by transaction hash
    before_dedup = len(combined)
    combined.drop_duplicates(subset="hash", inplace=True)
    after_dedup = len(combined)
    print(f"Removed {before_dedup - after_dedup} duplicate transactions based on hash.")

    # --- Generate and save wallet risk labels ---
    wallet_risk_labels = {}
    for _, row in combined.iterrows():
        addr_from = row['from'].lower()
        addr_to = row['to'].lower()
        label = row['label']

        # Prioritize labels: fraud > suspicious > normal
        current_label_from = wallet_risk_labels.get(addr_from, 'normal')
        current_label_to = wallet_risk_labels.get(addr_to, 'normal')

        if label == 'fraud' or current_label_from == 'fraud' or current_label_to == 'fraud':
            wallet_risk_labels[addr_from] = 'fraud'
            wallet_risk_labels[addr_to] = 'fraud'
        elif label == 'suspicious' or current_label_from == 'suspicious' or current_label_to == 'suspicious':
            wallet_risk_labels[addr_from] = 'suspicious'
            wallet_risk_labels[addr_to] = 'suspicious'
        elif addr_from not in wallet_risk_labels:
            wallet_risk_labels[addr_from] = 'normal'
        elif addr_to not in wallet_risk_labels:
            wallet_risk_labels[addr_to] = 'normal'

    # Save wallet_risk_labels
    models_output_dir = Path("output/models")
    models_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(wallet_risk_labels, models_output_dir / 'wallet_risk_labels.joblib')
    print(f"Wallet risk labels saved to: {models_output_dir / 'wallet_risk_labels.joblib'}")

    # --- Calculate and save global gas 95th percentile ---
    # Ensure 'gasPrice' column exists and is numeric
    if 'gasPrice' in combined.columns:
        combined['gasPrice'] = pd.to_numeric(combined['gasPrice'], errors='coerce').fillna(0)
        global_gas_95th_percentile = combined['gasPrice'].quantile(0.95) if not combined['gasPrice'].empty else 0
    else:
        global_gas_95th_percentile = 0
    
    joblib.dump(global_gas_95th_percentile, models_output_dir / 'global_gas_95th_percentile.joblib')
    print(f"Global gas 95th percentile saved to: {models_output_dir / 'global_gas_95th_percentile.joblib'}")

    combined.to_csv(output_path, index=False)
    print(f"Merged {after_dedup} unique transactions into {output_path}")

if __name__ == "__main__":
    main()