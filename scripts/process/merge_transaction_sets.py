import pandas as pd
from pathlib import Path

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

    # Deduplicate by transaction hash
    before_dedup = len(combined)
    combined.drop_duplicates(subset="hash", inplace=True)
    after_dedup = len(combined)
    print(f"Removed {before_dedup - after_dedup} duplicate transactions based on hash.")

    combined.to_csv(output_path, index=False)
    print(f"Merged {after_dedup} unique transactions into {output_path}")

if __name__ == "__main__":
    main()