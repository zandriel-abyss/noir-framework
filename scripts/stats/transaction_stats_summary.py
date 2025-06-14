#!/usr/bin/env python3
"""
transaction_stats_summary.py

Reads fraud, normal, and mixer transaction datasets and computes:
- Number of transactions
- Number of unique wallets
- Time span of the data (earliest and latest timestamps)
- Label type

Saves the summary to: `output/transaction_summary.csv`
"""

import pandas as pd
from pathlib import Path

# Paths
RAW_PATH = Path("datasource/raw")
OUT_PATH = Path("output")
OUT_PATH.mkdir(exist_ok=True)

FILES = {
    "fraud": RAW_PATH / "fraud_transactions.csv",
    "normal": RAW_PATH / "normal_transactions.csv",
    "mixer": RAW_PATH / "mixer_interactions.csv",
}

def load_and_summarize(name, path):
    df = pd.read_csv(path)
    
    summary = {
        "label": name,
        "num_transactions": len(df),
        "num_unique_wallets": df['from'].nunique() + df['to'].nunique(),
        "from_wallets": df['from'].nunique(),
        "to_wallets": df['to'].nunique(),
        "time_start": pd.to_datetime(df['timeStamp'], unit='s').min(),
        "time_end": pd.to_datetime(df['timeStamp'], unit='s').max(),
    }
    return summary

def main():
    summaries = []
    for label, path in FILES.items():
        if path.exists():
            print(f"Reading {label}: {path}")
            summaries.append(load_and_summarize(label, path))
        else:
            print(f"Missing file: {path}")

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(OUT_PATH / "transaction_summary.csv", index=False)
    print("\n✅ Saved summary to: output/transaction_summary.csv")

if __name__ == "__main__":
    main()