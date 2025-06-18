# Noir Framework – Blockchain Fraud Detection

## Overview

Noir is a modular machine learning framework designed for detecting fraud and enabling AML compliance in Ethereum-based blockchain ecosystems. It combines behavioral feature engineering, supervised and unsupervised modeling, and explainability layers to identify suspicious wallet activity.

## Objectives

- Detect anomalous and fraudulent wallet behavior
- Distinguish malicious activity from privacy-preserving use cases (e.g., mixers)
- Enable real-time risk scoring and compliance integration
- Provide explainable outputs for regulatory traceability

## Architecture Modules

1. **Data Collection**\
   Scripts to collect:

   - Fraudulent addresses (OFAC sanctions, hacker lists)
   - Mixer-linked addresses (e.g., Tornado Cash)
   - Normal wallets (ETH richlist)
   - Transaction logs from Etherscan APIs

2. **Wallet Feature Engineering**

   - Aggregate transaction data to compute:
     - Total tx count, volume
     - Activity span
     - Dormancy and awakenings
     - Counterparty uniqueness
     - Circular tx heuristics
   - Label wallets as `fraud`, `normal`, or `suspicious`

3. **Modeling & Detection Engine**

   - Unsupervised: Isolation Forest, DBSCAN
   - Supervised: Random Forest, XGBoost
   - Label prediction: 0 (normal), 1 (fraud), 2 (suspicious)
   - Live-testing support with temporal validation splits

4. **Explainability & Risk Tagging**

   - SHAP values for global/local importance
   - Rule-based tagging (e.g., dormant big tx, rapid fund movement)
   - Visual anomaly score distributions

5. **Evaluation & Reporting**

   - Confusion matrix, precision-recall, F1
   - Feature importances
   - Distribution drift analysis over time

6. **Live Test Hooks** (Planned)

   - Forward-inference on new wallet samples
   - Sandbox testing with simulated transactions

## Repository Structure

```
noir-framework/
|│
|├── datasource/
|│   ├── raw/              ← ofac, hackers, richlist, mixers, etc.
|│   └── processed/         ← cleaned and feature datasets
|
|├── scripts/
|│   ├── fetch/            ← data_fetch_*.py
|│   └── process/          ← wallet_features.py, label_wallets.py
|
|├── models/              ← train_model.py, shap_analysis.py
|├── notebooks/           ← exploratory analysis notebooks
|├── docs/                ← README.md (this file)
|├── requirements.txt
|└── .gitignore
```

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/zandriel-abyss/noir-framework.git
cd noir-framework

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Add your API key in a .env file
ETHERSCAN_API_KEY=your_etherscan_api_key
```

## Usage Examples

```bash
# 1. Fetch Ethereum Rich List
python scripts/fetch/fetch_richlist.py

# 2. Fetch Fraudulent Transactions
python scripts/fetch/fetch_fraud_transactions.py

# 3. Fetch Normal Transactions
python scripts/fetch/fetch_normal_transactions.py

# 4. Fetch Mixer Transactions
python scripts/fetch/data_fetch_mixers.py
```

CSV outputs are saved under `datasource/processed/`.

## Data Sources & Context

- **OFAC Ethereum Addresses** – Sanctioned addresses flagged by the US Treasury.
- **North Korean Hacker Addresses** – Public lists of cybercrime-linked wallets.
- **Ethereum Rich List** – Top 1,000 ETH holders from Etherscan.
- **Mixer Addresses** – Wallets linked to Tornado Cash mixers.
- **Etherscan API** – Primary data collection source for historical and live tx.

## Data Insights Summary (Pre-Model)

Before entering modeling, key insights were extracted from wallet-level features (L0–L3):

### Feature Correlation (Heatmap)
- Flags like `burst_tx_ratio` and `burst_tx_flag` are tightly correlated.
- Combined anomaly tags (`combined_risk_tag`) correlate with `anomaly_iso`, `anomaly_dbscan`, and counterparty scores.

### Behavioral Distributions by Label
- **`burst_tx_ratio`**: Fraud wallets cluster near 1 (hyperactive bursty behavior).
- **`dormant_awaken_count`**: Higher in fraud wallets, indicating suspicious gaps in activity.
- **`num_fraud_counterparties`**: Indicates indirect exposure to known bad actors.
- **`total_transactions` & `wallet_age_days`**: Fraud wallets have wider and more skewed distributions.

### Observations
- Labels are imbalanced (fraud-heavy), but meaningful behavioral separation exists.
- Many suspicious wallets have low active days and mid-level transaction counts.
- Feature quality is high enough to proceed with hybrid ML modeling (unsupervised + supervised).

## Roadmap

-  Finalize unified labeled dataset (fraud, normal, suspicious)
-  Build L0–L3 feature pipeline
-  Visualize feature distributions and correlations
-  Train hybrid ML pipeline (unsupervised + supervised)
-  Add SHAP-based explainability layer
-  Test on Layer-2 data via The Graph or Flipside
-  Simulate real-time wallet scoring + alert hooks

## References

- Taher, S. S., et al. (2024). Advanced Fraud Detection in Blockchain Transactions.
- Farrugia, S., et al. (2020). Detection of Illicit Accounts over Ethereum.
- Ralli, R., et al. (2024). Ensemble Fraud Detection.
- Chen, B., et al. (2021). Bitcoin Theft Detection.
- Zhang, S., et al. (2023). Dynamic Feature Fusion.
- Song, K., et al. (2024). Money Laundering Subgraphs.
- Weber, M., et al. (2019). AML via Graph Convolutions.
- Lin, D., et al. (2023). Cross-chain Tracking.
- BIS (2023). Project Aurora & Hertha.
- KPMG & Chainalysis (2023). AML Partnership.
- Daian, P., et al. (2020). Flash Boys 2.0.
- Qin, K., et al. (2022). Blockchain Extractable Value.
- Chainalysis (2024). Lazarus Group Laundering Routes.

## Author

Jzacksline Soosaiya\
Repo: [https://github.com/zandriel-abyss/noir-framework](https://github.com/zandriel-abyss/noir-framework)

---

See [Progress Report](https://github.com/zandriel-abyss/noir-framework/blob/main/docs/progress_report.md) or [Model Evaluation Summary](https://github.com/zandriel-abyss/noir-framework/blob/main/docs/evaluation_summary.md) for internal research updates.
