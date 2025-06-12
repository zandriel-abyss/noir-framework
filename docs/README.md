# Noir-Framework

A modular, Python-based framework for collecting, processing, and analyzing blockchain transaction data to support fraud detection and Anti-Money Laundering (AML) research. The project focuses on the Ethereum mainnet and Layer-2 ecosystems, creating a dataset that distinguishes between fraudulent activities and legitimate privacy-preserving behaviors.

---

## 🪄 What’s This Project?

The **Noir-Framework** is designed to:

✅ **Collect blockchain data**: Fetch Ethereum addresses and transaction histories for known fraudulent addresses (OFAC, North Korean hacker groups) and “normal” high-volume addresses (rich list).  
✅ **Feature engineering**: Generate wallet-level behavioral features that capture suspicious patterns (like dormant reawakening, circular fund flows, cross-chain hops).  
✅ **Model building**: Lay the foundation for a hybrid machine learning pipeline that combines anomaly detection and supervised classification.  
✅ **Explainability & Evaluation**: Integrate explainable AI (XAI) techniques (e.g., SHAP) and causal reasoning modules to produce transparent, regulator-friendly outputs.

This framework is a **data and feature engineering pipeline** feeding into the broader thesis research on **modular blockchain fraud detection and compliance analytics**.

---

## 🗂️ Directory Structure
noir-framework/
├── datasource/
│   ├── raw/                  # Raw input data (wallet addresses, scraped lists)
│   │   ├── ofac-eth-addresses.txt
│   │   ├── nk-hackers.txt
│   │   └── etherscan_richlist.txt
│   └── processed/            # Processed transaction CSVs
│       ├── fraud_transactions.csv
│       └── normal_transactions.csv
├── scripts/                  # Python scripts for data collection
│   ├── fetch_richlist.py
│   ├── fetch_fraud_transactions.py
│   └── fetch_normal_transactions.py
├── notebooks/                # Jupyter notebooks for exploratory analysis
│   └── exploratory.ipynb
├── .env                      # Environment variables (API keys)
├── requirements.txt          # Python dependencies
└── README.md                 # This file!

---

## ⚙️ How It Works

1. **Fetch Ethereum Rich List**  
   - Script: `scripts/fetch_richlist.py`  
   - Scrapes top 1,000 Ethereum wallet addresses from Etherscan.

2. **Fetch Fraudulent Transactions**  
   - Script: `scripts/fetch_fraud_transactions.py`  
   - Loads OFAC and North Korean hacker addresses from `datasource/raw/`, then fetches transaction histories via the Etherscan API.

3. **Fetch Normal Transactions**  
   - Script: `scripts/fetch_normal_transactions.py`  
   - Loads rich-list addresses (excluding flagged ones) and fetches transaction data for the top N normal wallets.

4. **Process & Save Data**  
   - All transactions are saved in CSV files (`datasource/processed/`) for downstream ML and analysis.

---

## 🔧 Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/noir-framework.git
   cd noir-framework

2.	**Create and activate a Python virtual environment**
    python3 -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    .venv\Scripts\activate     # Windows

3. **Install Python dependencies**
    pip install -r requirements.txt

4.	**Set environment variables**
    Create a .env file in the project root:
    ETHERSCAN_API_KEY=your_etherscan_api_key
    CHROMEDRIVER_PATH=/path/to/chromedriver  # if not on PATH

## 🔎 Data Sources & Context
	•	OFAC Ethereum Addresses – Sanctioned addresses flagged by the US Treasury.
	•	North Korean Hacker Addresses – Public lists of North Korean cybercrime groups (e.g., Lazarus Group).
	•	Ethereum Rich List – Top 1,000 ETH addresses, typically whales or institutional players.
	•	Etherscan API – Used to pull complete transaction histories.

## Usage Examples
    1.  Scrape Rich List
        python scripts/fetch_richlist.py
    2. Fetch Fraud Transactions
        python scripts/fetch_fraud_transactions.py
    3. Fetch Normal Transactions
        python scripts/fetch_normal_transactions.py

    Output CSVs will be saved in datasource/processed/.

🔬 Next Steps & Roadmap
	•	Build feature engineering pipeline (dormant periods, circular flows, etc.).
	•	Develop and evaluate ML models (hybrid unsupervised + supervised approach).
	•	Integrate SHAP-based explainability and reason codes.
	•	Extend to real-time Layer-2 data streams via The Graph or Dune APIs.
	•	Package as a modular, research-grade fraud detection tool.

⸻

📚 References
	•	Taher, S. S., et al. (2024). Advanced Fraud Detection in Blockchain Transactions. DOI
	•	Farrugia, S., et al. (2020). Detection of Illicit Accounts over Ethereum. DOI
	•	Ralli, R., et al. (2024). Ensemble Fraud Detection. DOI
	•	Chen, B., et al. (2021). Bitcoin Theft Detection. DOI
	•	Zhang, S., et al. (2023). Dynamic Feature Fusion. arXiv
	•	Song, K., et al. (2024). Money Laundering Subgraphs. DOI
	•	Weber, M., et al. (2019). AML via Graph Convolutions. arXiv
	•	Lin, D., et al. (2023). Cross-chain Tracking. DOI
	•	BIS (2023). Project Aurora & Hertha. Link
	•	KPMG & Chainalysis (2023). AML Partnership. Link
	•	Daian, P., et al. (2020). Flash Boys 2.0. DOI
	•	Qin, K., et al. (2022). Blockchain Extractable Value. DOI
	•	Chainalysis (2024). Lazarus Group Laundering Routes. Link
