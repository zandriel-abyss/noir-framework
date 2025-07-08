# Feature Engineering and Analysis Report

## Overview

This report documents the process and rationale behind the feature engineering pipeline for wallet-level fraud detection, as well as the results of feature analysis and visualization. The goal is to build a rich, interpretable set of features that capture both basic activity and complex risk signals, and to ensure these features are high-quality and informative for downstream machine learning models.

---

## Feature Engineering Pipeline

### 1. build_features_l0_aggregate.py

**Purpose:**  
Computes basic, aggregate statistics for each wallet.  
These are "raw" features, providing a baseline for all further analysis.

**Why L0?**  
L0 features are foundational: simple counts, sums, and averages that describe overall wallet activity.  
They establish a baseline for what is "normal" or "typical" for a wallet.

**How are features computed?**  
Reads all transactions.  
For each wallet:
- Counts total transactions.
- Sums and averages transaction values.
- Calculates wallet age (time between first and last transaction).
- Counts unique counterparties (from and to).
- Counts "circular" transactions (where from and to are the same wallet).
- Counts "dormant awakenings" (gaps >30 days between transactions).
- Assigns a label (if available).

**Output:**  
A CSV with one row per wallet.

---

### 2. build_features_l1_behavior.py

**Purpose:**  
Extracts behavioral patterns from transaction histories.

**Why L1?**  
L1 features go beyond simple aggregates to capture patterns over time, such as bursts of activity, dormancy, and error rates.  
These features help identify unusual or risky behaviors that aren’t visible from raw counts alone.

**How are features computed?**  
Reads all transactions, sorts by wallet and time.  
For each wallet:
- Calculates time differences between transactions.
- Computes:
    - Burst transaction ratio (fraction of transactions occurring within 1 hour of the previous).
    - Dormant awaken count (number of times a wallet is inactive for >30 days).
    - Failure ratio (fraction of failed transactions).

**Output:**  
A CSV with one row per wallet.

---

### 3. build_features_l2_riskflags.py

**Purpose:**  
Flags wallets based on their interactions with known risky counterparties.

**Why L2?**  
L2 features use information about the network of wallets, not just individual behavior.  
They quantify risk by association: how many times a wallet interacts with "fraud", "suspicious", or "normal" wallets.

**How are features computed?**  
Reads all transactions and L1 features.  
For each wallet:
- Counts how many times it interacts with wallets labeled as "fraud", "suspicious", or "normal".
- Merges these counts with L1 features.

**Output:**  
A CSV with one row per wallet.

---

### 4. build_features_l3_metaai.py

**Purpose:**  
Applies machine learning to detect anomalies and cluster wallets.  
Combines multiple signals into a single risk tag.

**Why L3?**  
L3 features use advanced analytics (ML models) to find subtle, complex patterns that may indicate risk.  
They synthesize lower-level features into higher-level, more actionable insights.

**How are features computed?**  
Reads L2 features.  
Normalizes numeric features.  
Runs:
- Isolation Forest (anomaly detection).
- DBSCAN (density-based clustering, flags outliers).
Creates a "combined risk tag" if any of the following are true:
- Many fraud/suspicious counterparties.
- Detected as anomaly by ML models.

**Output:**  
A CSV with all features and risk tags.

---

### 5. build_features_l3_xai_tags.py

**Purpose:**  
Adds XAI tags to each wallet, making risk reasons human-readable.

**Why?**  
XAI tags help users understand why a wallet is flagged as risky.  
They translate complex model outputs and feature combinations into simple, interpretable reason codes.

**How are features computed?**  
Reads L3 MetaAI features.  
Sets boolean flags for:
- Burst transaction behavior.
- Dormant awakening.
- Counterparty fraud links.
- Anomaly detection.
- High failure rate.
Generates a reason code string (e.g., "burst_tx|fraud_link").  
Adds a final XAI flag (true if any risk reason is present).

**Output:**  
A CSV with all features and XAI tags.

---

## Feature Analysis and Visualization

After building the features, we performed a thorough analysis to ensure data quality and to understand which features are most informative.

### **Summary of Analysis Results**

- **No missing values** in any feature columns.
- **No low-variance features** (all features have enough variation to be useful).
- **Highly correlated features:** Only `anomaly_dbscan` and `anomaly_flag` are almost perfectly correlated, suggesting one can be dropped to avoid redundancy.

### **Insights from Visualizations**

#### **1. Correlation Heatmap**
- The heatmap shows how features relate to each other.
- Most features are not strongly correlated, indicating a diverse set of predictors.
- The only strong correlation is between `anomaly_dbscan` and `anomaly_flag`, confirming redundancy.

#### **2. KDE Plots by Label**
- These plots show the distribution of key features for each class (fraud, normal, suspicious).
- Features like `total_transactions`, `burst_tx_ratio`, and `num_fraud_counterparties` show clear separation between classes, making them valuable for classification.
- Some features have overlapping distributions, indicating they may be less useful for distinguishing classes.

---

## **Conclusion**

- The feature engineering pipeline produces a comprehensive, multi-level set of features for each wallet, capturing both basic activity and complex risk signals.
- Feature analysis confirms the data is clean, non-redundant (except for one pair), and contains features that are informative for fraud detection.
- Visualizations support these findings and help guide feature selection for modeling.
- The final feature set is well-prepared for use in machine learning models, including GNNs and classical classifiers.

---