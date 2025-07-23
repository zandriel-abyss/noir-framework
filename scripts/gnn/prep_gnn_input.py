import torch
import networkx as nx
from torch_geometric.data import Data
import pandas as pd
import pickle

NODE_FEATURES_FILE = "datasource/processed/features_final_all_layers.csv"

# Load graph
with open("output/gnn/graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Relabel nodes to ensure consecutive integer indices
G = nx.convert_node_labels_to_integers(G, label_attribute='wallet_address')

node_id_to_wallet = nx.get_node_attributes(G, "wallet_address")
index_to_wallet = [node_id_to_wallet[i] for i in range(len(G.nodes))]

df = pd.read_csv(NODE_FEATURES_FILE)

if 'wallet_address' not in df.columns or 'label' not in df.columns:
    raise ValueError("Expected 'wallet_address' and 'label' columns in dataset.")

df.set_index("wallet_address", inplace=True)
df = df.reindex(index_to_wallet)  # retain all nodes, even if some features are missing
missing_wallets = set(index_to_wallet) - set(df.dropna().index)
if missing_wallets:
    print(f"⚠️ Warning: {len(missing_wallets)} wallets missing features. Keeping them with NaNs.")

feature_cols = [
    "total_transactions", "wallet_age_days", "active_days", "burst_tx_ratio",
    "dormant_awaken_count", "failure_ratio", "num_fraud_counterparties",
    "num_suspicious_counterparties", "num_normal_counterparties",
    "anomaly_iso", "anomaly_dbscan"
]
feature_cols = [col for col in feature_cols if col in df.columns]
x = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float)

label_map = {"normal": 0, "fraud": 1, "suspicious": 2}
y = torch.tensor(df["label"].map(label_map).fillna(-1).astype(int).values, dtype=torch.long)

# Build edge index
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

# Create torch_geometric data object
data = Data(x=x, edge_index=edge_index, y=y)

# Save
torch.save(data, "output/gnn/gnn_data.pt")
print(" GNN input saved to: output/gnn/gnn_data.pt")

with open("output/gnn/index_to_wallet.pkl", "wb") as f:
    pickle.dump(index_to_wallet, f)
print("index_to_wallet mapping saved to output/gnn/index_to_wallet.pkl")