

import torch
import networkx as nx
from torch_geometric.data import Data
import pandas as pd

# Load graph
import pickle
with open("output/gnn/graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Relabel nodes to ensure consecutive integer indices
G = nx.convert_node_labels_to_integers(G, label_attribute='wallet_address')

# Build edge index
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

# Build node features (from stored attributes)
node_attrs = []
for _, attr in G.nodes(data=True):
    features = [
        attr.get("total_transactions", 0),
        attr.get("total_value", 0.0),
        attr.get("wallet_age_days", 0),
        attr.get("active_days", 0),
        attr.get("burst_tx_ratio", 0.0),
        attr.get("dormant_awaken_count", 0),
        attr.get("failure_ratio", 0.0),
        attr.get("num_fraud_counterparties", 0),
        attr.get("num_suspicious_counterparties", 0),
        attr.get("num_normal_counterparties", 0),
        attr.get("anomaly_iso", 0),
        attr.get("anomaly_dbscan", 0),
    ]
    node_attrs.append(features)
x = torch.tensor(node_attrs, dtype=torch.float)

# Build labels (fraud: 1, suspicious: 2, normal: 0, unknown: -1)
label_map = {"normal": 0, "fraud": 1, "suspicious": 2}
y = torch.tensor([
    label_map.get(attr.get("label", "unknown"), -1)
    for _, attr in G.nodes(data=True)
], dtype=torch.long)

# Create torch_geometric data object
data = Data(x=x, edge_index=edge_index, y=y)

# Save
torch.save(data, "output/gnn/gnn_data.pt")
print(" GNN input saved to: output/gnn/gnn_data.pt")