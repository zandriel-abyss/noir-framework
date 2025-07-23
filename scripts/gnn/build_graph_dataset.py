# scripts/gnn/build_graph_dataset.py

import pandas as pd
import networkx as nx
from pathlib import Path

# File paths
TX_FILE = Path("datasource/raw/all_transactions_labeled.csv")
WALLET_FEATURES_FILE = Path("datasource/processed/features_for_training.csv")
EDGE_LIST_FILE = Path("output/gnn/graph_edgelist.csv")
NODE_FEATURES_FILE = Path("output/gnn/graph_node_features.csv")
GRAPH_PICKLE_FILE = Path("output/gnn/graph.gpickle")

# Create output dir
EDGE_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load data
df_tx = pd.read_csv(TX_FILE, low_memory=False)
df_nodes = pd.read_csv(WALLET_FEATURES_FILE)

# Basic cleanup
df_tx = df_tx[['from', 'to', 'value']]
df_tx = df_tx.dropna()
df_tx['from'] = df_tx['from'].str.lower()
df_tx['to'] = df_tx['to'].str.lower()

# Generate edge list with weights (sum of ETH transferred between wallets)
edge_weights = df_tx.groupby(['from', 'to'])['value'].sum().reset_index()
edge_weights.columns = ['source', 'target', 'weight']
edge_weights['weight'] = pd.to_numeric(edge_weights['weight'], errors='coerce')
edge_weights['weight'] = edge_weights['weight'].fillna(0)
edge_weights['weight'] = edge_weights['weight'].clip(upper=1e6)
edge_weights.to_csv(EDGE_LIST_FILE, index=False)
print("Clipped edge weights above 1e6 ETH")
print(f" Edge list saved to {EDGE_LIST_FILE}")

# Normalize wallet addresses in node features
df_nodes['wallet_address'] = df_nodes['wallet_address'].str.lower()
# Remove non-standard wallet addresses (e.g., 'genesis' or malformed addresses)
df_nodes = df_nodes[df_nodes['wallet_address'].str.match(r'^0x[a-fA-F0-9]{40}$')]

df_nodes = df_nodes.drop_duplicates(subset=['wallet_address'])

# Ensure only nodes present in the edge list are included
valid_addresses = set(edge_weights['source']).union(set(edge_weights['target']))
df_nodes = df_nodes[df_nodes['wallet_address'].isin(valid_addresses)]

# Drop node features with zero variance
nunique = df_nodes.nunique()
zero_var_cols = nunique[nunique <= 1].index.tolist()
df_nodes.drop(columns=zero_var_cols, inplace=True)
print(f"Dropped zero-variance columns: {zero_var_cols}")

df_nodes.to_csv(NODE_FEATURES_FILE, index=False)
print(f" Node features saved to {NODE_FEATURES_FILE}")

# Construct graph using NetworkX
G = nx.from_pandas_edgelist(edge_weights, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

# Attach node attributes
node_feature_dict = df_nodes.set_index('wallet_address').to_dict(orient='index')
nx.set_node_attributes(G, node_feature_dict)

# Save graph as pickle
import pickle
with open(GRAPH_PICKLE_FILE, 'wb') as f:
    pickle.dump(G, f)
print(f" Graph saved to {GRAPH_PICKLE_FILE}")
