import os
import pandas as pd
import numpy as np

# Paths
EDGE_FILE = "output/gnn/graph_edgelist.csv"
NODE_FILE = "output/gnn/graph_node_features.csv"
OUT_DIR = "output/gnn_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Edge List Analysis ---
edges = pd.read_csv(EDGE_FILE)
edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce').fillna(0)

edge_summary = {
    "num_edges": len(edges),
    "num_unique_sources": edges['source'].nunique(),
    "num_unique_targets": edges['target'].nunique(),
    "min_weight": edges['weight'].min(),
    "max_weight": edges['weight'].max(),
    "mean_weight": edges['weight'].mean(),
    "median_weight": edges['weight'].median(),
    "num_zero_weight": (edges['weight'] == 0).sum(),
    "num_nonzero_weight": (edges['weight'] != 0).sum(),
}

# Degree distributions
src_degree = edges['source'].value_counts()
dst_degree = edges['target'].value_counts()
all_wallets = pd.Index(src_degree.index.tolist() + dst_degree.index.tolist()).unique()
degree_df = pd.DataFrame({'wallet_address': all_wallets.astype(str)})
degree_df['out_degree'] = degree_df['wallet_address'].map(src_degree).fillna(0)
degree_df['in_degree'] = degree_df['wallet_address'].map(dst_degree).fillna(0)
degree_df['degree'] = degree_df['out_degree'] + degree_df['in_degree']
degree_df.to_csv(f"{OUT_DIR}/node_degrees.csv", index=False)

# --- Node Features Analysis ---
nodes = pd.read_csv(NODE_FILE)

# Label distribution
if 'label' in nodes.columns:
    label_counts = nodes['label'].value_counts()
    label_counts.to_csv(f"{OUT_DIR}/label_distribution.csv")

# Feature statistics
feature_cols = [col for col in nodes.columns if col not in ['wallet_address', 'label', 'xai_reason_code']]
feature_stats = nodes[feature_cols].describe().T
feature_stats.to_csv(f"{OUT_DIR}/node_feature_stats.csv")

# Correlation with label (mean per class)
if 'label' in nodes.columns:
    feature_means_by_label = nodes.groupby('label')[feature_cols].mean()
    feature_means_by_label.to_csv(f"{OUT_DIR}/feature_means_by_label.csv")

# Anomaly/risk flag summary
flag_cols = [col for col in nodes.columns if col.endswith('_flag') or 'anomaly' in col or 'risk' in col]
if flag_cols and 'label' in nodes.columns:
    flag_summary = nodes.groupby('label')[flag_cols].mean()
    flag_summary.to_csv(f"{OUT_DIR}/flag_summary_by_label.csv")

# XAI reason code counts
if 'xai_reason_code' in nodes.columns and 'label' in nodes.columns:
    xai_counts = nodes.groupby('label')['xai_reason_code'].value_counts().unstack(fill_value=0)
    xai_counts.to_csv(f"{OUT_DIR}/xai_reason_code_counts.csv")

# --- Save edge summary as text ---
with open(f"{OUT_DIR}/edge_summary.txt", "w") as f:
    f.write("Edge List Summary:\n")
    for k, v in edge_summary.items():
        f.write(f"{k}: {v}\n")
    f.write("\nDegree Distribution (summary):\n")
    f.write(str(degree_df[['out_degree', 'in_degree']].describe()))
    f.write("\n")

# --- Save node summary as text ---
with open(f"{OUT_DIR}/node_summary.txt", "w") as f:
    if 'label' in nodes.columns:
        f.write("Node Label Distribution:\n")
        f.write(str(label_counts))
        f.write("\n\n")
    f.write("Feature Statistics (mean/std/min/max):\n")
    f.write(str(feature_stats[['mean', 'std', 'min', 'max']]))
    if flag_cols and 'label' in nodes.columns:
        f.write("\n\nFlag Summary by Label:\n")
        f.write(str(flag_summary))
    if 'xai_reason_code' in nodes.columns and 'label' in nodes.columns:
        f.write("\n\nXAI Reason Code Counts by Label:\n")
        f.write(str(xai_counts))

print(f"Analysis complete. Results saved to {OUT_DIR}/")

# --- Network Visualization (Largest Connected Component) ---
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Build the graph from the edge list
G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

if 'label' in nodes.columns:
    label_dict = pd.Series(nodes.label.values, index=nodes.wallet_address).to_dict()
    nx.set_node_attributes(G, label_dict, "label")

    # Focus on top N nodes by degree for a denser, more informative subgraph
    top_n = 200
    top_nodes = degree_df.sort_values('degree', ascending=False).head(top_n)['wallet_address'].tolist()
    subG = G.subgraph(top_nodes)

    # Assign colors by label
    label_color_map = {'normal': 'green', 'fraud': 'red', 'suspicious': 'orange'}
    node_colors = [label_color_map.get(subG.nodes[n].get('label', 'normal'), 'gray') for n in subG.nodes]

    # Use a force-directed layout for better visualization
    pos = nx.spring_layout(subG, seed=42, k=0.15)

    plt.figure(figsize=(14, 10))
    nx.draw(subG, pos, with_labels=False, node_color=node_colors, node_size=80, edge_color='gray', alpha=0.7)
    legend_handles = [
        mpatches.Patch(color='green', label='normal'),
        mpatches.Patch(color='red', label='fraud'),
        mpatches.Patch(color='orange', label='suspicious')
    ]
    plt.legend(handles=legend_handles)
    plt.title("Top 100 Most Connected Wallets (colored by label)")
    plt.savefig(f"{OUT_DIR}/network_top_connected.png", bbox_inches='tight')
    plt.close()
    print(f"Network visualization of top connected nodes saved to {OUT_DIR}/network_top_connected.png")
else:
    print("No 'label' column found in node features; skipping network visualization.")
