import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

# Load updated labels from merged feature file
label_df = pd.read_csv("datasource/processed/wallet_node_labels_merged.csv")
wallet_to_label = dict(zip(label_df["wallet_address"], label_df["label_y"]))

# Load the preprocessed graph
data = torch.load("output/gnn/gnn_data.pt", weights_only=False)

# Replace data.y using the updated labels
with open("output/gnn/index_to_wallet.pkl", "rb") as f:
    index_to_wallet = pickle.load(f)

if isinstance(index_to_wallet, list):
    index_to_wallet = {i: w for i, w in enumerate(index_to_wallet)}

if not isinstance(index_to_wallet, dict):
    raise ValueError("index_to_wallet must be a dictionary")

new_labels = []
for idx in range(len(index_to_wallet)):
    wallet = index_to_wallet.get(idx)
    label = wallet_to_label.get(wallet, -1)
    if isinstance(label, str):
        label = {"normal": 0, "fraud": 1,"suspicious": 2}.get(label.lower(), -1)
    new_labels.append(label)

unmatched = [index_to_wallet.get(i) for i, label in enumerate(new_labels) if label == -1]
print(f"⚠️ Unmatched wallets: {len(unmatched)} (will be skipped)")

data.y = torch.tensor(new_labels, dtype=torch.long)

print(f"Initial node feature shape: {data.x.shape}")
print(f"Initial edge index shape: {data.edge_index.shape}")
# Filter out nodes with label -1 (unknown class)
valid_nodes = data.y != -1

# Check if we have any valid nodes
if not valid_nodes.any():
    print("Error: No valid nodes found (all labels are -1)")
    exit()

print(f"Total nodes: {len(data.y)}")
print(f"Valid nodes: {valid_nodes.sum().item()}")
print(f"Label distribution: {torch.bincount(data.y[valid_nodes])}")

# 1. Get mapping from old indices to new indices
valid_idx = valid_nodes.nonzero(as_tuple=False).view(-1).cpu().numpy()
old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_idx)}

# 2. Filter edges: keep only edges where both nodes are valid
edge_index_np = data.edge_index.cpu().numpy()
src, dst = edge_index_np
src_valid = np.isin(src, valid_idx)
dst_valid = np.isin(dst, valid_idx)
edge_mask = src_valid & dst_valid

filtered_src = src[edge_mask]
filtered_dst = dst[edge_mask]

 # 3. Remap edge indices to new indices
remapped_src = np.array([old_to_new_index.get(s, -1) for s in filtered_src])
remapped_dst = np.array([old_to_new_index.get(d, -1) for d in filtered_dst])
mask_valid_remap = (remapped_src != -1) & (remapped_dst != -1)
remapped_src = remapped_src[mask_valid_remap]
remapped_dst = remapped_dst[mask_valid_remap]
edge_index_remapped = torch.tensor([remapped_src, remapped_dst], dtype=torch.long)

# 4. Filter node features and labels
data.x = data.x[valid_nodes]
print(f"Filtered node feature shape: {data.x.shape}")
data.y = data.y[valid_nodes]
data.edge_index = edge_index_remapped

print(f"After filtering - Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Train/test split (e.g., 80/20)
num_nodes = data.num_nodes
num_train = int(num_nodes * 0.8)
perm = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:num_train]] = True
test_mask[perm[num_train:]] = True
data.train_mask = train_mask
data.test_mask = test_mask

print(f"Train nodes: {train_mask.sum().item()}, Test nodes: {test_mask.sum().item()}")

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(data.num_node_features, 32, 3).to(device)  # 3 classes: fraud, normal, suspicious
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Weighted loss logic - FIXED
num_classes = 3  # fraud, normal, suspicious
class_sample_count = torch.bincount(data.y, minlength=num_classes)
print(f"Class counts: {class_sample_count.tolist()}")

 # Use uniform weights if any class is missing
if (class_sample_count == 0).any():
    missing_classes = torch.where(class_sample_count == 0)[0].tolist()
    print(f"Warning: Missing class(es) in data: {missing_classes}. Using uniform weights.")
    class_weights = torch.ones(num_classes) / num_classes
else:
    weight = 1. / (class_sample_count.float() + 1e-8)
    class_weights = weight / weight.sum()

print(f"Class weights: {class_weights.tolist()}")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Check for NaN loss
    if torch.isnan(loss):
        print("Warning: NaN loss detected!")
        return float('nan')
    
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function - FIXED
def evaluate():
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1)
        y_true = data.y[data.test_mask].cpu()
        y_pred = preds[data.test_mask].cpu()
        
        # Check if test set is empty
        if len(y_true) == 0:
            print("Warning: Test set is empty!")
            return
        
        # Check if we have any valid predictions
        if len(y_pred) == 0:
            print("Warning: No predictions generated!")
            return
        
        print(" Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        print(" Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

# Run training
print(" Training GCN model...")
nan_count = 0
for epoch in range(1, 101):
    loss = train()
    if torch.isnan(torch.tensor(loss)):
        nan_count += 1
        if nan_count > 5:
            print("Too many NaN losses. Stopping training.")
            break
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate model
evaluate()

# t-SNE visualization of node embeddings
from sklearn.manifold import TSNE

def get_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        x = model.conv1(x, edge_index)
        x = F.relu(x)
        return x.cpu().numpy()

embeddings = get_embeddings(model, data)
y = data.y.cpu().numpy()


from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# 3D t-SNE
tsne = TSNE(n_components=3, random_state=42)
z = tsne.fit_transform(embeddings)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Manually define colors per class index
color_map = {0: 'blue', 1: 'red', 2: 'grey'}  # 0=normal, 1=fraud, 2=suspicious
colors = [color_map[label] for label in y]

scatter = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=colors, s=20)
ax.set_title("3D t-SNE Visualization of Node Embeddings")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

# Add manual legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='normal', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='fraud', markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='suspicious', markerfacecolor='grey', markersize=8),
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("output/gnn/tsne_embeddings_3d.png")
print("3D t-SNE plot saved to output/gnn/tsne_embeddings_3d.png")

# Optional Interactive Plotly 3D t-SNE Visualization
try:
    import plotly.express as px
    import plotly.io as pio

    label_names = {0: 'normal', 1: 'fraud', 2: 'suspicious'}
    color_map_plotly = {0: 'blue', 1: 'red', 2: 'grey'}

    plot_df = pd.DataFrame({
        'Component 1': z[:, 0],
        'Component 2': z[:, 1],
        'Component 3': z[:, 2],
        'Label': [label_names[label] for label in y],
        'Color': [color_map_plotly[label] for label in y]
    })

    fig_plotly = px.scatter_3d(
        plot_df,
        x='Component 1',
        y='Component 2',
        z='Component 3',
        color='Label',
        color_discrete_map={
            'normal': 'blue',
            'fraud': 'red',
            'suspicious': 'grey'
        },
        title='Interactive 3D t-SNE Embedding Visualization'
    )

    fig_plotly.write_html("output/gnn/tsne_embeddings_3d_interactive.html")
    print("✅ Interactive 3D plot saved to output/gnn/tsne_embeddings_3d_interactive.html")
except ImportError:
    print("⚠️ Plotly not installed. Skipping interactive plot.")

# After evaluation, get predictions for test set
model.eval()
with torch.no_grad():
    logits = model(data)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = data.y.cpu().numpy()

# Get indices of test nodes
test_indices = data.test_mask.cpu().numpy().nonzero()[0]

# Check if we have test indices
if len(test_indices) == 0:
    print("Warning: No test indices found!")
else:
    # You need the wallet address mapping for these indices
    with open("output/gnn/index_to_wallet.pkl", "rb") as f:
        index_to_wallet = pickle.load(f)

    if max(test_indices) >= len(index_to_wallet):
        print("⚠️ Warning: Wallet mapping does not align with test indices!")

    safe_indices = [i for i in test_indices if i < len(index_to_wallet)]
    df = pd.DataFrame({
        "wallet_address": [index_to_wallet[i] for i in safe_indices],
        "true_label": y_true[safe_indices],
        "gnn_pred": preds[safe_indices]
    })
    df.to_csv("output/gnn/gnn_predictions.csv", index=False)
    print("GNN predictions saved to output/gnn/gnn_predictions.csv")