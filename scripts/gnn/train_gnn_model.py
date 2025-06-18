

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the preprocessed graph
import torch

data = torch.load("output/gnn/gnn_data.pt")

# Filter out nodes with label -1 (unknown class)
valid_nodes = data.y != -1

# Remap node indices after filtering
old_to_new_index = {}
new_index = 0
valid_idx = valid_nodes.nonzero(as_tuple=False).view(-1)
for old_idx in valid_idx:
    old_to_new_index[int(old_idx)] = new_index
    new_index += 1

# Filter and remap edge_index
mask = valid_nodes[data.edge_index[0]] & valid_nodes[data.edge_index[1]]
edge_index_filtered = data.edge_index[:, mask]
edge_index_remapped = torch.zeros_like(edge_index_filtered)
for i in range(edge_index_filtered.size(1)):
    src = int(edge_index_filtered[0, i])
    dst = int(edge_index_filtered[1, i])
    edge_index_remapped[0, i] = old_to_new_index[src]
    edge_index_remapped[1, i] = old_to_new_index[dst]

data.edge_index = edge_index_remapped
data.x = data.x[valid_nodes]
data.y = data.y[valid_nodes]

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
# Weighted loss logic
class_sample_count = torch.bincount(data.y)
weight = 1. / class_sample_count.float()
class_weights = weight / weight.sum()  # normalize
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate():
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1)
        y_true = data.y[data.test_mask].cpu()
        y_pred = preds[data.test_mask].cpu()
        print(" Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        print(" Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

# Run training
print(" Training GCN model...")
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate model
evaluate()

# t-SNE visualization of node embeddings
from sklearn.manifold import TSNE

model.eval()
with torch.no_grad():
    embeddings = model(data).cpu().numpy()
    y = data.y.cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(embeddings)

plt.figure(figsize=(8,6))
plt.scatter(z[:, 0], z[:, 1], c=y, cmap='coolwarm', s=10)
plt.title("t-SNE Visualization of Node Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig("output/gnn/tsne_embeddings.png")
print("t-SNE plot saved to output/gnn/tsne_embeddings.png")