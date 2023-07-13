import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_max_pool, GCN, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import ExplainerDataset, BA2MotifDataset, BAMultiShapesDataset
from torch_geometric.data import DataLoader

# Load the graph classification dataset (e.g., MUTAG)
# dataset = TUDataset(root='data/TUDataset', name='MUTAG')
dataset = BA2MotifDataset(root="data/BA2Motif")
data = dataset[0]

# Create a GraphSAGE model
model = GraphSAGE(dataset.num_node_features, hidden_channels=128, out_channels=dataset.num_classes, num_layers=3, aggr="sum")

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Split the dataset into training and test sets
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop

for epoch in range(200):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # batch.x = (1/7)*torch.ones_like(batch.x)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        pooled = global_mean_pool(out, batch.batch)
        loss = criterion(pooled, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    print(f'Epoch: {epoch+1:03d}, Loss: {total_loss/len(dataset):.4f}')

# Evaluation on the test set
model.eval()
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
correct = 0
for batch in test_loader:
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
        pooled = global_max_pool(out, batch.batch)
        pred = pooled.argmax(dim=1)
        correct += int((pred == batch.y).sum())

test_acc = correct / len(dataset)
print(f'Test Accuracy: {test_acc:.4f}')