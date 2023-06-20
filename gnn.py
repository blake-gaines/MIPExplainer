import torch
from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import Variable
import networkx as nx

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
# from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.nn import global_mean_pool

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
import os


class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="sage"):
        super(GNN, self).__init__()
        self.conv_type = conv_type
        if conv_type == "gcn":
            conv = GCNConv
        elif conv_type == "sage":
            conv = SAGEConv
        else:
            raise NotImplementedError
        torch.manual_seed(12345)
        self.conv1 = conv(in_channels, 32)
        self.conv2 = conv(32, 48)
        self.conv3 = conv(48, 64)
        self.lin1 = Linear(64, 32)
        self.lin2 = Linear(32, 32)
        self.lin3 = Linear(32, out_channels)
    
    def get_embedding_outputs(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        embedding = self.embed(x, edge_index, batch, edge_weight)
        output =  self.classify(embedding)
        return embedding, output
    
    def get_all_layer_outputs(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        outputs = []
        outputs.append(self.conv1(x, edge_index))
        outputs.append(outputs[-1].relu())
        outputs.append(self.conv2(outputs[-1], edge_index))
        outputs.append(outputs[-1].relu())
        outputs.append(self.conv3(outputs[-1], edge_index))
        outputs.append(global_mean_pool(outputs[-1], batch))
        # TODO: Add ReLU
        outputs.append(self.lin1(outputs[-1]))
        outputs.append(outputs[-1].relu())
        outputs.append(self.lin2(outputs[-1]))
        outputs.append(outputs[-1].relu())
        outputs.append(self.lin3(outputs[-1]))
        return outputs 

    def forward(self, x, edge_index, batch, edge_weight=None):
        if edge_weight is None: edge_weight = torch.ones(edge_index.shape[1])
        embedding = self.embed(x, edge_index, batch, edge_weight)
        output =  self.classify(embedding)
        return output

    def embed(self, x, edge_index, batch, edge_weight=None):
        if edge_weight is None: edge_weight = torch.ones(edge_index.shape[1])
        
        # 1. Obtain node embeddings 
        if self.conv_type == "gcn":
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weight)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_weight)
        elif self.conv_type == "sage":
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x
    
    def classify(self, x):
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        return x
    
def train(model, train_loader, optimizer, criterion):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(model, loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

if __name__ == "__main__":
    if not os.path.isdir("data"): os.mkdir("data")
    if not os.path.isdir("models"): os.mkdir("models")

    epochs = 1000
    num_inits = 5
    num_explanations = 3
    conv_type = "sage"

    load_model = False
    model_path = "models/MUTAG_model.pth"

    log_run = False

    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    atom_indices = {
        0: "C",
        1: "N",
        2: "O",
        3: "F",
        4: "I",
        5: "Cl",
        6: "Br",
    }

    edge_indices = {
        0: "aromatic",
        1: "single",
        2: "double",
        3: "triple",
    }

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')


    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    if not load_model:
        model = GNN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes, conv_type=conv_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        print(model)
        pbar = tqdm(range(1,epochs))
        for epoch in pbar:
            train(model, train_loader, optimizer, criterion)
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            pbar.set_postfix_str(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        torch.save(model, model_path)
    else:
        model = torch.load(model_path)
        test_acc = test(model, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")