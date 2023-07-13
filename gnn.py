import torch
from torch_geometric.datasets import TUDataset
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from torch.nn import Linear, ModuleDict, ReLU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation, Aggregation
from torch_geometric.nn import global_mean_pool

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
import os

class GNN(torch.nn.Module):
    aggr_classes = {
        "mean": MeanAggregation,
        "sum": SumAggregation,
        "max": MaxAggregation,
    }
    def __init__(self, in_channels, out_channels, conv_features=[16, 8], lin_features=[], global_aggr="mean"):
        super(GNN, self).__init__()

        self.layers = ModuleDict()

        self.layers["Conv_0"] = SAGEConv(in_channels, conv_features[0])
        self.layers[f"Conv_0_Relu"] = ReLU()
        for i, shape in enumerate(zip(conv_features, conv_features[1:])):
            self.layers[f"Conv_{i+1}"] = SAGEConv(*shape)
            self.layers[f"Conv_{i+1}_Relu"] = ReLU()

        self.layers["Aggregation"] = self.aggr_classes[global_aggr]()

        if len(lin_features) > 0:
            self.layers["Lin_0"] = Linear(conv_features[-1], lin_features[0])
            self.layers[f"Lin_0_Relu"] = ReLU()
        if len(lin_features) > 1:
            for i, shape in enumerate(zip(lin_features, lin_features[1:])):
                self.layers[f"Lin_{i+1}"] = Linear(*shape)
                self.layers[f"Lin{i+1}Relu"] = ReLU()
        self.layers[f"Lin_Output"] = Linear(lin_features[-1] if len(lin_features)>0 else conv_features[-1], out_channels)
        
    def fix_data(self, data):
        if data.batch is None: data.batch = torch.zeros(data.x.shape[0], dtype=torch.int64)
        if data.edge_weight is None: data.edge_weight = torch.ones(data.edge_index.shape[1])
        return data
    
    def forwardXA(self, X, A):
        X = torch.Tensor(X).double()
        A = torch.Tensor(A)
        edge_index, edge_weight = dense_to_sparse(A)
        data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight)
        return self.forward(data)

    def forward(self, data):
        data = self.fix_data(data)
        x = data.x
        for layer in self.layers.values():
            if isinstance(layer, SAGEConv):
                x = layer(x, data.edge_index)
            elif isinstance(layer, Aggregation):
                x = layer(x, data.batch)
            else:
                x = layer(x)
        return x
    
    def get_all_layer_outputs(self, data):
        data = self.fix_data(data)
        outputs = [("Input", data.x)]
        for name, layer in self.layers.items():
            if isinstance(layer, SAGEConv):
                outputs.append((name, layer(outputs[-1][1], data.edge_index)))
            else:
                outputs.append((name, layer(outputs[-1][1])))
        return outputs

# class GNN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, conv_type="sage"):
#         super(GNN, self).__init__()
#         self.conv_type = conv_type
#         if conv_type == "gcn":
#             conv = GCNConv
#         elif conv_type == "sage":
#             conv = SAGEConv
#         else:
#             raise NotImplementedError
#         # self.conv1 = conv(in_channels, 32)
#         # self.conv2 = conv(32, 48)
#         # self.conv3 = conv(48, 64)
#         # self.lin1 = Linear(64, 32)
#         # self.lin2 = Linear(32, 32)
#         # self.lin3 = Linear(32, out_channels)
#         self.conv1 = conv(in_channels, 16)
#         self.conv2 = conv(16, 16)
#         self.conv3 = conv(16, 16)
#         self.lin1 = Linear(16, 16)
#         self.lin2 = Linear(16, 16)
#         self.lin3 = Linear(16, out_channels)
    
#     def get_embedding_outputs(self, data):
#         x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
#         embedding = self.embed(x, edge_index, batch, edge_weight)
#         output =  self.classify(embedding)
#         return embedding, output
    
#     def get_all_layer_outputs(self, data):
#         x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
#         outputs = []
#         outputs.append(self.conv1(x, edge_index))
#         outputs.append(outputs[-1].relu())
#         outputs.append(self.conv2(outputs[-1], edge_index))
#         outputs.append(outputs[-1].relu())
#         outputs.append(self.conv3(outputs[-1], edge_index))
#         outputs.append(global_mean_pool(outputs[-1], batch))
#         # TODO: Add ReLU
#         outputs.append(self.lin1(outputs[-1]))
#         outputs.append(outputs[-1].relu())
#         outputs.append(self.lin2(outputs[-1]))
#         outputs.append(outputs[-1].relu())
#         outputs.append(self.lin3(outputs[-1]))
#         return outputs 

#     def forward(self, x, edge_index, batch, edge_weight=None):
#         if edge_weight is None: edge_weight = torch.ones(edge_index.shape[1])
#         embedding = self.embed(x, edge_index, batch, edge_weight)
#         output =  self.classify(embedding)
#         return output

#     def embed(self, x, edge_index, batch, edge_weight=None):
#         if edge_weight is None: edge_weight = torch.ones(edge_index.shape[1])
        
#         # 1. Obtain node embeddings 
#         if self.conv_type == "gcn":
#             x = self.conv1(x, edge_index, edge_weight)
#             x = x.relu()
#             x = self.conv2(x, edge_index, edge_weight)
#             x = x.relu()
#             x = self.conv3(x, edge_index, edge_weight)
#         elif self.conv_type == "sage":
#             x = self.conv1(x, edge_index)
#             x = x.relu()
#             x = self.conv2(x, edge_index)
#             x = x.relu()
#             x = self.conv3(x, edge_index)

#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         return x
    
#     def classify(self, x):
#         # 3. Apply a final classifier
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin1(x)
#         x = x.relu()
#         x = self.lin2(x)
#         x = x.relu()
#         x = self.lin3(x)
#         return x
    
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        # out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        out = model(data) # Perform a single forward pass
        loss = criterion(out, data.y)  # Compute the loss.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += float(loss)
    return total_loss/len(train_loader)

@torch.no_grad()
def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        # out = model(data.x, data.edge_index, data.batch) 
        out = model(data) 
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

if __name__ == "__main__":
    torch.manual_seed(12345)

    if not os.path.isdir("data"): os.mkdir("data")
    if not os.path.isdir("models"): os.mkdir("models")

    epochs = 1000
    num_inits = 5
    num_explanations = 3
    conv_type = "sage"

    load_model = False
    model_path = "models/MUTAG_model.pth"

    log_run = False

    from torch_geometric.datasets import ExplainerDataset, BA2MotifDataset, BAMultiShapesDataset
    from torch_geometric.datasets.graph_generator import BAGraph
    from torch_geometric.datasets.motif_generator import HouseMotif
    from torch_geometric.datasets.motif_generator import CycleMotif

    dataset = TUDataset(root="data/TUDataset", name="MUTAG")
    # print(dataset[0].x)
    # dataset = BA2MotifDataset(root="data/BA2Motif")
    # print(dataset[0])
    # dataset = BAMultiShapesDataset(root="data/BAMultiShapes")
    # print(dataset[0].x)
    # sys.exit(0)

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, stratify=dataset.y, random_state=7)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    if not load_model:
        # model = GNN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes, conv_type=conv_type, aggr="sum")
        model = GNN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes, conv_features=[16, 16, 16], lin_features=[16, 16], global_aggr="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        print(model)
        pbar = tqdm(range(1,epochs+1))
        for epoch in pbar:
            avg_loss = train(model, train_loader, optimizer, criterion)
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            pbar.set_postfix_str(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Avg Loss: {avg_loss:.4f}')

        torch.save(model, model_path)
    else:
        model = torch.load(model_path)
        test_acc = test(model, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")