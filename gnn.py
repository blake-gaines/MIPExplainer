import torch
from torch_geometric.utils import dense_to_sparse

from torch.nn import Linear, ModuleDict, ReLU
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.nn.aggr import (
    SumAggregation,
    MeanAggregation,
    MaxAggregation,
    Aggregation,
)
from tqdm import tqdm
import os
import numpy as np
import pickle  # noqa: F401
from torch_geometric.datasets import TUDataset  # noqa: F401
from datasets import get_dataset


def prune_weights_below_threshold(module, threshold):
    for name, param in module.named_parameters():
        param.data[param.abs() < threshold] = 0.0


class GNN(torch.nn.Module):
    aggr_classes = {
        "mean": MeanAggregation,
        "sum": SumAggregation,
        "max": MaxAggregation,
    }

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_features,
        lin_features,
        global_aggr="mean",
        conv_aggr="mean",
        device="cpu",
    ):
        super(GNN, self).__init__()

        self.device = device
        self.to(device)

        self.layers = ModuleDict()

        # Add convolutional layers
        self.layers["Conv_0"] = SAGEConv(in_channels, conv_features[0], aggr=conv_aggr)
        self.layers["Conv_0_Relu"] = ReLU()
        for i, shape in enumerate(zip(conv_features, conv_features[1:])):
            self.layers[f"Conv_{i + 1}"] = SAGEConv(*shape, aggr=conv_aggr)
            self.layers[f"Conv_{i + 1}_Relu"] = ReLU()

        # Add Global Pooling Layer
        self.layers["Aggregation"] = self.aggr_classes[global_aggr]()

        # Add FC Layers
        if len(lin_features) > 0:
            self.layers["Lin_0"] = Linear(conv_features[-1], lin_features[0])
            self.layers["Lin_0_Relu"] = ReLU()
        if len(lin_features) > 1:
            for i, shape in enumerate(zip(lin_features, lin_features[1:])):
                self.layers[f"Lin_{i + 1}"] = Linear(*shape)
                self.layers[f"Lin_{i + 1}_Relu"] = ReLU()
        self.layers["Output"] = Linear(
            lin_features[-1] if len(lin_features) > 0 else conv_features[-1],
            out_channels,
        )

    def fix_data(self, data):
        # If the data does not have any batches, assign all the nodes to the same batch
        if data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.int64)
        # If there are no edge weights, assign weight 1 to all edges
        if data.edge_weight is None:
            data.edge_weight = torch.ones(data.edge_index.shape[1])
        data.x = data.x.to(next(self.parameters()).dtype)
        return data

    def forwardXA(self, X, A):
        # Same as forward, but takes node features and adjacency matrix instead of a Data object
        X = torch.Tensor(X)
        A = torch.Tensor(A)
        edge_index, edge_weight = dense_to_sparse(A.to(next(self.parameters()).device))
        data = Data(
            x=X.to(next(self.parameters()).device, next(self.parameters()).dtype),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return self.forward(data)

    def forward(self, data):
        data = self.fix_data(data)
        x = data.x.to(next(self.parameters()).device, next(self.parameters()).dtype)
        edge_index = data.edge_index.to(next(self.parameters()).device)
        for layer in self.layers.values():
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            elif isinstance(layer, Aggregation):
                x = layer(x, data.batch.to(next(self.parameters()).device))
            else:
                x = layer(x)
        return x.cpu()

    def get_all_layer_outputs(self, data):
        data = self.fix_data(data)
        outputs = [("Input", data.x.to(next(self.parameters()).device, next(self.parameters()).dtype))]
        edge_index = data.edge_index.to(next(self.parameters()).device)
        for name, layer in self.layers.items():
            if isinstance(layer, SAGEConv):
                outputs.append((name, layer(outputs[-1][1], edge_index)))
            elif isinstance(layer, Aggregation):
                outputs.append((name, layer(outputs[-1][1], data.batch.to(next(self.parameters()).device))))
            else:
                outputs.append((name, layer(outputs[-1][1])))
        return [(k, v.cpu()) for k, v in outputs]

    def get_layer_output(self, data, layer_name):
        data = self.fix_data(data)
        x = data.x.to(next(self.parameters()).device, next(self.parameters()).dtype)
        edge_index = data.edge_index.to(next(self.parameters()).device)
        if layer_name not in self.layers:
            raise ValueError(f"Network has no layer with name {layer_name}")
        for name, layer in self.layers.items():
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            elif isinstance(layer, Aggregation):
                x = layer(x, data.batch.to(next(self.parameters()).device))
            else:
                x = layer(x)
            if name == layer_name:
                return x.cpu()


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset
        out = model(data)  # Perform a single forward pass
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += float(loss)
    return total_loss / len(train_loader)  # Return average batch loss


@torch.no_grad()
def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def smallest_largest_weight(model):
    smallest = float("inf")  # Initialize with positive infinity as a placeholder
    largest = -float("inf")
    for param in model.parameters():
        # if param.requires_grad:
        param_min = torch.min((param[param != 0]).abs())  # Find the smallest non-zero weight in the parameter
        if param_min < smallest:
            smallest = param_min.item()
        param_max = torch.max((param[param != 0]).abs())  # Find the largest weight in the parameter
        if param_max > largest:
            largest = param_max.item()
    return (smallest if smallest != float("inf") else None), (largest if largest != -float("inf") else None)


if __name__ == "__main__":
    torch.manual_seed(12345)

    if not os.path.isdir("models"):
        os.mkdir("models")

    epochs = 200
    conv_type = "sage"
    global_aggr = "mean"
    conv_aggr = "sum"
    prune_threshold = 1e-5

    load_model = False

    dataset_name = "REDDIT-BINARY"

    model_path = f"models/{dataset_name}_model.pth"

    log_run = False

    dataset = get_dataset(dataset_name)
    dataset.split()
    train_dataset, test_dataset = dataset.train_data, dataset.test_data
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()

    print()
    print(f"Dataset: {str(dataset)[:20]}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")

    ys = [int(d.y) for d in dataset]
    print("YS", ys[:10])
    num_classes = len(set(ys))
    num_node_features = dataset[0].x.shape[1]

    print(f"Number of training graphs: {len(train_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print()
    if not load_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Training on", device)
        model = GNN(
            in_channels=num_node_features,
            out_channels=num_classes,
            # conv_features=[16, 16],
            # lin_features=[8],
            conv_features=[8, 8, 8, 8],
            lin_features=[8, 8],
            # conv_features=[64, 32],
            # lin_features=[16, 8],
            global_aggr=global_aggr,
            conv_aggr=conv_aggr,
            device=device,
        )
        model.to(device, torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # , weight_decay=1e-4)

        criterion = torch.nn.CrossEntropyLoss()
        print(model)
        print("Model Parameters:", sum(param.numel() for param in model.parameters()))
        pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:
            avg_loss = train(model, train_loader, optimizer, criterion)
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            sw, lw = smallest_largest_weight(model)
            pbar.set_postfix_str(
                f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Avg Loss: {avg_loss:.4f}, Log SW: {np.log(sw) / np.log(10):.2f}, Log LW: {np.log(lw) / np.log(10):.2f}"
            )
            torch.save(model, model_path)

        prune_weights_below_threshold(model, prune_threshold)
        test_acc = test(model, test_loader)
        print("Pruned Test Accuracy:", test_acc)
        sw, lw = smallest_largest_weight(model)
        print(f"Log SW: {np.log(sw) / np.log(10):.2f}, Log LW: {np.log(lw) / np.log(10):.2f}")
        torch.save(model, model_path)
    else:
        model = torch.load(model_path)
        test_acc = test(model, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")
