# Modified from https://github.com/yolandalalala/GNNInterpreter/blob/main/gnninterpreter/datasets/base_graph_dataset.py

import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataLoader
import networkx as nx
import pickle
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import random
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset, ABC):
    name = "ABC"
    root = "data"
    node_feature_type = "constant"

    def __init__(self, *, dtype=torch.float32, seed=7):
        self.dtype = dtype
        self.seed = seed
        self._seed_all(seed)
        self.data = self.get_data()

        self.ys = [int(d.y) for d in self.data]
        self.num_classes = len(set(self.ys))

    @abstractmethod
    def get_data(self, data_dir):
        raise NotImplementedError

    def read_pickle(self, data_dir):
        with open(data_dir, "rb") as f:
            self.data = pickle.load(f)

    def _seed_all(self, seed):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def split(self, test_size=0.2, val_size=None):
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=test_size, random_state=self.seed
        )
        if val_size is not None:
            self.test_data, self.val_data = train_test_split(
                self.test_data, test_size=val_size, random_state=self.seed
            )

    def loader(self, *args, **kwargs):
        return torch.data.DataLoader(self, *args, **kwargs)

    def get_train_loader(self, *args, **kwargs):
        if not hasattr(self, "train_data"):
            self.split()
        return self.loader(dataset=self.train_data, *args, **kwargs)

    def get_test_loader(self, *args, **kwargs):
        if not hasattr(self, "test_data"):
            self.split()
        return self.loader(dataset=self.test_data, *args, **kwargs)

    def show(self, idx, ax=None, **kwargs):
        data = self[idx]
        print(f"data: {data}")
        print(f"class: {self.GRAPH_CLS[data.G.graph['label']]}")
        self.draw(data.G, ax=ax, **kwargs)

    @torch.no_grad()
    def evaluate_model(self, model, batch_size=32):
        model.eval()

    def get_average_phi(self, nn, layer_name):
        embedding_sum = None
        num_classes = self.num_classes
        n_instances = torch.zeros(num_classes)
        for data in self:
            # data.x = data.x.double()
            embeddings = nn.get_layer_output(data, layer_name)
            if embedding_sum is None:
                embedding_sum = torch.zeros(num_classes, embeddings.shape[-1])
            embedding_sum[data.y] += torch.sum(embeddings, dim=0)
            n_instances[data.y] += 1
        return (embedding_sum / torch.unsqueeze(n_instances, 1)).detach().numpy()

    def get_random_graph(self, label=None, num_nodes=None):
        choices = self.data
        if label is not None:
            choices = [d for d in choices if d.y == label]
        if num_nodes is not None:
            choices = [d for d in choices if d.num_nodes == num_nodes]
        if not choices:
            raise ValueError("No graphs found with the given parameters.")
        return random.choice(choices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif type(key) == tuple and len(key) == 2:
            pass

    def __iter__(self):
        return (d for d in self.data)


class GraphDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_node_features = self.data[0].num_node_features
        self.num_edge_features = self.data[0].num_edge_features

    def loader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)

    def nx_to_pyg(self, G):
        raise NotImplementedError

    def pyg_to_nx(self, data):
        raise NotImplementedError

    def draw_graph(
        self,
        A=None,
        X=None,
        edge_index=None,
        data=None,
        nx_graph=None,
        directed=False,
        with_labels=True,
        fig=None,
        ax=None,
        label_key="label",
        **kwargs,
    ):
        if isinstance(A, torch.Tensor):
            A = A.detach().numpy()
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()

        if A is not None:
            G = nx.from_numpy_array(
                A, create_using=nx.DiGraph if directed else nx.Graph
            )
        elif edge_index is not None:
            G = nx.from_edgelist(
                edge_index, create_using=nx.DiGraph if directed else nx.Graph
            )
        elif data is not None:
            G = to_networkx(data, to_undirected=(not directed))
            if hasattr(data, "x"):
                X = data.x.detach().numpy()
        elif nx_graph is not None:
            G = nx_graph
        else:
            raise ValueError(
                "Supply adjacency matrix or edge list or nx graph for visualization."
            )

        pos = nx.spring_layout(G, seed=7)
        labels, node_color = None, None

        if X is not None:
            if X.shape[-1] == 1:
                x_indices = X.squeeze()
            else:
                x_indices = np.argmax(X, axis=1)

            if hasattr(self, "NODE_CLS"):
                labels = dict(zip(range(X.shape[0]), map(self.NODE_CLS.get, x_indices)))
            else:
                labels = None

            if hasattr(self, "NODE_COLOR"):
                node_color = list(
                    map(lambda i: self.NODE_COLOR.get(i, "skyblue"), x_indices)
                )
            else:
                node_color = None
        if fig is None:
            fig = plt.figure(frameon=False)
        if ax is None:
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=with_labels,
            labels=labels,
            node_color=node_color,
            ax=ax,
            node_size=800,
            font_size=18,
            **kwargs,
        )
        return fig, ax

    def random_connected_adj(self, num_nodes):
        # TODO: Make this reasonable
        adj = torch.randint(0, 2, (num_nodes, num_nodes))
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.T
        adj = torch.clip(adj, 0, 1)
        adj = adj.numpy()
        adj = np.clip(adj + np.eye(num_nodes, k=1), a_min=0, a_max=1)
        adj = np.clip(adj + np.eye(num_nodes, k=-1), a_min=0, a_max=1)
        adj = torch.Tensor(adj)
        return adj

    def dummy_graph(self, num_nodes):
        adj = self.random_connected_adj(num_nodes)
        if self.node_feature_type == "constant":
            x = torch.ones((num_nodes, self.num_node_features))
        elif self.node_feature_type == "one-hot":
            x = torch.eye(self.num_node_features)[
                torch.randint(self.num_node_features, (num_nodes,))
            ]
        elif self.node_feature_type == "degree":
            x = torch.unsqueeze(torch.sum(adj, dim=-1), dim=-1)
        return Data(x=x, edge_index=dense_to_sparse(adj)[0])
