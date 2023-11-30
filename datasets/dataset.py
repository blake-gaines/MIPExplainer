# Modified from https://github.com/yolandalalala/GNNInterpreter/blob/main/gnninterpreter/datasets/base_graph_dataset.py

import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_undirected, to_networkx
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class Dataset(torch.utils.data.Dataset, ABC):
    name = "ABC"
    root = "data"

    def __init__(self, *,
                 dtype=torch.float32,
                 seed=None):

        self.dtype = dtype
        self._seed_all(seed)
        self.data = self.get_data()

        self.ys = [int(d.y) for d in self.data]
        self.num_classes = len(set(self.ys))

    @abstractmethod
    def get_data(self, data_dir):
        raise NotImplementedError
    
    def read_pickle(self, data_dir):
        with open(data_dir, "rb") as f: self.data = pickle.load(f)

    def _seed_all(self, seed):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def loader(self, *args, **kwargs):
        return torch.data.DataLoader(self, *args, **kwargs)

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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if type(key) == int:
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
        return DataLoader(self, *args, **kwargs)

    def nx_to_pyg(self, G):
        raise NotImplementedError

    def pyg_to_nx(self, data):
        raise NotImplementedError

    def draw_graph(self, A=None, X=None, edge_index=None, data=None, label_dict=None, color_dict=None, directed=False, with_labels=False, **kwargs):
        if isinstance(A, torch.Tensor): A = A.detach().numpy()
        if isinstance(X, torch.Tensor): X = X.detach().numpy()

        if A is not None:
            G = nx.from_numpy_array(A, create_using=nx.DiGraph if directed else nx.Graph)
        elif edge_index is not None:
            G = nx.from_edgelist(edge_index, create_using=nx.DiGraph if directed else nx.Graph)
        elif data is not None:
            G = to_networkx(data, to_undirected=(not directed))
            if hasattr(data, "x"):
                X = data.x.detach().numpy()
        else:
            raise ValueError("Supply adjacency matrix or edge list for visualization.")

        pos = nx.spring_layout(G, seed=7)
        labels, node_color = None, None

        if X is not None:
            if X.shape[-1] == 1:
                x_indices = X.squeeze()
            else:
                x_indices = np.argmax(X, axis=1)
            labels = dict(zip(range(X.shape[0]), map(label_dict.get, x_indices))) if label_dict else dict(zip(range(X.shape[0]), x_indices))

            if color_dict is None:
                color_dict = {i:c for i,c in enumerate(random.choices(list(mcolors.CSS4_COLORS.values()), k=X.shape[1]))}
            node_color = list(map(lambda i: color_dict.get(i, "skyblue"), x_indices))

        fig, ax = plt.subplots()
        nx.draw_networkx(G, pos=pos, with_labels=with_labels, labels=labels, node_color=node_color, **kwargs)
        return fig, ax
