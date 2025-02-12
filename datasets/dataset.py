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


def isonleft(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0


def intersects(pos, edge1, edge2):
    return isonleft(pos[edge1[0]], pos[edge1[1]], pos[edge2[0]]) != isonleft(
        pos[edge1[0]], pos[edge1[1]], pos[edge2[1]]
    )


class Dataset(torch.utils.data.Dataset, ABC):
    name = "ABC"
    root = "data"
    node_feature_type = "constant"

    def __init__(self, *, dtype=torch.float32, seed=None):
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
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=self.seed)
        if val_size is not None:
            self.test_data, self.val_data = train_test_split(self.test_data, test_size=val_size, random_state=self.seed)

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
        elif isinstance(key, tuple) and len(key) == 2:
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
        node_color=None,
        edge_color=None,
        pos=None,
        match_graphs=True,
        **kwargs,
    ):
        component_colors = ["skyblue", "darkorange", "forestgreen"]
        labels, node_color = None, None
        if isinstance(A, torch.Tensor):
            A = A.detach().numpy()
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()

        if A is not None:
            # A = np.astype(int)
            G = nx.from_numpy_array(A, create_using=nx.DiGraph if directed else nx.Graph)

        elif edge_index is not None:
            G = nx.from_edgelist(edge_index, create_using=nx.DiGraph if directed else nx.Graph)
        elif data is not None:
            G = to_networkx(data, to_undirected=(not directed))
            if hasattr(data, "x"):
                X = data.x.detach().numpy()
        elif nx_graph is not None:
            G = nx_graph
        else:
            raise ValueError("Supply adjacency matrix or edge list or nx graph for visualization.")

        ccs = list(np.array(sorted(list(cc))) for cc in nx.connected_components(G))
        ccs.sort(key=lambda cc: cc[0])
        nccs = len(ccs)

        if nccs == 2 and len(ccs[0]) == len(ccs[1]):
            ## Make edge_color red for edges that are different between the two halves of the adjacency matrix
            N = A.shape[0] // 2
            c0, c1 = A[:N, :N], A[N:, N:]
            edges_diff = np.not_equal(c0, c1).astype(int)
            # edges_diff = ((A[:N, :N] - A[N:, N:]) != 0).astype(int)
            edge_color = []
            for edge in G.edges:
                if edges_diff[edge[0] % N, edge[1] % N]:
                    edge_color.append("red")
                else:
                    edge_color.append("black")

            ## Match the positions of the two graphs
            pos = nx.spring_layout(G.subgraph(ccs[1]), seed=7, iterations=100)
            init_pos = {i: p for i, p in zip(ccs[0], pos.values())}
            if match_graphs:
                ## Copy node positions
                pos = pos | init_pos
            else:
                ## Try to untangle graph without moving too many nodes
                new_edges = np.argwhere(np.tril(np.greater(c0, c1))).tolist()
                problem_edges = set()
                for new_edge in new_edges:
                    for existing_edge in G.subgraph(ccs[0]).edges:
                        if intersects(init_pos, new_edge, existing_edge):
                            problem_edges.add(tuple(new_edge))
                            break
                flattened_problems = [n for e in problem_edges for n in e]
                problem_nodes = sorted(
                    list(set(flattened_problems)), key=lambda node: flattened_problems.count(node), reverse=True
                )
                fixed = set(ccs[0])
                for node in problem_nodes:
                    fixed.remove(node)
                    problem_edges = {e for e in problem_edges if node not in e}
                    if not problem_edges:
                        break

                pos = pos | nx.spring_layout(
                    G.subgraph(ccs[0]),
                    seed=7,
                    iterations=100,
                    pos=init_pos,
                    fixed=list(fixed) if fixed else None,
                )

        fig = plt.figure(frameon=False, figsize=[6.4, 4.8 * nccs], dpi=150, layout="compressed")

        for j, cc in enumerate(ccs):
            subG = G.subgraph(cc)
            if pos is None:
                subpos = nx.spring_layout(
                    subG,
                    seed=7,
                    iterations=100,
                )
            else:
                subpos = pos

            if len(ccs) > 1:
                node_color = [component_colors[j]] * len(cc)
            if X is not None:
                x_indices = None
                if self.node_feature_type == "integer":
                    x_indices = X[cc].squeeze()
                elif self.node_feature_type == "one-hot":
                    x_indices = np.argmax(X[cc], axis=1)

                if labels is None and hasattr(self, "NODE_CLS") and x_indices is not None:
                    labels = dict(zip(range(X[cc].shape[0]), map(self.NODE_CLS.get, x_indices)))

                elif node_color is None and hasattr(self, "NODE_COLOR") and x_indices is not None:
                    node_color = list(map(lambda i: self.NODE_COLOR.get(i, "skyblue"), x_indices))

            # For Debugging
            # labels = {node: str(node % N) for node in subG.nodes}
            # for n in fixed:
            #     node_color[n] = "red"

            ax = fig.add_subplot(nccs, 1, j + 1)
            ax.set_axis_off()

            nx.draw_networkx(
                subG,
                pos=subpos,
                with_labels=with_labels,
                # with_labels=True,
                labels=labels,
                node_color=node_color,
                edge_color=[color for color, edge in zip(edge_color, G.edges) if edge in subG.edges]
                if edge_color is not None
                else None,
                ax=ax,
                node_size=500,
                font_size=18,
                **kwargs,
            )
        # if match_graphs:
        #     plt.savefig("graph.png")
        #     breakpoint()
        return fig

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

    def dummy_graph(self, num_nodes, XA=False):
        adj = self.random_connected_adj(num_nodes)
        if self.node_feature_type == "constant":
            x = torch.ones((num_nodes, self.num_node_features))
        elif self.node_feature_type == "one-hot":
            x = torch.eye(self.num_node_features)[torch.randint(self.num_node_features, (num_nodes,))]
        elif self.node_feature_type == "degree":
            x = torch.unsqueeze(torch.sum(adj, dim=-1), dim=-1)
        if XA:
            return x, adj
        else:
            return Data(x=x, edge_index=dense_to_sparse(adj)[0])
