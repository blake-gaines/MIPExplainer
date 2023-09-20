import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from torch_geometric.utils import to_networkx

def draw_graph(A=None, X=None, edge_index=None, data=None, label_dict=None, color_dict=None, directed=True, **kwargs):
    random.seed(7)
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
            k = max(x_indices) + 1
        else:
            x_indices = np.argmax(X, axis=1)
            k = X.shape[1]
        labels = dict(zip(range(X.shape[0]), map(label_dict.get, x_indices))) if label_dict else dict(zip(range(X.shape[0]), x_indices))

        if color_dict is None:
            color_dict = {i:c for i,c in enumerate(random.choices(list(mcolors.CSS4_COLORS.values()), k=X.shape[1]))}
        node_color = list(map(lambda i: color_dict.get(i, "skyblue"), x_indices))

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, with_labels=True, labels=labels, node_color=node_color, **kwargs)

    return fig, ax

def save_graph(A, X, index):
    np.save(f"solutions/X_{index}.npy", X)
    np.save(f"solutions/A_{index}.npy", A)

def get_average_phi(dataset, nn, layer_name):
    embedding_sum = None
    num_classes = len(set([data.y for data in dataset]))
    n_instances = torch.zeros(num_classes)
    for data in dataset:
        # data.x = data.x.double()
        embeddings = dict(nn.get_all_layer_outputs(data))[layer_name]
        if embedding_sum is None: 
            embedding_sum = torch.zeros(num_classes, embeddings.shape[-1])
        embedding_sum[data.y] += torch.sum(embeddings, dim=0)
        n_instances[data.y] += 1
    return (embedding_sum / torch.unsqueeze(n_instances, 1)).detach().numpy()

def get_matmul_bounds(MVar, W):
    lower_bounds = ((MVar.getAttr("lb") @ W.clip(min=0)) + (MVar.getAttr("ub") @ W.clip(max=0))).squeeze()
    upper_bounds = ((MVar.getAttr("ub") @ W.clip(min=0)) + (MVar.getAttr("lb") @ W.clip(max=0))).squeeze()
    return lower_bounds, upper_bounds