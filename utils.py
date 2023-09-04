import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def draw_graph(A, X, label_dict=None, color_dict=None, directed=True):
    random.seed(7)

    G = nx.from_numpy_array(A, create_using=nx.DiGraph if directed else nx.Graph)

    x_indices = np.argmax(X, axis=1)

    labels = dict(zip(range(X.shape[0]), map(label_dict.get, x_indices))) if label_dict else dict(zip(range(X.shape[0]), x_indices))
    pos = nx.spring_layout(G, seed=7)

    if color_dict is None:
        color_dict = {i:c for i,c in enumerate(random.choices(list(mcolors.CSS4_COLORS.values()), k=X.shape[1]))}
    node_color = list(map(color_dict.get, x_indices))

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, with_labels=True, labels=labels, node_color=node_color)

    return fig, ax

def save_graph(A, X, index):
    np.save(f"solutions/X_{index}.npy", X)
    np.save(f"solutions/A_{index}.npy", A)

def get_average_phi(dataset, nn, layer_name):
    embedding_sum = None
    n_instances = torch.zeros(dataset.num_classes)
    for data in dataset:
        data.x = data.x.double()
        embeddings = dict(nn.get_all_layer_outputs(data))[layer_name]
        if embedding_sum is None: 
            embedding_sum = torch.zeros(dataset.num_classes, embeddings.shape[-1])
        embedding_sum[data.y] += torch.sum(embeddings, dim=0)
        n_instances[data.y] += 1
    return (embedding_sum / torch.unsqueeze(n_instances, 1)).detach().numpy()