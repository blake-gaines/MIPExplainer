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