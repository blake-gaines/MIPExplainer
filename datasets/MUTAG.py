from .dataset import GraphDataset
from torch_geometric.datasets import TUDataset
import os


class MUTAG_dataset(GraphDataset):
    node_feature_type = "one-hot"

    NODE_CLS = {
        0: "C",
        1: "N",
        2: "O",
        3: "F",
        4: "I",
        5: "Cl",
        6: "Br",
    }

    NODE_COLOR = {
        0: "lightgray",
        1: "deepskyblue",
        2: "red",
        3: "cyan",
        4: "magenta",
        5: "springgreen",
        6: "chocolate",
    }

    GRAPH_CLS = {
        0: "Nonmutagen",
        1: "Mutagen",
    }

    EDGE_CLS = {
        0: "aromatic",
        1: "single",
        2: "double",
        3: "triple",
    }

    EDGE_WIDTH = {
        0: 3,
        1: 2,
        2: 4,
        3: 6,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        dataset = TUDataset(root=os.path.join(self.root, "TUDataset"), name="MUTAG")
        return dataset
