from .dataset import GraphDataset
from torch_geometric.datasets import TUDataset
import os


class ENZYMES_dataset(GraphDataset):
    node_feature_type = "one-hot"

    NODE_CLS = {
        0: "A",
        1: "B",
        2: "C",
    }

    NODE_COLOR = {
        0: "lightgray",
        1: "deepskyblue",
        2: "red",
    }

    GRAPH_CLS = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        dataset = TUDataset(root=os.path.join(self.root, "TUDataset"), name="ENZYMES")
        return dataset
