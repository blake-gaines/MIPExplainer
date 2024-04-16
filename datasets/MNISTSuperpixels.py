from torch_geometric.datasets import MNISTSuperpixels
import os
from .dataset import GraphDataset


class MNISTSuperpixels_dataset(GraphDataset):
    node_feature_type = "continuous"
    GRAPH_CLS = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    def get_data(self):
        dataset = MNISTSuperpixels(root=os.path.join(self.root, "MNISTSuperpixels"))
        return dataset

    def draw_graph(self, *args, **kwargs):
        return super().draw_graph(*args, **kwargs, with_labels=False)
