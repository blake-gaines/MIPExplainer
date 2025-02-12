from .dataset import GraphDataset
from torch_geometric.datasets import TUDataset
import os
import torch


class Reddit_Binary_dataset(GraphDataset):
    node_feature_type = "constant"

    GRAPH_CLS = {
        0: "QA",
        1: "Discussion",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        dataset = TUDataset(root=os.path.join(self.root, "TUDataset"), name="REDDIT-BINARY")
        new_dataset = []
        for data in dataset:
            data.x = torch.ones((data.num_nodes, 1))
            new_dataset.append(data)
        return new_dataset

    def draw_graph(self, *args, **kwargs):
        return super().draw_graph(*args, **kwargs, with_labels=False)
