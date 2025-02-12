from .dataset import GraphDataset
from torch_geometric.datasets import TUDataset
import os
import torch


class IMDB_Binary_dataset(GraphDataset):
    node_feature_type = "constant"

    GRAPH_CLS = {
        0: "Action",
        1: "Romance",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        dataset = TUDataset(root=os.path.join(self.root, "TUDataset"), name="IMDB-BINARY")
        # for i in range(len(dataset)):
        #     dataset[i].x = torch.ones(dataset[i].num_nodes, 1)
        new_dataset = []
        for data in dataset:
            data.x = torch.ones((data.num_nodes, 1))
            new_dataset.append(data)
        return new_dataset

    def draw_graph(self, *args, **kwargs):
        return super().draw_graph(*args, **kwargs, with_labels=False)
