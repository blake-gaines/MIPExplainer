import pickle
import os
from .dataset import GraphDataset
from .d4exp_datasets.NCI1_dataset import NCI1
from .d4exp_datasets.ba3motif_dataset import BA3Motif

# from .d4exp_datasets.bbbp_dataset import bbbp
# from .d4exp_datasets.mutag_dataset import Mutagenicity
import torch


class NCI1_dataset(GraphDataset):
    root = "data/GExp-dataset/NCI1"
    GRAPH_CLS = {
        0: "Non-Active",
        1: "Active",
    }
    NODE_CLS = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br", 7: "S"}
    NODE_COLOR = {
        0: "lightgray",
        1: "deepskyblue",
        2: "red",
        3: "cyan",
        4: "magenta",
        5: "springgreen",
        6: "chocolate",
        7: "gold",
    }

    def get_data(self):
        self.datasets = [NCI1(root=self.root, mode=mode) for mode in ["training", "evaluation", "testing"]]
        data = [d for dataset in self.datasets for d in dataset]
        print("NCI1: Original Dataset Length:", len(data))
        ## Taking out molecules with uncommon atom types
        atom_types = [torch.argwhere(d.x)[:, 1].unique().tolist() for d in data]
        atom_type_counts = torch.tensor([sum(1 for u in atom_types if i in u) for i in range(data[0].x.shape[1])])
        feature_include_indices = torch.argwhere(atom_type_counts > 100).flatten()
        for d in data:
            d.x = d.x[:, feature_include_indices]
        data = [d for d in data if not (d.x == 0).all(axis=1).any()]
        print("NCI1: Shortened Dataset Length:", len(data))
        # self.NODE_CLS = dict(enumerate(self.atoms))
        print("WARNING: I MADE UP THE NODE FEATURES FOR THIS DATASET WE HAVE TO FIGURE OUT THE REAL ONES")
        return data


class BA3Motif_dataset(GraphDataset):
    root = "data/GExp-dataset/BA3Motif"

    def get_data(self):
        self.datasets = [BA3Motif(root=self.root, mode=mode) for mode in ["training", "evaluation", "testing"]]
        data = [d for dataset in self.datasets for d in dataset]
        return data
