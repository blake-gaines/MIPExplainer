import pickle
import os
from .dataset import GraphDataset


class Is_Acyclic_Ones_dataset(GraphDataset):
    GRAPH_CLS = {
        0: "Cyclic",
        1: "Acyclic",
    }

    def get_data(self):
        dataset = pickle.load(
            open(os.path.join(self.root, "Is_Acyclic_Ones/dataset.pkl"), "rb")
        )
        return dataset

    def draw_graph(self, *args, **kwargs):
        return super().draw_graph(*args, **kwargs, with_labels=False)
