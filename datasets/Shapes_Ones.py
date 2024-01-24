import pickle
import os
from .dataset import GraphDataset


class Shapes_Ones_dataset(GraphDataset):
    GRAPH_CLS = {
        0: "Lollipop",
        1: "Wheel",
        2: "Grid",
        3: "Star",
        4: "Other",
    }

    def get_data(self):
        dataset = pickle.load(
            open(os.path.join(self.root, "Shapes_Ones/dataset.pkl"), "rb")
        )
        return dataset
