from .dataset import GraphDataset
from torch_geometric.datasets import TUDataset
import os

class MUTAG_dataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        dataset = TUDataset(root=os.path.join(self.root, "TUDataset"), name='MUTAG')
        return dataset