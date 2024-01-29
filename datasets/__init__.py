from .MUTAG import MUTAG_dataset
from .Shapes_Ones import Shapes_Ones_dataset
from .Is_Acyclic_Ones import Is_Acyclic_Ones_dataset


def get_dataset(name):
    if name == "MUTAG":
        dataset = MUTAG_dataset()
    elif name == "Shapes_Ones":
        dataset = Shapes_Ones_dataset()
    elif name == "Is_Acyclic_Ones":
        dataset = Is_Acyclic_Ones_dataset()
    else:
        raise ValueError(f"No dataset with the name '{name}' was found")
    return dataset
