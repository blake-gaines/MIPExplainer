def get_dataset(name):
    if name == "MUTAG":
        from .MUTAG import MUTAG_dataset

        dataset = MUTAG_dataset()
    elif name == "Shapes_Ones":
        from .Shapes_Ones import Shapes_Ones_dataset

        dataset = Shapes_Ones_dataset()
    elif name == "Is_Acyclic_Ones":
        from .Is_Acyclic_Ones import Is_Acyclic_Ones_dataset

        dataset = Is_Acyclic_Ones_dataset()
    elif name == "ENZYMES":
        from .ENZYMES import ENZYMES_dataset

        dataset = ENZYMES_dataset()
    else:
        raise ValueError(f"No dataset with the name '{name}' was found")
    return dataset
