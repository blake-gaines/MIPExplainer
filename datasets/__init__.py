from .MUTAG import MUTAG_dataset


def get_dataset(name):
    if name == "MUTAG":
        dataset = MUTAG_dataset()
    else:
        raise ValueError(f"No dataset with the name '{name}' was found")
    return dataset
