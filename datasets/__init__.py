def get_dataset(name):
    # TODO: Dataset Args
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
    elif name == "MNISTSuperpixels":
        from .MNISTSuperpixels import MNISTSuperpixels_dataset

        dataset = MNISTSuperpixels_dataset()
    elif name == "NCI1":
        from .GExp import NCI1_dataset

        dataset = NCI1_dataset()
    elif name == "IMDB-BINARY":
        from .IMDB_BINARY import IMDB_Binary_dataset

        dataset = IMDB_Binary_dataset()
    elif name == "REDDIT-BINARY":
        from .REDDIT_BINARY import Reddit_Binary_dataset

        dataset = Reddit_Binary_dataset()
    else:
        raise ValueError(f"No dataset with the name '{name}' was found")
    return dataset
