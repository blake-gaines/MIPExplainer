import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    sol_init_args = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument("-d", "--dataset_name", type=str, required="True", choices=["MUTAG", "Shapes", "Shapes_Clean", "OurMotifs", "Is_Acyclic"], help="Name of dataset")
    parser.add_argument("-m", "--max_class", type=int, required="True", help="Index of logit to be maximized")
    sol_init_args.add_argument("-n","--num_nodes", type=int, help="Number of nodes in the explanation graph")
    
    parser.add_argument("--model_path", type=str, help="Path to model file")

    parser.add_argument("-r", "--regularizers", type=str, action="extend", nargs="+", choices=["Cosine", "Squared L2", "L2"], default=[], help="Names of regularizers to apply")
    parser.add_argument("--regularizer_weights", type=float, action="extend", default=[], nargs="+", help="Regularizer weights, with order corresponding to the names provided to the 'regularizers' argument")

    parser.add_argument("--trim_unneeded_outputs", action="store_true", help="Excludes non-maximized outputs from the model")

    parser.add_argument("--init_with_data", action="store_true", help="If true, initialize with the graph in the dataset (index of init_index), otherwise start with a predefined graph")
    sol_init_args.add_argument("--init_index", type=int, help="Index of initialization graph")

    parser.add_argument("-o", "--output_file", type=str, default="./solutions.pkl", help="Name of output file")
    parser.add_argument("-p", "--param_file", type=str, default="./tune0.prm", help="Name of file containing solver parameters")
    parser.add_argument("--log", action='store_true', help="Log the run with Weights & Biases")
    parser.add_argument('--no-log', dest='log', action='store_false')

    return parser.parse_args()

