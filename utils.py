import argparse
import torch
from datasets import get_dataset
from gnn import GNN  # noqa: F401
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset_name", type=str, required=True, help="Name of dataset"
    )  # , choices=["MUTAG", "Shapes", "Shapes_Clean", "OurMotifs", "Is_Acyclic"]
    parser.add_argument(
        "-m",
        "--max_class",
        type=int,
        required=True,
        help="Index of logit to be maximized",
    )
    parser.add_argument(
        "-n",
        "--num_nodes",
        type=int,
        help="Number of nodes in the explanation graph",
    )

    parser.add_argument("--model_path", type=str, help="Path to model file")
    parser.add_argument("--device", type=int, help="Index of device to use")

    parser.add_argument(
        "--init_with_data",
        action="store_true",
        help="If true, initialize with the graph in the dataset (index of init_index), otherwise start with a predefined graph",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Name of output file",
    )
    parser.add_argument(
        "-p",
        "--param_file",
        type=str,
        default="./tune0.prm",
        help="Name of file containing solver parameters",
    )
    parser.add_argument("--log", action="store_true", help="Log the run with Weights & Biases")
    parser.add_argument("--mask_index", type=int, help="Index of mask to be applied to the graph")
    parser.add_argument("--budget", type=int, help="Edge Budget", default=1)
    parser.add_argument("--no-log", dest="log", action="store_false")
    parser.add_argument(
        "--tags",
        action="extend",
        nargs="+",
        type=str,
        help="Add tags for the run",
        default=[],
    )

    return parser.parse_args()


def setup():
    args = parse_args()

    if args.log:
        import wandb

        wandb.login()
        wandb.init(project="GNN-Inverter")
        if args.param_file is not None:
            wandb.save(args.param_file, policy="now")
        wandb.run.log_code(".")

    if args.output_file is not None and os.path.isfile(args.output_file):
        pass
    elif args.log and args.output_file is not None:
        args.output_file = f"{args.output_file}/{wandb.run.id}.pkl"
    else:
        args.output_file = "./results.pkl"

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.log:
        wandb.config.update(vars(args))
        wandb.save(args.param_file, policy="now")
        wandb.save(args.output_file, policy="end")
        wandb.run.log_code(".")
        wandb.run.tags += tuple(args.tags)

    args.device = (
        args.device if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not args.model_path:
        args.model_path = f"models/{args.dataset_name}_model.pth"

    # Load the model
    nn = torch.load(args.model_path, fix_imports=True, map_location=args.device)
    nn.device = args.device
    nn.eval()
    nn.to(torch.float64)
    if args.log:
        wandb.config["architecture"] = str(nn)

    dataset = get_dataset(args.dataset_name)

    return args, nn, dataset


def get_logging_callback(args, inverter, draw_function=None):
    import wandb

    default_callback = inverter.get_default_callback()

    def logging_callback(model, where):
        r = default_callback(model, where)
        if r is None:
            return
        key, data = r
        if args.log and key == "Solution":
            if data["Divergence"] < 1e-4:
                if data["Output"].shape[1] == 1:
                    wandb.log(
                        {f"Output Logit {i}": data["Output"].squeeze()[i] for i in range(data["Output"].shape[1])},
                        commit=False,
                    )
                else:
                    for j in range(data["Output"].shape[0]):
                        wandb.log(
                            {
                                f"Graph {j} Output Logit {i}": data["Output"][j][i]
                                for i in range(data["Output"].shape[1])
                            },
                            commit=False,
                        )
                fig = draw_function(A=data["A"], X=data["X"], match_graphs=True)
                wandb.log({"fig": wandb.Image(fig)}, commit=False)
                plt.close()
                fig = draw_function(A=data["A"], X=data["X"], match_graphs=False)
                wandb.log({"fig2": wandb.Image(fig)}, commit=False)
                plt.close()
            else:
                print("Skipped drawing solution with divergence", data["Divergence"])
        wandb.log(data)
        plt.close()

    return logging_callback
