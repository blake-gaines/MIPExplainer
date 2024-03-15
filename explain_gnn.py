import torch
import gurobipy as gp
from gurobipy import GRB
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import os
import pickle
from arg_parser import parse_args
from datasets import get_dataset
from inverter import Inverter, ObjectiveTerm
import invert_utils
import numpy as np
import random
from gnn import GNN  # noqa: F401
import time


def callback(model, where):
    r = inverter.get_default_callback()(model, where)
    if r is None:
        return
    key, data = r
    if key == "Solution" and args.log:
        fig, _ = dataset.draw_graph(A=data["A"], X=data["X"])
        wandb.log(
            {
                f"Output Logit {i}": data["Output"].squeeze()[i]
                for i in range(data["Output"].shape[1])
            },
            commit=False,
        )
        wandb.log({"fig": wandb.Image(fig)}, commit=False)
        # plt.close()
    wandb.log(data)


def convert_inputs(X, A):
    X = torch.Tensor(X)
    A = torch.Tensor(A)
    return {"data": Data(x=X, edge_index=dense_to_sparse(A)[0])}


args = parse_args()

dataset_name = args.dataset_name
model_path = args.model_path
max_class = args.max_class
sim_weights = dict(zip(args.regularizers, args.regularizer_weights))
sim_methods = args.regularizers
num_nodes = args.num_nodes
device = (
    args.device
    if args.device is not None
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
if not model_path:
    model_path = f"models/{dataset_name}_model.pth"

dataset = get_dataset(dataset_name)
num_node_features = dataset.num_node_features

# Load the model
nn = torch.load(model_path, fix_imports=True, map_location=device)
nn.device = device
nn.eval()
nn.to(torch.float64)


run_data = vars(args)
run_data["architecture"] = str(nn)

# Track hyperparameters
if args.log:
    import wandb

    wandb.login()
    wandb.init(
        project="GNN-Inverter",
        config=run_data,
    )
    wandb.save(args.param_file, policy="now")
    wandb.run.log_code(".")

if args.output_file is not None:
    output_file = args.output_file
elif args.log:
    output_file = f"results/{wandb.run.id}.pkl"
    wandb.config["output_file"] = output_file
    wandb.save(output_file, policy="end")
else:
    output_file = "./results/results.pkl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("Args:", args)
print("Device:", device)
print("Number of Classes", dataset.num_classes)
print("Number of Node Features", num_node_features)

## TODO: Put this in the dataset class
if args.init_with_data:
    # Initialize with a graph from the dataset
    print(f"Initializing from dataset graph with {num_nodes} nodes")
    init_graph = random.choice(
        [d for d in dataset if int(d.y) == max_class and d.num_nodes == num_nodes]
    )
else:  ## TODO: Add arg for this
    print("Initializing with dummy graph")
    ## Randomly initialized adjacency matrix of a connected graph
    init_graph_adj = torch.randint(0, 2, (num_nodes, num_nodes))
    init_graph_adj = torch.triu(init_graph_adj, diagonal=1)
    init_graph_adj = init_graph_adj + init_graph_adj.T
    init_graph_adj = torch.clip(init_graph_adj, 0, 1)
    init_graph_adj = init_graph_adj.numpy()
    init_graph_adj = np.clip(init_graph_adj + np.eye(num_nodes, k=1), a_min=0, a_max=1)
    init_graph_adj = np.clip(init_graph_adj + np.eye(num_nodes, k=-1), a_min=0, a_max=1)
    init_graph_adj = torch.Tensor(init_graph_adj)

    if dataset_name in ["Is_Acyclic", "Shapes", "Shapes_Clean"]:
        init_graph_x = torch.unsqueeze(torch.sum(init_graph_adj, dim=-1), dim=-1)
    elif dataset_name in ["MUTAG", "OurMotifs"]:
        # init_graph_x = torch.eye(num_node_features)[torch.randint(num_node_features, (num_nodes,)),:]
        init_graph_x = torch.eye(num_node_features)[torch.randint(1, (num_nodes,)), :]
    elif dataset_name in ["Shapes_Ones", "Is_Acyclic_Ones"]:
        init_graph_x = torch.ones((num_nodes, num_node_features))

    # init_graph_adj = torch.randint(0, 2, (num_nodes, num_nodes))
    # init_graph_adj = torch.ones((num_nodes, num_nodes))
    init_graph = Data(x=init_graph_x, edge_index=dense_to_sparse(init_graph_adj)[0])

print(nn)

num_model_params = sum(param.numel() for param in nn.parameters())
print("Model Parameters:", num_model_params)
run_data["# Model Parameter"] = num_model_params

env = gp.Env(logfilename="")


start_time = time.time()

inverter = Inverter(args, nn, dataset, env, convert_inputs)
m = inverter.model

# Add and constrain decision variables for adjacency matrix
A = m.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY, name="A")
invert_utils.force_connected(m, A)
invert_utils.force_undirected(m, A)
invert_utils.remove_self_loops(m, A)
# m.addConstr(gp.quicksum(A) >= 1, name="non_isolatied") # Nodes need an edge. Need this for SAGEConv inverse to work. UNCOMMENT IF NO OTHER CONSTRAINTS DO THIS

# Add and constrain decision variables for node feature matrix
if dataset_name in ["MUTAG", "OurMotifs"]:
    X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X")
    m.addConstr(gp.quicksum(X.T) == 1, name="categorical_features")
elif dataset_name in ["Is_Acyclic", "Shapes", "Shapes_Clean"]:
    X = m.addMVar(
        (num_nodes, num_node_features),
        lb=0,
        ub=init_graph.num_nodes,
        name="X",
        vtype=GRB.INTEGER,
    )
    m.addConstr(X == gp.quicksum(A)[:, np.newaxis], name="features_are_node_degrees")
elif dataset_name in ["Shapes_Ones", "Is_Acyclic_Ones"]:
    X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X")
    m.addConstr(X == 1, name="features_are_ones")
    X.setAttr("lb", 1)
    X.setAttr("ub", 1)
else:
    raise ValueError(f"Unknown Decision Variables for {dataset_name}")

inverter.set_input_vars({"X": X, "A": A})
inverter.set_tracked_vars({"X": X, "A": A})

# invert_utils.order_onehot_features(inverter.m, A, X) # TODO: See if this works better for MUTAG

invert_utils.canonicalize_graph(init_graph)

inverter.encode_seq_nn(
    {
        "X": init_graph.x.detach().numpy(),
        "A": to_dense_adj(init_graph.edge_index).squeeze().detach().numpy(),
    }
)

# print("Objective: All Pairwise Distances")
# inverter.model.setObjective(
#     sum(
#         (inverter.output_vars["Output"][0, i] - inverter.output_vars["Output"][0, j])
#         * (inverter.output_vars["Output"][0, i] - inverter.output_vars["Output"][0, j])
#         for i in range(1, dataset.num_classes)
#         for j in range(i)
#     ),
#     GRB.MINIMIZE,
# )

next_class = max_class + 1 if max_class < dataset.num_classes - 1 else 0
# Constrain max_class and next_class logits to be close
m.addConstr(
    inverter.output_vars["Output"][0, max_class]
    - inverter.output_vars["Output"][0, next_class]
    <= 0.1,
    name="target_class_closeness",
)
m.addConstr(
    inverter.output_vars["Output"][0, next_class]
    - inverter.output_vars["Output"][0, max_class]
    <= 0.1,
    name="target_class_closeness",
)
other_outputs_vars = [
    inverter.output_vars["Output"][0, j]
    for j in range(dataset.num_classes)
    if j != max_class and j != next_class
]
other_outputs_max = invert_utils.add_max_constraint(
    m, other_outputs_vars, name="max_of_other_outputs"
)

# # List of decision variables representing the logits that are not the max_class logit
# other_outputs_vars = [
#     inverter.output_vars["Output"][0, j]
#     for j in range(dataset.num_classes)
#     if j != max_class
# ]
# invert_utils.add_max_constraint(
#     m, other_outputs_vars, name="max_of_other_outputs"
# )

inverter.add_objective_term(
    ObjectiveTerm("Max Non-Target Class Output", other_outputs_max, weight=-1)
)


inverter.add_objective_term(
    ObjectiveTerm("Target Class Output", inverter.output_vars["Output"][0, max_class])
)

print("Objective:", inverter.model.getObjective())
if args.log:
    wandb.run.tags += tuple(inverter.objective_terms.keys())

m.update()


bound_summary = inverter.bounds_summary()
run_data.update(bound_summary)

# Run Optimization
solve_data = inverter.solve(
    callback,
    TimeLimit=round(3600 * 4),
)

if args.log:
    wandb.run.summary.update(run_data)

run_data.update({"mip_information": inverter.mip_data, "solutions": inverter.solutions})

if m.Status in [3, 4]:  # If the model is infeasible, see why
    inverter.computeIIS()

end_time = time.time()
run_data["runtime"] = end_time - start_time


with open(output_file, "wb") as f:
    pickle.dump(run_data, f)
