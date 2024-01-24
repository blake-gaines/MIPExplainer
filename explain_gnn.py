import torch
import gurobipy as gp
from gurobipy import GRB
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from arg_parser import parse_args
from datasets import get_dataset
from inverter import Inverter, ObjectiveTerm
import invert_utils
import numpy as np
import random
from gnn import GNN  # noqa: F401

args = parse_args()

dataset_name = args.dataset_name
model_path = args.model_path
max_class = args.max_class
output_file = args.output_file
sim_weights = dict(zip(args.regularizers, args.regularizer_weights))
sim_methods = args.regularizers
num_nodes = args.num_nodes

if not model_path:
    model_path = f"models/{dataset_name}_model.pth"

torch.manual_seed(12345)
# TODO: Seed for Gurobi

if not os.path.isdir("solutions"):
    os.mkdir("solutions")

dataset = get_dataset(dataset_name)
num_node_features = dataset.num_node_features


def canonicalize_graph(graph):
    # This function will reorder the nodes of a given graph (PyTorch Geometric "Data" Object) to a canonical (maybe) version
    # TODO: Generalize to non one-hot vector node features

    # Lexicographic ordering of node features (one-hot)
    feature_ordering = np.argsort(np.argmax(graph.x.detach().numpy(), axis=1))
    node_degree_ordering = to_dense_adj(graph.edge_index).squeeze().sum(axis=1)
    lexicographic_ordering = np.lexsort((feature_ordering, node_degree_ordering))

    G = to_networkx(init_graph)

    # Sort by node degree, then by lexicographic ordering
    ## DFS to get a node ordering, prioritizing first by node degree, then by lexicographic ordering of node features
    def lexicographic_dfs_ordering(G, node, visited):
        visited[node] = True
        ## Get the neighbors of the current node
        neighbors = list(G.neighbors(node))
        ## Sort them lexicographically by featuer and degree
        tree_order = [node]
        for node in feature_ordering:
            if node in neighbors:
                if not visited[node]:
                    subtree_order = lexicographic_dfs_ordering(G, node, visited)
                    tree_order.extend(subtree_order)
        return tree_order

    visited = [False] * len(graph.x)
    sorted_nodes = lexicographic_dfs_ordering(G, lexicographic_ordering[0], visited)

    # # Get node ordering
    # sorted_nodes = list(nx.dfs_preorder_nodes(G, source=min_node))
    # sorted_nodes.reverse()

    # Create a mapping of old labels to new labels
    label_mapping = {node: i for i, node in enumerate(sorted_nodes)}

    # Relabel the graph
    G = nx.relabel_nodes(G, label_mapping)

    graph.x = graph.x[sorted_nodes, :]
    graph.edge_index = torch.Tensor(list(G.edges)).to(torch.int64).T


# Load the model
nn = torch.load(model_path, fix_imports=True)
nn.eval()
nn.to(torch.float64)

# Track hyperparameters
if args.log:
    import wandb

    wandb.login()
    config = {
        "architecture": str(nn),
        "model_path": model_path,
    }
    config.update(vars(args))
    wandb.init(
        project="GNN-Inverter",
        config=config,
    )
    wandb.save(args.param_file, policy="now")
    wandb.save(output_file, policy="end")
    wandb.run.log_code(".")

print("Args:", args)
print("Number of Classes", dataset.num_classes)
print("Number of Node Features", num_node_features)

if args.init_with_data:
    # Initialize with a graph from the dataset
    if args.init_index is not None:
        print(f"Initializing from dataset with graph at index {args.init_index}")
        init_graph = dataset[args.init_index]
    elif num_nodes is not None:
        print(f"Initializing from dataset graph with {num_nodes} nodes")
        init_graph = random.choice(
            [d for d in dataset if int(d.y) == max_class and d.num_nodes == num_nodes]
        )
    A = to_dense_adj(init_graph.edge_index).detach().numpy().squeeze()
    print(
        "Connected before reordering:",
        all(
            [
                sum(A[i][j] + A[j][i] for j in range(i + 1, A.shape[0])) >= 1
                for i in range(A.shape[0] - 1)
            ]
        ),
    )

    A = to_dense_adj(init_graph.edge_index).detach().numpy().squeeze()
    assert all(
        [
            sum(A[i][j] + A[j][i] for j in range(i + 1, A.shape[0])) >= 1
            for i in range(A.shape[0] - 1)
        ]
    ), "Initialization graph was not connected"

    num_nodes = init_graph.num_nodes

else:
    # Initialize with a dummy graph
    # By default, will generate a line graph with uniform random node features
    print("Initializing with dummy graph")
    # init_graph_adj = np.clip(init_graph_adj + np.eye(num_nodes, k=1), a_min=0, a_max=1)
    init_graph_adj = torch.diag_embed(
        torch.diag(torch.ones((num_nodes, num_nodes)), diagonal=-1), offset=-1
    ) + torch.diag_embed(
        torch.diag(torch.ones((num_nodes, num_nodes)), diagonal=1), offset=1
    )
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

# Each row of phi is the average embedding of the graphs in the corresponding class of the dataset
phi = dataset.get_average_phi(nn, "Aggregation")

print(nn)
num_model_params = sum(param.numel() for param in nn.parameters())
print("Model Parameters:", num_model_params)
if args.log:
    wandb.run.summary["# Model Parameter"] = num_model_params

env = gp.Env(logfilename="")


def convert_inputs(X, A):
    X = torch.Tensor(X)
    A = torch.Tensor(A)
    return {"data": Data(x=X, edge_index=dense_to_sparse(A)[0])}


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
else:
    raise ValueError(f"Unknown Decision Variables for {dataset_name}")

inverter.set_input_vars({"X": X, "A": A})

# Enforce canonical (maybe) representation
# Only works for one-hot node features and assumes an undirected graph
m.update()
if X[0][0].vtype == GRB.BINARY:
    # messages [i][j][k] is 1 if node i is a neighbor of node j and node i's feature k is 1
    messages = m.addMVar((num_nodes, num_nodes, num_node_features), vtype=GRB.BINARY)
    messages = m.addVars(
        [
            (i, j, k)
            for i in range(num_nodes)
            for j in range(num_nodes)
            for k in range(num_node_features)
        ],
        vtype=GRB.BINARY,
        name="initial_messages_constraint",
    )
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_node_features):
                m.addGenConstrIndicator(
                    messages[i, j, k],
                    1,
                    A[i, j] + X[i, k],
                    GRB.EQUAL,
                    2,
                    name=f"initial_message_{i}_{j}_{k}",
                )

    # neighborhoods_match[i,j,k] is a binary decision variable, constrained to 1 if node i and node j have the same number of neighbors with feature k equal to 1, 0 otherwise
    neighborhoods_match = m.addVars(
        [
            (i, j, k)
            for i in range(num_nodes - 1)
            for j in range(i + 1, num_nodes)
            for k in range(num_node_features)
        ],
        vtype=GRB.BINARY,
        name="neighborhoods_match",
    )
    # obeys_orderings is only 1 if two nodes have the correct partial ordering based on their node features
    # If they have the same neighborhood, then the node with the smaller features lexicographically should come first in the ordering
    obeys_orderings = m.addVars(
        [(i, j) for i in range(num_nodes - 1) for j in range(i + 1, num_nodes)],
        vtype=GRB.BINARY,
        name="obeys_orderings",
    )
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            for k in range(num_node_features):
                # Constrain the neighborhoods_match variables to the correct values
                i_neighborhood_ks = messages.select(i, "*", k)
                i_neighborhood_ks.pop(i)
                j_neighborhood_ks = messages.select(j, "*", k)
                j_neighborhood_ks.pop(j)
                m.addGenConstrIndicator(
                    neighborhoods_match[i, j, k],
                    1,
                    sum(i_neighborhood_ks) - sum(j_neighborhood_ks),
                    GRB.EQUAL,
                    0,
                    name=f"neighborhood_match_constraint_{i}_{j}_{k}",
                )

            # Constrain the obeys_orderings variables to the correct values
            # For each k in the number of features, the sum of the features of node i before or at k equal the sum of the features of node j at or after k
            m.addGenConstrIndicator(
                obeys_orderings[i, j],
                1,
                sum(
                    sum(X[i][: k + 1]) - sum(X[j][k:]) for k in range(num_node_features)
                ),
                GRB.GREATER_EQUAL,
                0,
                name=f"obeys_ordering_constraint_{i}_{j}",
            )

            # If the neighborhoods match, the ordering must be obeyed
            m.addConstr(
                sum(neighborhoods_match.select(i, j, "*")) - obeys_orderings[i, j]
                <= num_node_features - 1,
                name=f"nb_ordering_{i}_{j}",
            )

    ## The first node will have the highest degree and come first lexicographically among nodes with the highest degree
    weighted_feature_sums = A @ (
        X @ np.linspace(1, num_node_features, num=num_node_features)
    )
    for j in range(1, num_nodes):
        # TODO: Fix for directed graphs, integer features
        m.addConstr(
            sum(A[0]) * num_node_features * num_nodes + sum(weighted_feature_sums[0])
            <= sum(A[j]) * num_node_features * num_nodes
            + sum(weighted_feature_sums[j]),
            name=f"node_0_smallest_{j}",
        )
    m.update()

## Build a MIQCP for the trained neural network
## For each layer, create and constrain decision variables to represent the output
previous_layer_output = X
for name, layer in nn.layers.items():
    previous_layer_output = invert_utils.invert_torch_layer(
        inverter.model,
        layer,
        name=name,
        X=previous_layer_output,
        A=A,
    )
    inverter.output_vars[name] = previous_layer_output

invert_utils.order_onehot_features(inverter.m, A, X)
canonicalize_graph(init_graph)

## Create decision variables to represent (unweighted) regularizer terms based on embedding similarity/distance
## These can also be used in constraints!!!
embedding = inverter.output_vars["Aggregation"][0]
regularizers = {}
if "Cosine" in sim_methods:
    var, calc = invert_utils.get_cosine_similarity(
        inverter.model, embedding, phi[max_class]
    )
    inverter.add_objective_term(
        ObjectiveTerm(
            name="Cosine Similarity",
            var=var,
            calc=calc,
            weight=sim_weights["Cosine"],
            required_vars=[embedding],
        ),
    )
if "L2" in sim_methods:
    var, calc = invert_utils.get_l2_distance(inverter.model, embedding, phi[max_class])
    inverter.add_objective_term(
        ObjectiveTerm(
            name="L2 Distance",
            var=var,
            calc=calc,
            weight=sim_weights["L2"],
            required_vars=[embedding],
        ),
    )
if "Squared L2" in sim_methods:
    var, calc = invert_utils.get_l2_distance(inverter.model, embedding, phi[max_class])
    inverter.add_objective_term(
        ObjectiveTerm(
            name="Squared L2 Distance",
            var=var,
            calc=calc,
            weight=sim_weights["Squared L2"],
            required_vars=[embedding],
        ),
    )
m.update()

# List of decision variables representing the logits that are not the max_class logit
other_outputs_vars = [
    inverter.output_vars["Output"][0, j]
    for j in range(dataset.num_classes)
    if j != max_class
]

# # Create a decision variable and constrain it to the maximum of the non max_class logits
other_outputs_max = m.addVar(
    name="other_outputs_max",
    lb=max(v.getAttr("lb") for v in other_outputs_vars),
    ub=max(v.getAttr("ub") for v in other_outputs_vars),
)
m.addGenConstrMax(other_outputs_max, other_outputs_vars, name="max_of_other_outputs")

max_output_var = (
    inverter.output_vars["Output"][0]
    if args.trim_unneeded_outputs
    else inverter.output_vars["Output"][0, max_class]
)
## MIQCP objective function
inverter.add_objective_term(ObjectiveTerm("Target Class Output", max_output_var))
inverter.add_objective_term(
    ObjectiveTerm("Max Non-Target Class Output", other_outputs_max, weight=-1)
)

m.update()

# Save a copy of the model
model_files = inverter.save_model()
if args.log:
    for fn in model_files:
        wandb.save(fn, policy="now")


# Define the callback function for the solver to save intermediate solutions, other metrics
def callback(model, where):
    inverter.get_default_callback()(model, where)
    if where == GRB.Callback.MIPSOL:
        print("New Solution Found:", len(inverter.solutions))
        if args.log and inverter.solutions:
            solution = inverter.solutions[-1]
            fig, _ = dataset.draw_graph(A=solution["A"], X=solution["X"])
            # plt.savefig("test.png")
            wandb.log(solution, commit=False)
            wandb.log(
                {
                    f"Output Logit {i}": solution["Output"].squeeze()[i]
                    for i in range(solution["Output"].shape[1])
                },
                commit=False,
            )
            wandb.log({"fig": wandb.Image(fig)})
            plt.close()

        # with open(output_file, "wb") as f:
        #     pickle.dump(inverter.solutions, f)


## Warm start - create an initial solution for the model
bound_summary = inverter.warm_start(
    {"X": init_graph.x, "A": to_dense_adj(init_graph.edge_index).squeeze()},
    debug_mode=False,
)
print(bound_summary)
if args.log:
    wandb.run.summary.update(bound_summary)

# Get solver parameters
m.read(args.param_file)

# Run Optimization
inverter.solve(
    callback,
    TimeLimit=3600 * 6,
)

# Save all solutions
with open(output_file, "wb") as f:
    pickle.dump(inverter.solutions, f)

print("Model Status:", m.Status)

if m.Status in [3, 4]:  # If the model is infeasible, see why
    inverter.computeIIS()
