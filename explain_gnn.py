import torch
import gurobipy as gp
from gurobipy import GRB
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import pickle
import utils
from inverter import Inverter, ObjectiveTerm  # noqa: F401
import invert_utils
import time
import wandb
import numpy as np  # noqa: F401
from gnn import GNN  # noqa: F401


def convert_inputs(X, A):
    X = torch.Tensor(X)
    A = torch.Tensor(A)
    return {"data": Data(x=X, edge_index=dense_to_sparse(A)[0])}


args, nn, dataset = utils.setup()
run_data = vars(args)

print("Args:", args)
print("args.device:", args.device)
print("Number of Classes", dataset.num_classes)
print("Number of Node Features", dataset.num_node_features)

if args.init_with_data:
    print(f"Initializing from dataset graph with {args.num_nodes} nodes")
    init_graph = dataset.get_random_graph(args.max_class, num_nodes=args.num_nodes)
    args.num_nodes = init_graph.num_nodes
else:
    print("Initializing with dummy graph")
    init_graph = dataset.dummy_graph(args.num_nodes)
invert_utils.canonicalize_graph(init_graph)

print(nn)

num_model_params = sum(param.numel() for param in nn.parameters())
print("Model Parameters:", num_model_params)
run_data["# Model Parameters"] = num_model_params
if args.log:
    wandb.run.summary["# Model Parameters"] = num_model_params

start_time = time.time()

env = gp.Env(logfilename="")
inverter = Inverter(args, nn, dataset, env, convert_inputs)
m = inverter.model

# Add and constrain decision variables for adjacency matrix
A = m.addMVar((args.num_nodes, args.num_nodes), vtype=GRB.BINARY, name="A")
# invert_utils.force_connected(m, A)
invert_utils.force_undirected(m, A)
invert_utils.remove_self_loops(m, A)
invert_utils.s1_constraint(m, A)
invert_utils.s3_constraint(m, A)
# m.addConstr(gp.quicksum(A) >= 1, name="non_isolatied") # Nodes need an edge. Need this for SAGEConv inverse to work. UNCOMMENT IF NO OTHER CONSTRAINTS DO THIS

# Add and constrain decision variables for node feature matrix
if dataset.node_feature_type == "one-hot":
    print("Adding One-Hot Features")
    X = invert_utils.get_one_hot_features(m, args.num_nodes, dataset.num_node_features)
    # invert_utils.order_onehot_features(
    #     inverter.m, A, X
    # )  ## TODO: See if this works/improves solution time
    print("Adding S2 Constraint")
    invert_utils.s2_constraint(m, X)
elif dataset.node_feature_type == "degree":
    print("Adding Degree Features")
    X = invert_utils.get_node_degree_features(
        m, args.num_nodes, dataset.num_node_features, A
    )
elif dataset.node_feature_type == "constant":
    print("Adding Constant Features")
    X = invert_utils.get_constant_features(
        m,
        (args.num_nodes, dataset.num_node_features),
        1,
        vtype=GRB.BINARY,
    )
else:
    print("Adding Continuous Features")
    X = m.addMVar(
        (args.num_nodes, dataset.num_node_features),
        lb=-float("inf"),
        ub=float("inf"),
        name="X",
    )
inverter.m.update()
inverter.set_input_vars({"X": X, "A": A})
inverter.set_tracked_vars({"X": X, "A": A})

inverter.encode_seq_nn(
    {
        "X": init_graph.x.detach().numpy(),
        "A": to_dense_adj(init_graph.edge_index).squeeze().detach().numpy(),
    },
    debug=False,
    max_bound=None,
)
inverter.m.update()

bounds_summary = inverter.bounds_summary()
run_data.update(bounds_summary)
if args.log:
    wandb.run.summary.update(bounds_summary)

# ### Finding Initial Solution ###
# next_class = (args.max_class + 1) % dataset.num_classes
# if args.log:
#     wandb.run.summary["neg_class"] = next_class
#     wandb.run.name = (
#         dataset.GRAPH_CLS[args.max_class].lower()
#         + "-"
#         + dataset.GRAPH_CLS[next_class].lower()
#         + "-"
#         + wandb.run.id[-4:]
#     )

# inverter.model.setObjective(
#     (
#         inverter.output_vars["Output"][0, args.max_class]
#         - inverter.output_vars["Output"][0, next_class]
#     )
#     * (
#         inverter.output_vars["Output"][0, args.max_class]
#         - inverter.output_vars["Output"][0, next_class]
#     ),
#     sense=GRB.MINIMIZE,
# )

# inverter.model.update()

# print("Objective:", inverter.model.getObjective())

# solve_data = inverter.solve(
#     callback=utils.get_logging_callback(args, inverter, dataset.draw_graph)
#     if args.log
#     else None,
#     TimeLimit=round(3600 * 0.01),
#     param_file=args.param_file,
# )

# new_init_X = inverter.solutions[-1]["X"]
# new_init_A = inverter.solutions[-1]["A"]

# # distance = abs(inverter.solutions[-1]["Output"][:, args.max_class] - inverter.solutions[-1]["Output"][:, next_class])
# distance = 0.1  # TODO: Move to args

# if args.log:
#     wandb.config["distance"] = distance
# assert (
#     abs(
#         inverter.solutions[-1]["Output"][:, args.max_class]
#         - inverter.solutions[-1]["Output"][:, next_class]
#     )
#     <= distance
# )

# print("Constraining Distance Max:", distance)
# ## Constrain args.max_class and next_class logits to be close
# m.addConstr(
#     inverter.output_vars["Output"][0, args.max_class]
#     - inverter.output_vars["Output"][0, next_class]
#     <= distance,
#     name="target_class_closeness_1",
# )
# m.addConstr(
#     inverter.output_vars["Output"][0, next_class]
#     - inverter.output_vars["Output"][0, args.max_class]
#     <= distance,
#     name="target_class_closeness_2",
# )

# inverter.encode_seq_nn({"X": new_init_X, "A": new_init_A}, add_layers=False)

# inverter.reset_objective()

### Main Solve ###

# other_outputs_vars = [
#     inverter.output_vars["Output"][0, j]
#     for j in range(dataset.num_classes)
#     if j != args.max_class and j != next_class
# ]
# other_outputs_max = invert_utils.get_max(
#     m, other_outputs_vars, name="max_of_other_outputs"
# )

# inverter.add_objective_term(
#     ObjectiveTerm(
#         "Target Class Output",
#         inverter.output_vars["Output"][0, args.max_class],
#         weight=-1,
#     )
# )

# inverter.add_objective_term(
#     ObjectiveTerm(
#         "Target Class Output", inverter.output_vars["Output"][0, args.max_class]
#     )
# )

# if args.log:
#     wandb.run.name = dataset.GRAPH_CLS[args.max_class].lower() + "-" + wandb.run.id[-4:]

# inverter.add_objective_term(
#     ObjectiveTerm(
#         "Next Class Output", inverter.output_vars["Output"][0, next_class], weight=-1
#     )
# )

# # List of decision variables representing the logits that are not the args.max_class logit
# other_outputs_vars = [
#     inverter.output_vars["Output"][0, j]
#     for j in range(dataset.num_classes)
#     if j != args.max_class
# ]
# other_outputs_max = invert_utils.get_max(
#     m, other_outputs_vars, name="max_of_other_outputs"
# )

# inverter.add_objective_term(
#     ObjectiveTerm("Max Non-Target Class Output", other_outputs_max, weight=-1)
# )

inverter.model.setObjective(-gp.quicksum(inverter.output_vars["Output"][0, :4]))

inverter.model.update()

print("Objective:", inverter.model.getObjective())

if args.log:
    wandb.run.tags += tuple(inverter.objective_terms.keys())

# Save a copy of the model
model_files = inverter.save_model()
if args.log:
    for fn in model_files:
        wandb.save(fn, policy="now")

solve_data = inverter.solve(
    callback=utils.get_logging_callback(args, inverter, dataset.draw_graph)
    if args.log
    else None,
    param_file=args.param_file,
    TimeLimit=round(3600 * 3),
)

# if m.Status in [3, 4]:  # If the model is infeasible, see why
#     inverter.computeIIS()

end_time = time.time()
run_data["runtime"] = end_time - start_time
if args.log:
    wandb.run.summary.update(solve_data)
    wandb.run.summary["runtime"] = run_data["runtime"]

run_data.update(
    {
        "mip_information": inverter.mip_data,
        "solutions": inverter.solutions,
        "solve_data": solve_data,
    }
)

with open(args.output_file, "wb") as f:
    pickle.dump(run_data, f)
