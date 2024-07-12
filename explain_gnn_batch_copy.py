import torch
import gurobipy as gp
from gurobipy import GRB
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
import pickle
import utils
from inverter import Inverter, ObjectiveTerm  # noqa: F401
import invert_utils
import time
import wandb
import numpy as np  # noqa: F401
from gnn import GNN  # noqa: F401

args, nn, dataset = utils.setup()
run_data = vars(args)

print("Args:", args)
print("args.device:", args.device)
print("Number of Classes", dataset.num_classes)
print("Number of Node Features", dataset.num_node_features)

batch_size = 2

init_graph_comp = dataset.dummy_graph(args.num_nodes)
# init_graph_comp = dataset.get_random_graph(args.max_class, num_nodes=args.num_nodes)

init_graphs = []
for b in range(batch_size):
    if args.init_with_data:
        init_graphs.append(dataset.get_random_graph(args.max_class, num_nodes=args.num_nodes))
        args.num_nodes = init_graphs[-1].num_nodes
    else:
        print("Initializing with dummy graph")
        # init_graphs.append(dataset.dummy_graph(args.num_nodes))
        init_graphs.append(init_graph_comp.clone())
    invert_utils.canonicalize_graph(init_graphs[-1])

init_graph = Batch.from_data_list(init_graphs)
batch = init_graph.batch


def convert_inputs(X, A):
    X = torch.Tensor(X)
    A = torch.Tensor(A)
    return {
        "data": Data(
            x=X,
            edge_index=dense_to_sparse(A)[0],
            batch=batch,
        )
    }


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
A = m.addMVar((init_graph.num_nodes, init_graph.num_nodes), vtype=GRB.BINARY, name="A")
m.addConstr(A[args.num_nodes :, : args.num_nodes] == 0)
m.addConstr(A[: args.num_nodes, args.num_nodes :] == 0)

sizes = [0] + np.cumsum([g.num_nodes for g in init_graphs]).tolist()
As = []

for i, j in zip(sizes, sizes[1:]):
    As.append(A[i:j, i:j])
    m.addConstr(A[i:j, j:] == 0)
    m.addConstr(A[j:, i:j] == 0)
    A[i:j, j:].setAttr("ub", 0)
    A[j:, i:j].setAttr("ub", 0)


for Ai in As:
    invert_utils.force_undirected(m, Ai)
    invert_utils.remove_self_loops(m, Ai)
    invert_utils.s1_constraint(m, Ai)
    invert_utils.s3_constraint(m, Ai)

# m.addConstr(gp.quicksum(A) >= 1, name="non_isolatied") # Nodes need an edge. Need this for SAGEConv inverse to work. UNCOMMENT IF NO OTHER CONSTRAINTS DO THIS

# Add and constrain decision variables for node feature matrix
if dataset.node_feature_type == "one-hot":
    print("Adding One-Hot Features")
    X = invert_utils.get_one_hot_features(m, args.num_nodes, dataset.num_node_features)
    # invert_utils.order_onehot_features(
    #     m, A, X
    # )  ## TODO: See if this works/improves solution time
    print("Adding S2 Constraint")
    invert_utils.s2_constraint(m, X)
elif dataset.node_feature_type == "degree":
    print("Adding Degree Features")
    X = invert_utils.get_node_degree_features(m, init_graph.num_nodes, dataset.num_node_features, A)
elif dataset.node_feature_type == "constant":
    print("Adding Constant Features")
    X = invert_utils.get_constant_features(
        m,
        (init_graph.num_nodes, dataset.num_node_features),
        1,
        vtype=GRB.BINARY,
    )
else:
    print("Adding Continuous Features")
    X = m.addMVar(
        (init_graph.num_nodes, dataset.num_node_features),
        lb=-float("inf"),
        ub=float("inf"),
        name="X",
    )

m.update()


inverter.set_input_vars({"X": X, "A": A})
inverter.set_tracked_vars({"X": X, "A": A})

inverter.encode_seq_nn(
    {
        "X": init_graph.x.detach().numpy(),
        "A": to_dense_adj(init_graph.edge_index).squeeze().detach().numpy(),
    },
    debug=False,
    max_bound=None,
    batch=batch.detach().numpy(),
)
m.update()

bounds_summary = inverter.bounds_summary()
run_data.update(bounds_summary)
if args.log:
    wandb.run.summary.update(bounds_summary)

# embeddings = inverter.output_vars["Aggregation"]
# emb_diffs = embeddings[0] - embeddings[1]
# emb_distance = m.addVar(lb=0, ub=emb_diffs.size, name="emb_dist")
# m.addConstr(emb_distance == gp.quicksum(emb_diffs * emb_diffs))
# # m.addGenConstrNorm(emb_distance, emb_diffs.tolist(), 2)

# inverter.tracked_vars["Embedding Distance"] = emb_distance

# m.addConstr(emb_distance <= 0.1, name="embs_close")

# a_diffs = As[0].reshape((-1,)) - As[1].reshape((-1,))
# a_distance = m.addVar(lb=0, ub=A.size, name="a_dist")
# m.addConstr(a_distance == gp.quicksum(a_diffs * a_diffs))

# inverter.add_objective_term(
#     ObjectiveTerm(
#         "A Distance",
#         a_distance,
#         weight=1,
#     )
# )

# m.addConstr(
#     gp.quicksum(As[1].reshape((-1,))) - gp.quicksum(As[0].reshape((-1,))) <= args.budget
# )

other_outputs_vars_0 = [
    inverter.output_vars["Output"][0, j].item() for j in range(dataset.num_classes) if j != args.max_class
]
max_of_outputs_0 = invert_utils.get_max(m, other_outputs_vars_0, name="max_of_outputs_0")
other_outputs_vars_1 = [
    inverter.output_vars["Output"][1, j].item() for j in range(dataset.num_classes) if j != args.max_class
]
max_of_outputs_1 = invert_utils.get_max(m, other_outputs_vars_1, name="max_of_outputs_1")

# m.addConstr(gp.quicksum((As[1].reshape((-1,)) - As[0].reshape((-1,)))) <= args.budget)
# m.addConstr(As[1] - As[0] >= 0)
m.addConstr(
    gp.quicksum(
        (As[1][i, j] - As[0][i, j]) * (As[1][i, j] - As[0][i, j]) for i in range(args.num_nodes) for j in range(i)
    )
    <= args.budget,
    name="A_Distance_Budget",
)
if args.log:
    wandb.run.tags += ("budget",)

# m.setObjective(
#     inverter.output_vars["Output"][1][args.max_class]
#     - inverter.output_vars["Output"][0][args.max_class]
# )


def minus(m, a, b):
    m.update()
    r = m.addVar(lb=a.lb - b.ub, ub=a.ub - b.lb)
    m.addConstr(r == a - b)
    return r


obj_diff_1 = minus(m, inverter.output_vars["Output"][1, args.max_class], max_of_outputs_1)
obj_diff_2 = minus(m, max_of_outputs_0, inverter.output_vars["Output"][0, args.max_class])

m.update()

# obj1 = invert_utils.add_relu_constraint(m, gp.MVar.fromvar(obj_diff_1))
# obj2 = invert_utils.add_relu_constraint(m, gp.MVar.fromvar(obj_diff_2))
# m.setObjective(obj1 - obj2)
m.setObjective(obj_diff_1 + obj_diff_2)


# m.setObjective(
#     (inverter.output_vars["Output"][1, args.max_class].item() - max_of_outputs_1)
#     - (inverter.output_vars["Output"][0, args.max_class].item() - max_of_outputs_0)
# )
if args.log:
    wandb.run.tags += ("max-prob-diff",)

m.update()

print("Objective:", m.getObjective())
# print("Sense:", m.getObjective().getAttr(GRB.Attr.Sense))
print("Sense:", m.getAttr("ModelSense"))

if args.log:
    wandb.run.name = f"{dataset.GRAPH_CLS[args.max_class].lower()}-{args.budget}-{wandb.run.id[-4:]}"

# if args.log:
#     wandb.run.tags += tuple(inverter.objective_terms.keys())

# Save a copy of the model
model_files = inverter.save_model()
if args.log:
    for fn in model_files:
        wandb.save(fn, policy="now")

solve_data = inverter.solve(
    callback=utils.get_logging_callback(args, inverter, dataset.draw_graph) if args.log else None,
    param_file=args.param_file,
    TimeLimit=round(3600 * 11.5),
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

m.dispose()
