import torch
from gnn import GNN
import gurobipy as gp
from gurobipy import GRB
from invert import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
import networkx as nx
import os
from torch.nn import Linear, ReLU
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from torch_geometric.nn import SAGEConv
import time
from torch_geometric.datasets import TUDataset
import pickle

print(time.strftime("\nStart Time: %H:%M:%S", time.localtime()))

if not os.path.isdir("solutions"): os.mkdir("solutions")

# dataset = TUDataset(root='data/TUDatascet', name='MUTAG')
# num_nodes = 10
# init_graphs = (d for d in dataset if d.num_nodes==num_nodes)
# num_node_features = dataset.num_node_features

with open("data/OurMotifs/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
num_nodes = dataset[0].x.shape[0]
num_node_features = dataset[0].x.shape[1]
init_graphs = (d for d in dataset[:5])
next(init_graphs)

def torch_fc_constraint(model, X, layer, name=None):
    return add_fc_constraint(model, X, W=layer.get_parameter(f"weight").detach().numpy(), b=layer.get_parameter(f"bias").detach().numpy(), name=name)

def torch_sage_constraint(model, A, X, layer, name=None):
    # TODO: Cleanup
    lin_r_weight = layer.get_parameter(f"lin_r.weight").detach().numpy()
    lin_l_weight = layer.get_parameter(f"lin_l.weight").detach().numpy()
    lin_l_bias = layer.get_parameter(f"lin_l.bias").detach().numpy()
    lin_weight, lin_bias = None, None
    if layer.project and hasattr(layer, 'lin'):
        lin_weight = layer.get_parameter(f"lin.weight").detach().numpy()
        lin_bias = layer.get_parameter(f"lin.bias").detach().numpy()
    return add_sage_constraint(model, A, X, lin_r_weight=lin_r_weight, lin_l_weight=lin_l_weight, lin_l_bias=lin_l_bias, lin_weight=lin_weight, lin_bias=lin_bias, project=layer.project, aggr=layer.aggr, name=name)

def to_batch(X, A):
    g = dict()
    g["edge_index"], g["edge_weight"] = dense_to_sparse(A)
    g["x"] = X
    return Batch.from_data_list([Data(**g)])

def doubler(G):
    G.x = G.x.double()
    if G.edge_weight is not None: G.edge_weight = G.edge_weight.to(torch.int64)

def save_graph(A, X, index):
    np.save(f"solutions/X_{index}.npy", X)
    np.save(f"solutions/A_{index}.npy", A)

if __name__ == "__main__":
    # model_path = "models/MUTAG_model.pth"
    model_path = "models/OurMotifs_model.pth"
    nn = torch.load(model_path)
    nn.eval()
    nn.double()

    print(nn)

    m = gp.Model("GNN Inverse")

    A = m.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY, name="A")
    X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X") # vtype for BINARY node features
    m.update()

    # X = m.addMVar((10, 7), lb=-float("inf"), ub=float("inf"), name="X")
    m.addConstr(gp.quicksum(A) >= 1) # Nodes need an edge. Need this for SAGEConv inverse to work
    force_undirected(m, A) # Generate an undirected graph
    m.addConstr(gp.quicksum(X.T) == 1) # REMOVE for non-categorical features

    ## Continuous Alternatives
    # A = m.addMVar((10, 10), lb=-5, ub=5, name="A")
    # X = m.addMVar((10, 7), lb=-5, ub=5, name="x")

    ## Build a MIQCP for the trained neural network
    output_vars = [X]
    for name, layer in nn.layers.items():
        previous_layer_output = output_vars[-1]
        if isinstance(layer, Linear):
            output_vars.append(torch_fc_constraint(m, previous_layer_output, layer, name=name))
        elif isinstance(layer, SAGEConv):
            output_vars.append(torch_sage_constraint(m, A, previous_layer_output, layer, name=name))
        elif isinstance(layer, MeanAggregation):
            output_vars.append(global_mean_pool(m, previous_layer_output, name=name))
        elif isinstance(layer, SumAggregation):
            output_vars.append(global_add_pool(m, previous_layer_output, name=name))
        elif isinstance(layer, ReLU):
            output_vars.append(add_relu_constraint(m, output_vars[-1], name=name))
        else:
            raise NotImplementedError(f"layer type {layer} has no MIQCP analog")

    ## MIQCP objective function
    m.setObjective(output_vars[-1][0], GRB.MAXIMIZE)

    m.update()
    m.write("model.mps") # Save model file

    # Define the callback function
    solution_count = 0
    def callback(model, where):
        global A, X, solution_count
        if where == GRB.Callback.MIPSOL:
            # A new incumbent solution has been found
            print(f"New incumbent solution found (ID {solution_count}) Objective value: {model.cbGet(GRB.Callback.MIPSOL_OBJ)}")

            solution_count += 1
            X_sol = model.cbGetSolution(X)
            A_sol = model.cbGetSolution(A)
            output_sol = model.cbGetSolution(output_vars[-1])

            ## Sanity CHeck
            print("NN output given X:", nn.forwardXA(X_sol, A_sol))
            print("predicted output:", output_sol)
            save_graph(A=A_sol, X=X_sol, index=solution_count)

    ## Start with a graph from the dataset
    m.NumStart = 1
    m.update()
    for s in range(m.NumStart):
        m.params.StartNumber = s
        # X.Start = np.zeros(X.shape)
        # A.Start = np.zeros(A.shape)
        # batch = to_batch(torch.zeros(X.shape).double(), torch.zeros(A.shape).double())
        G = next(init_graphs)
        doubler(G)
        print(X.shape, G.x.shape)
        # X.Start = G.x.detach().numpy()
        A.Start = to_dense_adj(G.edge_index).detach().numpy().squeeze()
        batch = Batch.from_data_list([G])
        all_outputs = nn.get_all_layer_outputs(batch)
        assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
        for var, output in zip(output_vars, all_outputs):
            output = output[1].detach().numpy().squeeze()
            assert var.shape == output.shape
            var.Start = output
    m.update()
    save_graph(A.Start, X.Start, 0)

    # # Tune
    # print("Tuning")
    # m.setParam("TuneTimeLimit", 3600*1)
    # m.tune()
    # m.getTuneResult(0).writeSolution("tune_report.txt")
    # for i in range(m.tuneResultCount):
    #     m.getTuneResult(i)
    #     m.write('tune'+str(i)+'.prm')
    # print("Done Tuning")

    # # Use tuned parameters
    # m.read("./tune0.prm")

    m.setParam("TimeLimit", 3600*4)
    m.optimize(callback)

    print("Status:", m.Status)

    if m.Status in [3, 4]: # If the model is infeasible, see why
        m.computeIIS()
        m.write("model.ilp")
    else:
        save_graph(A.X, X.X, "Final")
        batch = to_batch(torch.Tensor(X.X).double(), torch.Tensor(A.X.astype(int)))
        print("NN output given X", nn.forwardXA(X.X, A.X))
        print("predicted output", output_vars[-1].X)

print(time.strftime("\nEnd Time: %H:%M:%S", time.localtime()))