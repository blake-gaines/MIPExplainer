import torch
from gnn import GNN
import gurobipy as gp
from gurobipy import GRB
from invert import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
import networkx as nx
import os

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
num_nodes = 12
num_node_features = dataset.num_node_features
init_graphs = (d for d in dataset if d.num_nodes==num_nodes)
next(init_graphs)

def torch_fc_constraint(model, X, layer, name=None):
    return add_fc_constraint(model, X, W=layer.get_parameter(f"weight").detach().numpy(), b=layer.get_parameter(f"bias").detach().numpy(), name=name)

def torch_sage_constraint(model, A, X, layer, name=None):
    lin_r_weight = layer.get_parameter(f"lin_r.weight").detach().numpy()
    lin_l_weight = layer.get_parameter(f"lin_l.weight").detach().numpy()
    lin_l_bias = layer.get_parameter(f"lin_l.bias").detach().numpy()
    lin_weight, lin_bias = None, None
    if layer.project and hasattr(layer, 'lin'):
        lin_weight = layer.get_parameter(f"lin.weight").detach().numpy()
        lin_bias = layer.get_parameter(f"lin.bias").detach().numpy()
    return add_sage_constraint(model, A, X, lin_r_weight=lin_r_weight, lin_l_weight=lin_l_weight, lin_l_bias=lin_l_bias, lin_weight=lin_weight, lin_bias=lin_bias, project=layer.project, name=name)

def to_batch(X, A):
    g = dict()
    g["edge_index"], g["edge_weight"] = dense_to_sparse(A)
    g["x"] = X
    return Batch.from_data_list([Data(**g)])

def doubler(G):
    G.x = G.x.double()
    if G.edge_weight is not None: G.edge_weight = G.edge_weight.double()

def save_graph(A, X, index):
    np.save(f"solutions/X_{index}.npy", X)
    np.save(f"solutions/A_{index}.npy", A)

if __name__ == "__main__":
    model_path = "models/MUTAG_model.pth"
    nn = torch.load(model_path)
    nn.eval()
    nn.double()

    print('\n'.join(f"{t[0]}:".ljust(20)+f"{t[1].shape}" for t in nn.named_parameters()))

    m = gp.Model("GNN Inverse")

    # m.setParam("MIQCPMethod", 1) 
    A = m.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY, name="A")
    X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X")
    m.update()

    # X = m.addMVar((10, 7), lb=-float("inf"), ub=float("inf"), name="X")
    m.addConstr(gp.quicksum(A) >= 1) # Connectedness, may need a transpose
    force_undirected(m, A)
    m.addConstr(gp.quicksum(X.T) == 1) # REMOVE for non-categorical features
    # A = m.addMVar((10, 10), lb=-5, ub=5, name="A")
    # X = m.addMVar((10, 7), lb=-5, ub=5, name="x")

    output_vars = []

    output_vars.append(torch_sage_constraint(m, A, X, nn.conv1, name="conv1"))
    output_vars.append(add_relu_constraint(m, output_vars[-1], name="conv1_activation"))
    output_vars.append(torch_sage_constraint(m, A, output_vars[-1], nn.conv2, name="conv2"))
    output_vars.append(add_relu_constraint(m, output_vars[-1], name="conv2_activation"))
    output_vars.append(torch_sage_constraint(m, A, output_vars[-1], nn.conv3, name="conv3"))
    # output_vars.append(add_relu_constraint(m, output_vars[-1]))
    # TODO: Add ReLU
    output_vars.append(global_mean_pool(m, output_vars[-1], name="pool"))
    output_vars.append(torch_fc_constraint(m, output_vars[-1], nn.lin1, name="lin1"))
    output_vars.append(add_relu_constraint(m, output_vars[-1], name="lin1_activation"))
    output_vars.append(torch_fc_constraint(m, output_vars[-1], nn.lin2, name="lin2"))
    output_vars.append(add_relu_constraint(m, output_vars[-1], name="lin2_activation"))
    output_vars.append(torch_fc_constraint(m, output_vars[-1], nn.lin3, name="lin3"))

    # m.setObjective((output[0]-1.5)*(output[0]-1.5), GRB.MINIMIZE)
    m.setObjective(output_vars[-1][0], GRB.MAXIMIZE)

    m.update()
    m.write("model.mps")

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

            batch = to_batch(torch.Tensor(X_sol).double(), torch.Tensor(A_sol.astype(int)))
            print("NN output given X:", nn.get_embedding_outputs(batch)[1].detach().numpy())
            print("predicted output:", output_sol)
            save_graph(A=A_sol, X=X_sol, index=solution_count)

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
        X.Start = G.x.detach().numpy()
        A.Start = to_dense_adj(G.edge_index).detach().numpy().squeeze()
        batch = Batch.from_data_list([G])
        all_outputs = nn.get_all_layer_outputs(batch)
        assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
        for var, output in zip(output_vars, all_outputs):
            output = output.detach().numpy().squeeze()
            assert var.shape == output.shape
            var.Start = output
    m.update()
    save_graph(A.Start, X.Start, 0)

    # X_save = np.load("./X.npy")
    # A_save = np.load("./A.npy")
    # X.Start = X_save
    # A.Start = A_save
    # batch = to_batch(torch.Tensor(X_save).double(), torch.Tensor(A_save).double())
    # all_outputs = nn.get_all_layer_outputs(batch)
    # assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
    # for var, output in zip(output_vars, all_outputs):
    #     output = output.detach().numpy().squeeze()
    #     assert var.shape == output.shape
    #     var.Start = output

    # m.addConstr(X == np.zeros(X.shape))
    # batch = to_batch(torch.zeros(X.shape).double(), torch.zeros(A.shape).double())
    # all_outputs = nn.get_all_layer_outputs(batch)
    # assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
    # for var, output in zip(output_vars, all_outputs):
    #     output = output.detach().numpy().squeeze()
    #     assert var.shape == output.shape
    #     m.addConstr(var == output)

    # print("Tuning")
    # m.setParam("TuneTimeLimit", 120)
    # m.tune()
    # for i in range(m.tuneResultCount):
    #     m.getTuneResult(i)
    #     m.write('tune'+str(i)+'.prm')
    # print("Done Tuning")

    # m.setParam("TimeLimit", 1000)
    m.optimize(callback)

    print("Status:", m.Status)

    if m.Status in [3, 4]:
        m.computeIIS()
        m.write("model.ilp")
    else:
        save_graph(A.X, X.X, "Final")
        # print("NN output given X", nn(x=torch.Tensor(X.X), edge_index=dense_to_sparse(torch.Tensor(A.X.astype(np.int64)))[0], batch=torch.zeros(10,dtype=torch.int64)))
        batch = to_batch(torch.Tensor(X.X).double(), torch.Tensor(A.X.astype(int)))
        print("NN output given X", nn.get_embedding_outputs(batch)[1].detach().numpy())
        print("NN output given embedding", nn.classify(torch.Tensor(output_vars[5].X).double()))
        print("predicted output", output_vars[-1].X)

        # a = nn.conv1(torch.Tensor(X.X), dense_to_sparse(torch.Tensor(A.X.astype(int)))[0]).detach().numpy()
        # b = hidden.X.astype(np.float32)
        # print((a==b).astype(int))
        # idx = np.nonzero(1-(a==b).astype(int))
        # print(a[idx], b[idx])
        # print("ADJ MATRIX", A.X)
        # print("X", X.X)