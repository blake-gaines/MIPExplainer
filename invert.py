import gurobipy as gp
from gurobipy import GRB
import numpy as np

M = 1024
big_number = 1024

def add_fc_constraint(model, X, W, b, name=None):
    ts = model.addMVar((W.shape[0],), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts==X@W.T + b, name=f"{name}_output_constraint" if name else None)
    return ts[np.newaxis, :]

def add_gcn_constraint(model, A, X, W, b, name=None): # Unnormalized Adjacency Matrix
    ts = model.addMVar((A.shape[0], W.shape[1]), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts == A @ X @ W + b, name=f"{name}_output_constraint" if name else None)
    return ts

def add_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 1)

def remove_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 0)

def force_undirected(model, A):
    for i in range(A.shape[0]):
        for j in range(i, A.shape[1]):
            model.addConstr(A[i][j] == A[j][i], name=f"undirected_{i}_{j}")

# def add_relu_constraint(model, X, name=None):
#     activations = model.addMVar(X.shape, name=name)
#     for index in product(*[range(r) for r in X.shape]):
#         model.addGenConstrMax(activations[index], [X[index].tolist()], constant=0, name=f"{name}_relu_constraint_{index}".replace(" ","") if name else None)
#     return activations

def add_relu_constraint(model, X, name=None):
    ts = model.addMVar(X.shape, lb=0, ub=big_number)
    ss = model.addMVar(X.shape, lb=0, ub=big_number)
    zs = model.addMVar(X.shape, vtype=GRB.BINARY)
    model.addConstr(ts - ss == X, name=f"{name}_constraint_1" if name else None)
    model.addConstr(ts <= M*zs, name=f"{name}_constraint_2" if name else None)
    model.addConstr(ss <= M*(1-zs), name=f"{name}_constraint_3" if name else None)
    return ts

def add_sage_constraint(model, A, X, lin_r_weight, lin_l_weight, lin_l_bias=None, lin_weight=None, lin_bias=None, project=False, name=None, aggr="mean"):
    if project:
        X = add_fc_constraint(model, X, lin_weight, lin_bias)
        X = add_relu_constraint(model, X)
        
    model.update()
    aggregated_features = model.addMVar(X.shape, lb=X.lb, ub=X.ub)

    if aggr=="mean":
        # aggregated_features[i][j] is the sum of all node i's neighbors' feature j divided by the number of node i's neighbors
        # Ensure gp.quicksum(A) does not have any zeros
        model.addConstr(aggregated_features*gp.quicksum(A)[:, np.newaxis] == A@X, name=f"{name}_averages_constraint" if name else None) # may need to transpose
    elif aggr=="sum":
        model.addConstr(aggregated_features == A@X, name=f"{name}_sum_constraint" if name else None)

    ts = model.addMVar((X.shape[0], lin_r_weight.shape[0]), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts == (X@lin_r_weight.T) + (aggregated_features@lin_l_weight.T) + np.expand_dims(lin_l_bias, 0), name=f"{name}_output_constraint" if name else None) 
    return ts

def global_add_pool(model, X, name=None):
    sums = model.addMVar((X.shape[1],), lb=-big_number, ub=big_number, name=name)
    model.addConstr(sums == gp.quicksum(X), name=f"{name}_constraint" if name else None)
    return sums[np.newaxis, :]

def global_mean_pool(model, X, name=None):
    averages = model.addMVar((X.shape[1],), lb=-big_number, ub=big_number, name=name)
    model.addConstr(averages == gp.quicksum(X)/X.shape[0], name=f"{name}_constraint" if name else None)
    return averages[np.newaxis, :]

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