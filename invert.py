import gurobipy as gp
from gurobipy import GRB
import numpy as np

M = 128
big_number = 128

def add_fc_constraint(model, X, W, b, name=None):
    ts = model.addMVar((W.shape[0],), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts==X@W.T + b, name=f"{name}_output_constraint" if name else None)
    return ts

def add_gcn_constraint(model, A, X, W, b, name=None): # Unnormalized Adjacency Matrix
    ts = model.addMVar((A.shape[0], W.shape[1]), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts == A @ X @ W + b, name=f"{name}_output_constraint" if name else None)
    return ts

def add_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 1)

def force_undirected(model, A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            model.addConstr(A[i][j] == A[j][i])

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
    return sums

def global_mean_pool(model, X, name=None):
    averages = model.addMVar((X.shape[1],), lb=-big_number, ub=big_number, name=name)
    model.addConstr(averages == gp.quicksum(X)/X.shape[0], name=f"{name}_constraint" if name else None)
    return averages