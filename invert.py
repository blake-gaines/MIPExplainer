import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product

M = 100
big_number = float("inf")

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

def add_sage_constraint(model, A, X, lin_r_weight, lin_l_weight, lin_l_bias=None, lin_weight=None, lin_bias=None, project=False, name=None, aggr="mean"): #lin_r_weight has lin_r_weight, lin_l_weight has lin_l_weight
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

if __name__ == "__main__":
    W1 = np.array([[1, 1, 1, 1, 1]])
    b1 = np.array([[0, -1, -2, -3, -4]])
    W2 = np.array([[3.2, -3, 1.4, 2.2, -1]]).T
    b2 = np.array([[-0.2]]).T
    target = 1.5

    m = gp.Model("MLP Inverse")
    x = m.addMVar((1, 1), lb=-big_number, ub=big_number, name="x")

    hidden_activations = add_fc_constraint(m, x, W1, b1, "hidden")
    hidden_activations = add_relu_constraint(m, hidden_activations)
    # hidden_activations = add_relu_constraint(m, hidden_activations)
    output = add_fc_constraint(m, hidden_activations, W2, b2, "output")

    m.setObjective((output-1.5)*(output-1.5), GRB.MINIMIZE)
    m.optimize()

    m.update()
    print(m.display())

    print(x.X)

    m = gp.Model("GCN Inverse")
    m.params.LogFile = "./log.txt"
    m.params.NonConvex = 0
    m.params.MIQCPMethod = 1
    A = m.addMVar((3, 3), vtype=GRB.BINARY, name="x")
    X = m.addMVar((3, 1), lb=-big_number, ub=big_number, name="x")

    node_embeddings = add_gcn_constraint(m, A, X, W1, b1)
    node_embeddings = add_relu_constraint(m, node_embeddings)
    # node_embeddings = add_relu_constraint(m, node_embeddings)
    graph_embedding = global_add_pool(node_embeddings)
    output = add_fc_constraint(m, graph_embedding, W2, b2)

    m.setObjective((output-1.5)*(output-1.5), GRB.MINIMIZE)
    m.optimize()

    print(X.X)
    print(A.X)