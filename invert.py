import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product

M = 10000

def add_fc_constraint(model, X, W, b, name=None, relu=False):
    if relu:
        ts = model.addMVar((1, W.shape[1]), lb=0, ub=float("inf"), name=f"{name}_t" if name else None)
        ss = model.addMVar((1, W.shape[1]), lb=0, ub=float("inf"), name=f"{name}_s" if name else None)
        zs = model.addMVar((1, W.shape[1]), vtype=GRB.BINARY, name=f"{name}_z" if name else None)

        model.addConstr(ts-ss==X @ W + b)
        model.addConstr(ts <= M*zs)
        model.addConstr(ss <= M*(1-zs))
    else:
        ts = model.addMVar((1, W.shape[1]), lb=-float("inf"), ub=float("inf"), name=f"{name}_t" if name else None)

        model.addConstr(ts==X @ W + b)

    return ts

def add_gcn_constraint(model, A, X, W, b, name=None, relu=False):
    if relu:
        ts = model.addMVar((1, W.shape[1]), lb=0, ub=float("inf"), name=f"{name}_t" if name else None)
        ss = model.addMVar((1, W.shape[1]), lb=0, ub=float("inf"), name=f"{name}_s" if name else None)
        zs = model.addMVar((1, W.shape[1]), vtype=GRB.BINARY, name=f"{name}_z" if name else None)

        model.addConstr(ts-ss==A @ X @ W + b)
        model.addConstr(ts <= M*zs)
        model.addConstr(ss <= M*(1-zs))

    else:
        ts = model.addMVar((1, W.shape[1]), lb=-float("inf"), ub=float("inf"), name=f"{name}_t" if name else None)
        model.addConstr(ts==A @ X @ W + b)

    return ts

# def add_relu_constraint(model, X):
#     activations = model.addMVar(X.shape)
#     print("AH", X.shape)
#     for index in product(range):
#             model.addGenConstrMax(activations[i][j], [X[i][j].tolist()], constant=0)
#     return activations

# def add_gcn_constraint(model, A, X, W, b, name=None, add_self_loops=True):
#     num_nodes = A.shape[0]
#     # Xp = model.addMVar(X.shape, lb=0, ub=100)
#     # Xp = add_fc_constraint(model, X, W, b)
#     Xp = X
#     # Get incoming messages for each node and sum
#     Xos = []
#     for i in range(num_nodes):
#         # print(A.shape, Xp.shape)
#         Xp = gp.quicksum(A[i][j]*Xp[j] for j in range(num_nodes))
#         I = np.ones(X.shape[1:])
#         # if add_self_loops:
#         #     print(I-A[i][i], X[i])
#         #     Xp += (I-A[i][i])*X[i]
#         print(Xp[i].shape, W.shape, b.shape)
#         Xo = add_fc_constraint(model, Xp[i], W, b)
#         Xos.append(Xo)
#     return Xo

def global_add_pool(model, A, X, name="pool"):
    return gp.quicksum(X)

def global_mean_pool(model, A, X, name="pool"):
    return gp.quicksum(X)/A.shape[0]

W1 = np.array([[1, 1, 1, 1, 1]])
b1 = np.array([[0, -1, -2, -3, -4]])
W2 = np.array([[3.2, -3, 1.4, 2.2, -1]]).T
b2 = np.array([[-0.2]]).T
target = 1.5

m = gp.Model("MLP Inverse")
x = m.addMVar((1, 1), lb=-float("inf"), ub=float("inf"), name="x")

hidden_activations = add_fc_constraint(m, x, W1, b1, "hidden", relu=True)
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
X = m.addMVar((3, 1), lb=-float("inf"), ub=float("inf"), name="x")

node_embeddings = add_gcn_constraint(m, A, X, W1, b1, relu=True)
# node_embeddings = add_relu_constraint(m, node_embeddings)
graph_embedding = global_add_pool(m, A, node_embeddings)
output = add_fc_constraint(m, graph_embedding, W2, b2)

m.setObjective((output-1.5)*(output-1.5), GRB.MINIMIZE)
m.optimize()

print(X.X)
print(A.X)