import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product

M = 1000

def add_fc_constraint(model, X, W, b, name=None):
    ts = model.addMVar((1, W.shape[1]), lb=-float("inf"), ub=float("inf"), name=f"{name}_t" if name else None)
    model.addConstr(ts==X @ W + b)
    return ts

def add_gcn_constraint(model, A, X, W, b, name=None): # Unnormalized Adjacency Matrix
    ts = model.addMVar((A.shape[0], W.shape[1]), lb=-float("inf"), ub=float("inf"), name=f"{name}_t" if name else None)
    model.addConstr(ts==A @ X @ W + b)
    return ts

def add_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 1)

def add_relu_constraint(model, X):
    activations = model.addMVar(X.shape)
    for index in product(*[range(r) for r in X.shape]):
        model.addGenConstrMax(activations[index], [X[index].tolist()], constant=0)
    return activations

def add_sage_constraint(model, A, X, W1, W2, b, W3=None, name=None):
    feature_averages = model.addMVar((W2.shape[0], A.shape[0]))
    model.addConstr(gp.quicksum(A@X)==gp.quicksum((feature_averages@A).T)) # TODO: Something like this
    ts = model.addMVar((1, W1.shape[1]), lb=-float("inf"), ub=float("inf"), name=f"{name}_t" if name else None)

    model.addConstr(ts == W1@X+W2@feature_averages)

def global_add_pool(X, name="pool"):
    return gp.quicksum(X)

def global_mean_pool(X, name="pool"):
    return gp.quicksum(X)/X.shape[0]

if __name__ == "__main__":
    W1 = np.array([[1, 1, 1, 1, 1]])
    b1 = np.array([[0, -1, -2, -3, -4]])
    W2 = np.array([[3.2, -3, 1.4, 2.2, -1]]).T
    b2 = np.array([[-0.2]]).T
    target = 1.5

    m = gp.Model("MLP Inverse")
    x = m.addMVar((1, 1), lb=-float("inf"), ub=float("inf"), name="x")

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
    X = m.addMVar((3, 1), lb=-float("inf"), ub=float("inf"), name="x")

    node_embeddings = add_gcn_constraint(m, A, X, W1, b1)
    node_embeddings = add_relu_constraint(m, node_embeddings)
    # node_embeddings = add_relu_constraint(m, node_embeddings)
    graph_embedding = global_add_pool(node_embeddings)
    output = add_fc_constraint(m, graph_embedding, W2, b2)

    m.setObjective((output-1.5)*(output-1.5), GRB.MINIMIZE)
    m.optimize()

    print(X.X)
    print(A.X)