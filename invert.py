import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product

M = 1000
big_number = 1000

def add_fc_constraint(model, X, W, b, name=None):
    ts = model.addMVar((W.shape[0],), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    print(W.shape, X.shape, b.shape, (W@X).shape, (X@W.T).shape)
    model.addConstr(ts==X@W.T + b)
    return ts

def add_gcn_constraint(model, A, X, W, b, name=None): # Unnormalized Adjacency Matrix
    ts = model.addMVar((A.shape[0], W.shape[1]), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts == A @ X @ W + b)
    return ts

def add_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 1)

def add_relu_constraint(model, X):
    activations = model.addMVar(X.shape)
    for index in product(*[range(r) for r in X.shape]):
        model.addGenConstrMax(activations[index], [X[index].tolist()], constant=0)
    return activations

def add_sage_constraint(model, A, X, W1, W2, b2=None, W3=None, b3=None, project=False, name=None): #lin_r has W1, lin_l has W2
    feature_averages = model.addMVar((A.shape[0], X.shape[1]))
    model.addConstr(gp.quicksum(A@X)==gp.quicksum((A@feature_averages))) # TODO: Something like this
    ts = model.addMVar((X.shape[0], W1.shape[0]), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts == (X@W1.T) + (feature_averages@W2.T) + np.expand_dims(b2, 0)) 
    # model.addConstr(ts == (X@W1.T) + (feature_averages@W2.T)) 
    return ts

def global_add_pool(X, name="pool"):
    return gp.quicksum(X)

# def global_mean_pool(X, name="pool"):
#     return gp.quicksum(X)/X.shape[0]
def global_mean_pool(model, X, name="pool"):
    averages = model.addMVar((X.shape[1],))
    model.addConstr(averages == gp.quicksum(X)/X.shape[0])
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