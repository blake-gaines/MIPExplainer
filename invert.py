import gurobipy as gp
from gurobipy import GRB
import numpy as np

def add_fc_constraint(model, X, W, b, name):
    ts = model.addMVar((W.shape[0], 1), lb=0, ub=100, name=f"{name}_t" if name else None)
    ss = model.addMVar((W.shape[0], 1), lb=0, ub=100, name=f"{name}_s" if name else None)
    zs = model.addMVar((W.shape[0], 1), vtype=GRB.BINARY, name=f"{name}_z" if name else None)

    for j in range(W.shape[0]):
        c = model.addConstr(ts[j]-ss[j]==gp.quicksum(W[j, :] @ X + b[j]))

    m.addConstr(ts <= 10000*zs)
    m.addConstr(ss <= 10000*(1-zs))

    return ts


W1 = np.array([[1, 1, 1, 1, 1]]).T
b1 = np.array([[0, -1, -2, -3, -4]]).T
W2 = np.array([[3.2, -3, 1.4, 2.2, -1]])
b2 = np.array([[-0.2]])
target = 1.5

m = gp.Model("GNN Inverse")
x = m.addMVar((1, 1), lb=-float("inf"), ub=float("inf"), name="x")

hidden_activations = add_fc_constraint(m, x, W1, b1, "hidden")
output = add_fc_constraint(m, hidden_activations, W2, b2, "output")

m.setObjective((output-1.5)*(output-1.5), GRB.MINIMIZE)
m.optimize()

print(x.X)

# # print(output.X)
# for v in m.getVars():
#     print('%s %g' % (v.VarName, v.X))

# print('Obj: %g' % m.ObjVal)


