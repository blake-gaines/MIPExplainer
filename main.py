import torch
from gnn import GNN
import gurobipy as gp
from gurobipy import GRB
from invert import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch

def torch_fc_constraint(model, X, nn, name):
    return add_fc_constraint(model, X, W=nn.get_parameter(f"{name}.weight").detach().numpy(), b=nn.get_parameter(f"{name}.bias").detach().numpy())

def torch_sage_constraint(model, A, X, nn, name):
    W1 = nn.get_parameter(f"{name}.lin_r.weight").detach().numpy()
    W2 = nn.get_parameter(f"{name}.lin_l.weight").detach().numpy()
    # b1 = nn.get_parameter(f"{name}.lin_r.bias").detach().numpy()
    b2 = nn.get_parameter(f"{name}.lin_l.bias").detach().numpy()
    return add_sage_constraint(model, A, X, W1=W1, W2=W2, b2=b2)

def to_batch(X, A):
    g = dict()
    g["edge_index"], g["edge_weight"] = dense_to_sparse(A)
    g["x"] = X
    return Batch.from_data_list([Data(**g)])

model_path = "models/MUTAG_model.pth"
nn = torch.load(model_path)
nn.eval()

print('\n'.join(f"{t[0]}:".ljust(20)+f"{t[1].shape}" for t in nn.named_parameters()))

m = gp.Model("GNN Inverse")
A = m.addMVar((10, 10), vtype=GRB.BINARY, name="x")
X = m.addMVar((10, 7), lb=-float("inf"), ub=float("inf"), name="x")



hidden1 = torch_sage_constraint(m, A, X, nn, "conv1")
hidden2 = torch_sage_constraint(m, A, hidden1, nn, "conv2")
hidden3 = torch_sage_constraint(m, A, hidden2, nn, "conv3")
hidden4 = global_mean_pool(m, hidden3)
hidden5 = torch_fc_constraint(m, hidden4, nn, "lin1")
hidden6 = torch_fc_constraint(m, hidden5, nn, "lin2")
output = torch_fc_constraint(m, hidden6, nn, "lin3")

m.optimize()

# print("NN output given X", nn(x=torch.Tensor(X.X), edge_index=dense_to_sparse(torch.Tensor(A.X.astype(np.int64)))[0], batch=torch.zeros(10,dtype=torch.int64)))
batch = to_batch(torch.Tensor(X.X), torch.Tensor(A.X.astype(int)))
print("NN output given X", nn.get_embedding_outputs(batch)[1].detach().numpy())
print("NN output given hidden4", nn.classify(torch.Tensor(hidden4.X)))
print("predicted output", output.X)

# a = nn.conv1(torch.Tensor(X.X), dense_to_sparse(torch.Tensor(A.X.astype(int)))[0]).detach().numpy()
# b = hidden.X.astype(np.float32)
# print((a==b).astype(int))
# idx = np.nonzero(1-(a==b).astype(int))
# print(a[idx], b[idx])
# print("ADJ MATRIX", A.X)
# print("X", X.X)