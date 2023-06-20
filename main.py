import torch
from gnn import GNN
import gurobipy as gp
from gurobipy import GRB
from invert import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch

def torch_fc_constraint(model, X, layer, name=None):
    return add_fc_constraint(model, X, W=layer.get_parameter(f"weight").detach().numpy(), b=layer.get_parameter(f"bias").detach().numpy(), name=name)

def torch_sage_constraint(model, A, X, layer, name=None):
    lin_r_weight = layer.get_parameter(f"lin_r.weight").detach().numpy()
    lin_l_weight = layer.get_parameter(f"lin_l.weight").detach().numpy()
    # b1 = nn.get_parameter(f"{name}.lin_r.bias").detach().numpy()
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

model_path = "models/MUTAG_model.pth"
nn = torch.load(model_path)
nn.eval()
nn.double()

print('\n'.join(f"{t[0]}:".ljust(20)+f"{t[1].shape}" for t in nn.named_parameters()))

m = gp.Model("GNN Inverse")

# m.setParam("MIQCPMethod", 1) 
A = m.addMVar((10, 10), vtype=GRB.BINARY, name="A")
X = m.addMVar((10, 7), vtype=GRB.BINARY, name="x")
# X = m.addMVar((10, 7), lb=-float("inf"), ub=float("inf"), name="X")
m.addConstr(gp.quicksum(X) <= 1) # REMOVE for non-categorical features
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

# print("Tuning")
# # m.setParam("TuneTimeLimit", 120)
# m.tune()
# for i in range(m.tuneResultCount):
#     m.getTuneResult(i)
#     m.write('tune'+str(i)+'.prm')
# print("Done Tuning")

X.Start = np.zeros(X.shape)
batch = to_batch(torch.zeros(X.shape).double(), torch.zeros(A.shape).double())
all_outputs = nn.get_all_layer_outputs(batch)
assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
m.params.StartNumber = 0
for var, output in zip(output_vars, all_outputs):
    output = output.detach().numpy().squeeze()
    assert var.shape == output.shape
    var.Start = output

# m.addConstr(X == np.zeros(X.shape))
# batch = to_batch(torch.zeros(X.shape).double(), torch.zeros(A.shape).double())
# all_outputs = nn.get_all_layer_outputs(batch)
# assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
# for var, output in zip(output_vars, all_outputs):
#     output = output.detach().numpy().squeeze()
#     assert var.shape == output.shape
#     m.addConstr(var == output)

m.optimize()

if m.Status >= 3:
    m.computeIIS()
    m.write("model.ilp")
else:
    # print("NN output given X", nn(x=torch.Tensor(X.X), edge_index=dense_to_sparse(torch.Tensor(A.X.astype(np.int64)))[0], batch=torch.zeros(10,dtype=torch.int64)))
    batch = to_batch(torch.Tensor(X.X).double(), torch.Tensor(A.X.astype(int)).double())
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