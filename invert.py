import gurobipy as gp
from gurobipy import GRB
import numpy as np
from utils import get_matmul_bounds

M = 256
big_number = 256

def add_fc_constraint(model, X, W, b, name=None):
    model.update()
    # lower_bounds = ((X.getAttr("lb") @ W.T.clip(min=0)) + (X.getAttr("ub") @ W.T.clip(max=0))).squeeze()
    # upper_bounds = ((X.getAttr("ub") @ W.T.clip(min=0)) + (X.getAttr("lb") @ W.T.clip(max=0))).squeeze()
    lower_bounds, upper_bounds = get_matmul_bounds(X, W.T)
    # lower_bounds, upper_bounds = -big_number, big_number
    # lower_bounds, upper_bounds = lower_bounds.clip(min=-big_number, max=big_number), upper_bounds.clip(min=-big_number, max=big_number)
    # print(lower_bounds, upper_bounds)
    ts = model.addMVar((W.shape[0],), lb=lower_bounds, ub=upper_bounds, name=f"{name}_t" if name else None)
    model.addConstr(ts==X@W.T + b, name=f"{name}_output_constraint" if name else None)
    return ts[np.newaxis, :]

def add_gcn_constraint(model, A, X, W, b, name=None): # Unnormalized Adjacency Matrix
    ts = model.addMVar((A.shape[0], W.shape[1]), lb=-big_number, ub=big_number, name=f"{name}_t" if name else None)
    model.addConstr(ts == A @ X @ W + b, name=f"{name}_output_constraint" if name else None)
    return ts

def add_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 1, name=f"self_loops_node_{i}")

def remove_self_loops(model, A):
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 0, name=f"no_self_loops_node_{i}")

def force_undirected(model, A):
    for i in range(A.shape[0]):
        for j in range(i, A.shape[1]):
            model.addConstr(A[i][j] == A[j][i], name=f"undirected_{i}_{j}")

def force_connected(model, A):
    for i in range(A.shape[0]-1):
        model.addConstr(gp.quicksum(A[i][j] + A[j][i] for j in range(i+1,A.shape[0])) >= 1, name=f"node_{i}_connected")

# def add_relu_constraint(model, X, name=None):
#     activations = model.addMVar(X.shape, name=name)
#     for index in product(*[range(r) for r in X.shape]):
#         model.addGenConstrMax(activations[index], [X[index].tolist()], constant=0, name=f"{name}_relu_constraint_{index}".replace(" ","") if name else None)
#     return activations

def add_relu_constraint(model, X, name=None):
    model.update()
    ts = model.addMVar(X.shape, lb=0, ub=X.getAttr("ub").clip(min=0), name=f"{name}_ts") #.clip(min=0, max=big_number)
    ss = model.addMVar(X.shape, lb=0, ub=(-X.getAttr("lb")).clip(min=0), name=f"{name}_ss") #.clip(min=0, max=big_number)
    model.update()
    zs = model.addMVar(X.shape, vtype=GRB.BINARY, name=f"{name}_zs")
    model.addConstr(ts - ss == X, name=f"{name}_constraint_1" if name else None)
    model.addConstr(ts <= ts.getAttr("ub")*zs, name=f"{name}_constraint_2" if name else None)
    model.addConstr(ss <= ss.getAttr("ub")*(1-zs), name=f"{name}_constraint_3" if name else None)
    return ts

def add_sage_constraint(model, A, X, lin_r_weight, lin_l_weight, lin_l_bias, lin_weight=None, lin_bias=None, project=False, name=None, aggr="mean"):
    model.update()
    if project:
        X = add_fc_constraint(model, X, lin_weight, lin_bias, name=name+"projection_fc")
        X = add_relu_constraint(model, X, name=name+"projection_relu")
        
    aggregated_features = model.addMVar(X.shape, lb=-big_number, ub=big_number, name=f"{name}_aggregated_features")

    if aggr=="mean":
        # aggregated_features[i][j] is the sum of all node i's neighbors' feature j divided by the number of node i's neighbors
        # Ensure gp.quicksum(A) does not have any zeros
        aggregated_features.setAttr("lb", np.repeat(X.getAttr("lb").mean(axis=0)[np.newaxis, :], X.shape[0], axis=0))#.clip(min=-big_number, max=big_number))
        aggregated_features.setAttr("ub", np.repeat(X.getAttr("ub").mean(axis=0)[np.newaxis, :], X.shape[0], axis=0))#.clip(min=-big_number, max=big_number))
        model.addConstr(aggregated_features*gp.quicksum(A)[:, np.newaxis] == A@X, name=f"{name}_averages_constraint" if name else None) # may need to transpose
    elif aggr=="sum":
        aggregated_features.setAttr("lb", np.repeat(X.getAttr("lb").sum(axis=0)[np.newaxis, :], X.shape[0], axis=0) )#.clip(min=-big_number, max=big_number))
        aggregated_features.setAttr("ub", np.repeat(X.getAttr("ub").sum(axis=0)[np.newaxis, :], X.shape[0], axis=0) )#.clip(min=-big_number, max=big_number))
        model.addConstr(aggregated_features == A@X, name=f"{name}_sum_constraint" if name else None)

    model.update()

    first_lower_bounds, first_upper_bounds = get_matmul_bounds(X, lin_r_weight.T)
    second_lower_bounds, second_upper_bounds = get_matmul_bounds(aggregated_features, lin_l_weight.T)

    if name == "Conv_0" and aggr=="mean" and model.getConstrByName("categorical_features[0]") is not None:
        print("Tightening bounds for first term in Conv_0 (mean) for categorical features")
        first_lower_bounds = np.maximum(first_lower_bounds, np.repeat(lin_r_weight.min(axis=1)[np.newaxis, :], X.shape[0], axis=0))
        first_upper_bounds = np.minimum(first_upper_bounds, np.repeat(lin_r_weight.max(axis=1)[np.newaxis, :], X.shape[0], axis=0))

    ts_lower_bounds = (first_lower_bounds + second_lower_bounds + np.expand_dims(lin_l_bias, 0))#.clip(min=-big_number, max=big_number)
    ts_upper_bounds = (first_upper_bounds + second_upper_bounds + np.expand_dims(lin_l_bias, 0))#.clip(min=-big_number, max=big_number)
    ts = model.addMVar((X.shape[0], lin_r_weight.shape[0]), lb=ts_lower_bounds, ub=ts_upper_bounds, name=f"{name}_t" if name else None)
    ts.setAttr("lb", ts_lower_bounds)
    ts.setAttr("ub", ts_upper_bounds)
    model.addConstr(ts == (X@lin_r_weight.T) + (aggregated_features@lin_l_weight.T) + np.expand_dims(lin_l_bias, 0), name=f"{name}_output_constraint" if name else None) 
    return ts

def global_add_pool(model, X, name=None):
    sums = model.addMVar((X.shape[1],), lb=-big_number, ub=big_number, name=name)
    model.addConstr(sums == gp.quicksum(X), name=f"{name}_constraint" if name else None)
    return sums[np.newaxis, :]

def global_mean_pool(model, X, name=None):
    model.update()
    averages = model.addMVar((X.shape[1],), lb=-big_number, ub=big_number, name=name)
    averages.setAttr("lb", (X.getAttr("lb").sum(axis=0)/X.shape[0]))#.clip(min=-big_number, max=big_number))
    averages.setAttr("ub", (X.getAttr("ub").sum(axis=0)/X.shape[0]))#.clip(min=-big_number, max=big_number))
    model.addConstr(averages == gp.quicksum(X)/X.shape[0], name=f"{name}_constraint" if name else None)
    return averages[np.newaxis, :]

def torch_fc_constraint(model, X, layer, name=None, max_output = None):
    weight = layer.get_parameter(f"weight").detach().numpy()
    bias = layer.get_parameter(f"bias").detach().numpy()
    if max_output is not None:
            weight = weight[max_output][np.newaxis, :] 
            bias = np.atleast_2d(bias[max_output]) 

    return add_fc_constraint(model, X, W=weight, b=bias, name=name)

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