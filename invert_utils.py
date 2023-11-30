import gurobipy as gp
import numpy as np
from numpy.linalg import norm
from torch.nn import Linear, ReLU
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation
from torch_geometric.nn import SAGEConv


def invert_torch_layer(model, layer, **kwargs):
    if isinstance(layer, Linear):
        output = torch_fc_constraint(model, layer=layer, **kwargs)
    elif isinstance(layer, SAGEConv):
        output = torch_sage_constraint(model, layer=layer, **kwargs)
    elif isinstance(layer, MeanAggregation):
        output = global_mean_pool(model, **kwargs)
    elif isinstance(layer, SumAggregation):
        output = global_add_pool(model, **kwargs)
    elif isinstance(layer, ReLU):
        output = add_relu_constraint(model, **kwargs)
    else:
        raise NotImplementedError(f"layer type {layer} has no MIQCP analog")
    return output


def get_matmul_bounds(MVar, W):
    # Define the bounds of AW, where A is a matrix of decision variables and W is a matrix of fixed scalars
    lower_bounds = (
        (MVar.getAttr("lb") @ W.clip(min=0)) + (MVar.getAttr("ub") @ W.clip(max=0))
    ).squeeze()
    upper_bounds = (
        (MVar.getAttr("ub") @ W.clip(min=0)) + (MVar.getAttr("lb") @ W.clip(max=0))
    ).squeeze()
    assert np.less_equal(lower_bounds, upper_bounds).all()
    return lower_bounds, upper_bounds


def add_fc_constraint(model, X, W, b, name=None):
    model.update()
    lower_bounds, upper_bounds = get_matmul_bounds(X, W.T)
    ts = model.addMVar(
        (W.shape[0],),
        lb=lower_bounds + b,
        ub=upper_bounds + b,
        name=f"{name}_t" if name else None,
    )
    model.addConstr(
        ts == X @ W.T + b, name=f"{name}_output_constraint" if name else None
    )
    return ts[np.newaxis, :]


def add_gcn_constraint(model, A, X, W, b, name=None):  # Unnormalized Adjacency Matrix
    model.update()
    lower_bounds, upper_bounds = get_matmul_bounds(X, W.T)
    ts = model.addMVar(
        (A.shape[0], W.shape[1]),
        lb=lower_bounds + b,
        ub=upper_bounds + b,
        name=f"{name}_t" if name else None,
    )
    model.addConstr(
        ts == A @ X @ W + b, name=f"{name}_output_constraint" if name else None
    )
    return ts


def add_self_loops(model, A):
    # Ensure every node is connected to itself
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 1, name=f"self_loops_node_{i}")


def remove_self_loops(model, A):
    # Ensure no nodes are connected to themselves
    for i in range(A.shape[0]):
        model.addConstr(A[i][i] == 0, name=f"no_self_loops_node_{i}")


def force_undirected(model, A):
    # Constrain A to be symmetric
    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[1]):
            model.addConstr(A[i][j] == A[j][i], name=f"undirected_{i}_{j}")


# def force_connected(model, A):
#     # Enforce partial ordering on nodes to ensure connectivity
#     for i in range(A.shape[0]-1):
#         model.addConstr(gp.quicksum(A[i][j] + A[j][i] for j in range(i+1,A.shape[0])) >= 1, name=f"node_{i}_connected")


def force_connected(model, A):
    # Enforce partial ordering on nodes to ensure connectivity
    for i in range(1, A.shape[0]):
        model.addConstr(
            gp.quicksum(A[i][j] + A[j][i] for j in range(i)) >= 1,
            name=f"node_{i}_connected",
        )


def order_onehot_features(model, A, X):
    # Lexicographic ordering of one-hot node features
    # Combined with connected constraints: We're adding each next-biggest featured node and connecting it to the existing structure.
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            for k in range(X.shape[1]):
                model.addConstr(sum(X[i, :k]) >= sum(X[j, :k]))


# def add_relu_constraint(model, X, name=None):
#     # Returns a matrix of decision variables constrained to ReLU(X), where X is also a matrix of decision variables
#     model.update()
#     ts = model.addMVar(X.shape, lb=0, ub=X.getAttr("ub").clip(min=0), name=f"{name}_ts")
#     ss = model.addMVar(X.shape, lb=0, ub=(-X.getAttr("lb")).clip(min=0), name=f"{name}_ss")
#     model.update()
#     zs = model.addMVar(X.shape, vtype=GRB.BINARY, name=f"{name}_zs")
#     model.addConstr(ts - ss == X, name=f"{name}_constraint_1" if name else None)
#     model.addConstr(ts <= ts.getAttr("ub")*zs, name=f"{name}_constraint_2" if name else None)
#     model.addConstr(ss <= ss.getAttr("ub")*(1-zs), name=f"{name}_constraint_3" if name else None)
#     return ts


def add_relu_constraint(model, X, name=None, **kwargs):
    # Returns a matrix of decision variables constrained to ReLU(X), where X is also a matrix of decision variables
    model.update()
    ts = model.addMVar(X.shape, lb=0, ub=X.getAttr("ub").clip(min=0), name=f"{name}_ts")

    X_list = [x for x_row in X.tolist() for x in x_row]
    t_list = [t for t_row in ts.tolist() for t in t_row]

    model.update()

    for i, (x, t) in enumerate(zip(X_list, t_list)):
        model.addGenConstrMax(t, [x], constant=0, name=f"{name}_constraint_{i}")
    return ts


def add_sage_constraint(
    model,
    A,
    X,
    lin_r_weight,
    lin_l_weight,
    lin_l_bias,
    lin_weight=None,
    lin_bias=None,
    project=False,
    name=None,
    aggr="mean",
    **kwargs,
):
    # Returns the output of a GraphSAGE convolutional layer, see the implementation in PyTorch-Geometric for details about the parameters
    model.update()
    if project:
        X = add_fc_constraint(
            model, X, lin_weight, lin_bias, name=name + "projection_fc"
        )
        X = add_relu_constraint(model, X, name=name + "projection_relu")

    # Create decision variables to store the aggregated features of each node's neighborhood
    aggregated_features = model.addMVar(
        X.shape, lb=1, ub=-1, name=f"{name}_aggregated_features"
    )

    model.update()

    if aggr == "mean":
        # aggregated_features[i][j] is the sum of all node i's neighbors' feature j divided by the number of node i's neighbors
        # Ensure gp.quicksum(A) does not have any zeros
        aggregated_features.setAttr(
            "lb",
            np.repeat(X.getAttr("lb").mean(axis=0)[np.newaxis, :], X.shape[0], axis=0),
        )
        aggregated_features.setAttr(
            "ub",
            np.repeat(X.getAttr("ub").mean(axis=0)[np.newaxis, :], X.shape[0], axis=0),
        )
        model.addConstr(
            aggregated_features * gp.quicksum(A)[:, np.newaxis] == A @ X,
            name=f"{name}_averages_constraint" if name else None,
        )  # may need to transpose
    elif aggr == "sum":
        # aggregated_features[i][j] is the sum of all node i's neighbors' feature j
        aggregated_features.setAttr(
            "lb",
            np.repeat(X.getAttr("lb").sum(axis=0)[np.newaxis, :], X.shape[0], axis=0),
        )
        aggregated_features.setAttr(
            "ub",
            np.repeat(X.getAttr("ub").sum(axis=0)[np.newaxis, :], X.shape[0], axis=0),
        )
        model.addConstr(
            aggregated_features == A @ X,
            name=f"{name}_sum_constraint" if name else None,
        )

    model.update()

    # The bounds for the output of the layer will be the sums of the bounds of its addends
    first_lower_bounds, first_upper_bounds = get_matmul_bounds(X, lin_r_weight.T)
    second_lower_bounds, second_upper_bounds = get_matmul_bounds(
        aggregated_features, lin_l_weight.T
    )
    # If node features are categorical, we can tighten the bounds on the output.
    # if name == "Conv_0" and aggr=="mean" and model.getConstrByName("categorical_features[0]") is not None:
    #     print("Tightening bounds for first term in Conv_0 (mean) for categorical features")
    #     first_lower_bounds = np.maximum(first_lower_bounds, np.repeat(lin_r_weight.min(axis=1)[np.newaxis, :], X.shape[0], axis=0))
    #     first_upper_bounds = np.minimum(first_upper_bounds, np.repeat(lin_r_weight.max(axis=1)[np.newaxis, :], X.shape[0], axis=0))
    ts_lower_bounds = (
        first_lower_bounds + second_lower_bounds + np.expand_dims(lin_l_bias, 0)
    )
    ts_upper_bounds = (
        first_upper_bounds + second_upper_bounds + np.expand_dims(lin_l_bias, 0)
    )
    ts = model.addMVar(
        (X.shape[0], lin_r_weight.shape[0]),
        lb=ts_lower_bounds,
        ub=ts_upper_bounds,
        name=f"{name}_t" if name else None,
    )
    ts.setAttr("lb", ts_lower_bounds)
    ts.setAttr("ub", ts_upper_bounds)

    # Constrain outputs to correct values
    model.addConstr(
        ts
        == (X @ lin_r_weight.T)
        + (aggregated_features @ lin_l_weight.T)
        + np.expand_dims(lin_l_bias, 0),
        name=f"{name}_output_constraint" if name else None,
    )
    return ts


def global_add_pool(model, X, name=None, **kwargs):
    # Outputs variables constrained to the sum of node features element-wise
    model.update()
    sums = model.addMVar(
        (X.shape[1],),
        lb=X.getAttr("lb").sum(axis=0),
        ub=X.getAttr("ub").sum(axis=0),
        name=name,
    )
    model.addConstr(sums == gp.quicksum(X), name=f"{name}_constraint" if name else None)
    return sums[np.newaxis, :]


def global_mean_pool(model, X, name=None, **kwargs):
    # Outputs variables constrained to the mean of node features element-wise
    model.update()
    averages = model.addMVar(
        (X.shape[1],),
        lb=X.getAttr("lb").mean(axis=0),
        ub=X.getAttr("ub").mean(axis=0),
        name=name,
    )
    model.addConstr(
        averages == gp.quicksum(X) / X.shape[0],
        name=f"{name}_constraint" if name else None,
    )
    return averages[np.newaxis, :]


def torch_fc_constraint(model, X, layer, name=None, max_output=None, **kwargs):
    # Encodes a PyTorch Linear layer based on the input X
    weight = layer.get_parameter("weight").detach().numpy()
    bias = layer.get_parameter("bias").detach().numpy()
    if max_output is not None:
        weight = weight[max_output][np.newaxis, :]
        bias = np.atleast_2d(bias[max_output])

    return add_fc_constraint(model, X, W=weight, b=bias, name=name)


def torch_sage_constraint(model, A, X, layer, name=None, **kwargs):
    # Encodes a PyTorch-Geometric GraphSAGE convolutional layer object based on the input X and A
    # TODO: Cleanup
    lin_r_weight = layer.get_parameter("lin_r.weight").detach().numpy()
    lin_l_weight = layer.get_parameter("lin_l.weight").detach().numpy()
    lin_l_bias = layer.get_parameter("lin_l.bias").detach().numpy()
    lin_weight, lin_bias = None, None
    if layer.project and hasattr(layer, "lin"):
        lin_weight = layer.get_parameter("lin.weight").detach().numpy()
        lin_bias = layer.get_parameter("lin.bias").detach().numpy()
    return add_sage_constraint(
        model,
        A,
        X,
        lin_r_weight=lin_r_weight,
        lin_l_weight=lin_l_weight,
        lin_l_bias=lin_l_bias,
        lin_weight=lin_weight,
        lin_bias=lin_bias,
        project=layer.project,
        aggr=layer.aggr,
        name=name,
    )


def get_cosine_similarity(model, var, vec, name="cosine_similarity"):
    # Cosine Similarity between variable var and scalar vec
    cosine_similarity = model.addVar(lb=0, ub=1, name=name)
    var_magnitude = model.addVar(
        lb=0, ub=sum(var.getAttr("ub") ** 2) ** 0.5, name=f"{name}_magnitude"
    )  # Variables for embedding magnitude, intermediate value in calculation
    vec_magnitude = np.linalg.norm(vec)
    # m.addConstr(var_magnitude*var_magnitude == gp.quicksum(var*var), name=f"{name}_magnitude_constr")
    model.addGenConstrNorm(var_magnitude, var, which=2, name=f"{name}_magnitude_constr")
    model.addConstr(
        gp.quicksum(var * vec) == var_magnitude * vec_magnitude * cosine_similarity,
        name=f"{name}_constr",
    )  # u^Tv=|u||v|cos_sim(u,v)
    return cosine_similarity, lambda newvec: np.dot(newvec, vec) / (
        norm(newvec) * norm(vec)
    )


def get_l2_distance(model, var, vec, name="l2_distance"):
    ub = np.linalg.norm(
        np.maximum(
            var.getAttr("ub").clip(max=0).abs(),
            np.maximum(var.getAttr("ub").clip(min=0).abs()),
        )
    )
    l2_distance = model.addVar(lb=0, ub=ub, name=name)
    model.addConstr(
        l2_distance * l2_distance == gp.quicksum((vec - var) * (vec - var)),
        name=f"{name}_constr",
    )  # l2_dist(u,v)^2 = (u-v)^T(u-v)
    return l2_distance, lambda newvec: norm(newvec - vec)


def get_squared_l2_distance(model, var, vec, name="squared_l2_distance"):
    ub = np.linalg.norm(
        np.maximum(
            var.getAttr("ub").clip(max=0).abs(),
            np.maximum(var.getAttr("ub").clip(min=0).abs()),
        )
    )
    squared_l2_distance = model.addVar(lb=0, ub=ub, name=name)
    model.addConstr(
        squared_l2_distance >= gp.quicksum((vec - var) * (vec - var)),
        name=f"{name}_constr",
    )  # l2_dist(u,v)^2 = (u-v)^T(u-v)
    return squared_l2_distance, lambda newvec: norm(newvec - vec) ** 2