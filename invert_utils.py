import gurobipy as gp
import numpy as np
from numpy.linalg import norm
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Flatten
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation
from torch_geometric.nn import SAGEConv
from math import floor
from tqdm.autonotebook import tqdm
from torch_geometric.utils import to_networkx
from networkx import relabel_nodes
from torch import Tensor
from gurobipy import GRB
from functools import cmp_to_key


def invert_torch_layer(model, layer, **kwargs):
    if isinstance(layer, Linear):
        return torch_fc_constraint(model, layer=layer, **kwargs)
    elif isinstance(layer, SAGEConv):
        return torch_sage_constraint(model, layer=layer, **kwargs)
    elif isinstance(layer, MeanAggregation):
        return global_mean_pool(model, **kwargs)
    elif isinstance(layer, SumAggregation):
        return global_add_pool(model, **kwargs)
    elif isinstance(layer, ReLU):
        return add_relu_constraint(model, **kwargs)
    elif isinstance(layer, Conv2d):
        return add_torch_conv2d_constraint(model, layer, **kwargs)
    elif isinstance(layer, MaxPool2d):
        return add_torch_maxpool2d_constraint(model, layer, **kwargs)
    elif isinstance(layer, Flatten):
        return flatten(**kwargs)
    else:
        raise NotImplementedError(f"layer type {layer} has no MIQCP analog")


def get_matmul_bounds(V, W):
    # Define the bounds of AW, where A is a matrix of decision variables and W is a matrix of fixed scalars
    lower_bounds = (V.getAttr("lb") @ W.clip(min=0)) + (V.getAttr("ub") @ W.clip(max=0))
    upper_bounds = (V.getAttr("ub") @ W.clip(min=0)) + (V.getAttr("lb") @ W.clip(max=0))
    if not np.less_equal(lower_bounds, upper_bounds).all():
        breakpoint()
    assert np.less_equal(lower_bounds, upper_bounds).all()
    return lower_bounds, upper_bounds


def add_fc_constraint(model, X, W, b, name=None):
    model.update()
    lower_bounds, upper_bounds = get_matmul_bounds(X, W.T)
    ts = model.addMVar(
        (X.shape[0], W.shape[0]),
        lb=lower_bounds + b,
        ub=upper_bounds + b,
        name=f"{name}_t" if name else None,
    )
    model.addConstr(ts == X @ W.T + b, name=f"{name}_output_constraint" if name else None)
    return ts


def flatten(X, name=None, **kwargs):
    return X.reshape((1, -1), order="C")  # TODO: Check order


def add_torch_maxpool2d_constraint(model, layer, X, name=None):
    model.update()

    N = X.shape[0]
    C = X.shape[-3]
    Hout = floor(((X.shape[-2] + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) / layer.stride) + 1)
    Wout = floor(((X.shape[-1] + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) / layer.stride) + 1)

    ts = model.addMVar(
        (
            N,
            C,
            Hout,
            Wout,
        ),
        lb=-float("inf"),
        ub=float("inf"),
    )

    stride = layer.stride

    for channel in tqdm(range(C), desc=name):
        for i in range(0, Hout):
            for j in range(0, Wout):
                a, b, c, d = (
                    i * stride,
                    i * stride + layer.kernel_size,
                    j * stride,
                    j * stride + layer.kernel_size,
                )
                model.addGenConstrMax(
                    ts[:, channel, i, j],
                    X[:, channel, a:b, c:d].reshape(-1).tolist(),
                    name=f"{name}_max_constr_{channel}_{i}_{j}" if name else None,  # TODO: Broken for N>1
                )
                ts[:, channel, i, j].setAttr(
                    "lb",
                    X[:, channel, a:b, c:d].getAttr("lb").max(),
                )
                ts[:, channel, i, j].setAttr(
                    "ub",
                    X[:, channel, a:b, c:d].getAttr("ub").max(),
                )

    return ts


def add_torch_conv2d_constraint(model, layer, X, name=None):
    model.update()

    weight = layer.weight.cpu().detach().numpy()
    bias_vector = layer.bias.cpu().detach().numpy()

    N = X.shape[0]
    Cout = weight.shape[0]
    Hout = floor(
        ((X.shape[-2] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0])
        + 1
    )
    Wout = floor(
        ((X.shape[-1] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1])
        + 1
    )

    ts = model.addMVar(
        (
            N,
            Cout,
            Hout,
            Wout,
        ),
        lb=-float("inf"),
        ub=float("inf"),
        name=f"{name}_t" if name else None,
    )

    model.update()

    X_array = X  # np.array(X.tolist())

    assert (
        weight[0][np.newaxis, :].shape == X_array[:, :, 0 : 0 + weight[0].shape[-2], 0 : 0 + weight[0].shape[-1]].shape
    ), (
        weight[0][np.newaxis, :].shape,
        X_array[:, :, 0 : 0 + weight[0].shape[-2], 0 : 0 + weight[0].shape[-1]].shape,
    )

    # TODO: Parallelize
    for kernel_index in tqdm(range(Cout), desc=name):
        kernel = weight[kernel_index]
        bias = bias_vector[kernel_index]
        ## TODO: USE QUICKSUM DUDE
        for i in range(0, Hout):
            for j in range(0, Wout):
                model.addConstr(
                    ts[:, kernel_index, i, j]
                    == (
                        kernel[np.newaxis, :]
                        * X_array[
                            :,
                            :,
                            i : i + kernel.shape[-2],
                            j : j + kernel.shape[-1],
                        ]
                    ).sum()  # TODO: Broken for N>1
                    + bias,
                    name=f"{name}_output_constraint_{kernel_index}_{i}_{j}" if name else None,
                )
                ts[:, kernel_index, i, j].setAttr(
                    "lb",
                    (
                        kernel.clip(min=0)[np.newaxis, :]
                        * X_array[:, :, i : i + kernel.shape[-2], j : j + kernel.shape[-1]].getAttr("lb")
                        + kernel.clip(max=0)[np.newaxis, :]
                        * X_array[:, :, i : i + kernel.shape[-2], j : j + kernel.shape[-1]].getAttr("ub")
                    )
                    .reshape((N, -1))
                    .sum(axis=1)
                    + bias,
                    # TODO: Check Axis
                )
                ts[:, kernel_index, i, j].setAttr(
                    "ub",
                    (
                        kernel.clip(min=0)[np.newaxis, :]
                        * X_array[:, :, i : i + kernel.shape[-2], j : j + kernel.shape[-1]].getAttr("ub")
                        + kernel.clip(max=0)[np.newaxis, :]
                        * X_array[:, :, i : i + kernel.shape[-2], j : j + kernel.shape[-1]].getAttr("lb")
                    )
                    .reshape((N, -1))
                    .sum(axis=1)
                    + bias,
                )
    model.update()
    return ts


def add_gcn_constraint(model, A, X, W, b, name=None):  # Unnormalized Adjacency Matrix
    model.update()
    lower_bounds, upper_bounds = get_matmul_bounds(X, W.T)
    ts = model.addMVar(
        (A.shape[0], W.shape[1]),
        lb=lower_bounds + b,
        ub=upper_bounds + b,
        name=f"{name}_t" if name else None,
    )
    model.addConstr(ts == A @ X @ W + b, name=f"{name}_output_constraint" if name else None)
    return ts


def add_self_loops(model, A):
    # Ensure every node is connected to itself
    for i in range(A.shape[0]):
        # model.addConstr(A[i][i] == 1, name=f"self_loops_node_{i}")
        A[i][i].setAttr("lb", 1)
        A[i][i].setAttr("ub", 1)


def remove_self_loops(model, A):
    # Ensure no nodes are connected to themselves
    for i in range(A.shape[0]):
        # model.addConstr(A[i][i] == 0, name=f"no_self_loops_node_{i}")
        A[i][i].setAttr("lb", 0)
        A[i][i].setAttr("ub", 0)


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
    # Combined with connected constraints: Ensure no lower-indexed node with the same parent has a higher feature value
    # Ordering equivalent to BFS with high-feature nodes prioritized
    # Whenever A[i,j]=1, if A[i,k]=1 for k<j, then X[k] must be lexicographically less than or equal to X[j]
    model.update()
    N = X.shape[0]
    F = X.shape[1]
    for parent in range(N - 2):
        for prev_child in range(parent + 1, X.shape[0] - 1):
            for next_child in range(prev_child + 1, N):
                both_connected = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"both_connected_{parent}_{prev_child}_{next_child}",
                )
                model.addConstr(
                    both_connected == gp.and_(A[parent, prev_child], A[parent, next_child]),
                    name=f"both_connected_constr_{parent}_{prev_child}_{next_child}",
                )
                model.addConstr(
                    gp.quicksum((2 ** (F - f - 1)) * X[prev_child, f] * both_connected for f in range(F))
                    <= gp.quicksum((2 ** (F - f - 1)) * X[next_child, f] * A[parent, next_child] for f in range(F)),
                    name=f"one_hot_order_{parent}_{prev_child}_{next_child}",
                )


def lex_adj_matrix(model, A):
    # Lexicographic ordering of adjacency matrix
    # WARNING: Unclear how this restricts the space of graphs, solution may not be optimal
    model.update()
    powers = 2 ** np.arange(A.shape[0])
    for i in range(A.shape[0] - 1):
        print(f"lex_{i}_{i + 1}: {A[i] @ powers} >= {A[i + 1] @ powers}")
        model.addConstr(A[i] @ powers >= A[i + 1] @ powers, name=f"lex_{i}_{i + 1}")


def s1_constraint(model, A):
    model.update()
    for i in range(1, A.shape[0]):
        model.addConstr(
            gp.quicksum(A[j][i] for j in range(i)) >= 1,
            name=f"node_{i}_connected",
        )


def s2_constraint(model, X):
    # From https://arxiv.org/pdf/2305.09420.pdf
    model.update()
    N = X.shape[0]
    F = X.shape[1]
    for v in range(1, N):
        model.addConstr(
            gp.quicksum((2 ** (F - f - 1)) * X[0, f] for f in range(F))
            >= gp.quicksum((2 ** (F - f - 1)) * X[v, f] for f in range(F)),
            name=f"s2_{v}",
        )


def s3_constraint(model, A):
    # From https://arxiv.org/pdf/2305.09420.pdf
    model.update()
    N = A.shape[0]
    for v in range(1, N - 1):
        model.addConstr(
            gp.quicksum((2 ** (N - u - 1)) * A[u, v] for u in range(N) if u != v and u != v + 1)
            >= gp.quicksum((2 ** (N - u - 1)) * A[u, v + 1] for u in range(N) if u != v and u != v + 1),
            name=f"s3_{v}",
        )


def get_one_hot_features(model, num_nodes, num_node_features, name="X"):
    # One-hot node features
    X = model.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name=name)
    model.addConstr(gp.quicksum(X.T) == 1, name="categorical_features")
    return X


def get_node_degree_features(model, num_nodes, num_node_features, A, name="X"):
    # Node degree node features
    X = model.addMVar((num_nodes, num_node_features), lb=0, ub=num_nodes, name=name, vtype=GRB.INTEGER)
    model.addConstr(X == gp.quicksum(A)[:, np.newaxis], name="features_are_node_degrees")
    return X


def get_constant_features(model, shape, constant, name="X", vtype=GRB.CONTINUOUS):
    # Constant node features
    # "Constant" argument could be a scalar or a numpy array
    X = model.addMVar(
        shape,
        # lb=constant,
        # ub=constant,
        name=name,
        vtype=vtype,
    )
    model.addConstr(X == constant, name="features_are_constant")
    return X


# def add_relu_constraint(model, X, name=None, **kwargs):
#     # Returns a matrix of decision variables constrained to ReLU(X), where X is also a matrix of decision variables
#     model.update()
#     ts = model.addMVar(X.shape, lb=0, ub=X.getAttr("ub").clip(min=0), name=f"{name}_ts")
#     ss = model.addMVar(
#         X.shape, lb=0, ub=(-X.getAttr("lb")).clip(min=0), name=f"{name}_ss"
#     )
#     model.update()
#     zs = model.addMVar(X.shape, vtype=GRB.BINARY, name=f"{name}_zs")
#     model.addConstr(ts - ss == X, name=f"{name}_constraint_1" if name else None)
#     model.addConstr(
#         ts <= ts.getAttr("ub") * zs, name=f"{name}_constraint_2" if name else None
#     )
#     model.addConstr(
#         ss <= ss.getAttr("ub") * (1 - zs), name=f"{name}_constraint_3" if name else None
#     )
#     return ts


# def add_relu_constraint(model, X, name=None, p=1, **kwargs):
#     # Returns a matrix of decision variables constrained to ReLU(X), where X is also a matrix of decision variables
#     model.update()
#     ts = model.addMVar(X.shape, lb=0, ub=X.getAttr("ub").clip(min=0), name=f"{name}_ts")
#     model.update()
#     zs = model.addMVar(X.shape, vtype=GRB.BINARY, name=f"{name}_zs")
#     model.addConstr(ts >= X, name=f"{name}_constraint_4a" if name else None)
#     model.addConstr(
#         ts <= X - X.getAttr("lb") * (1 - zs),
#         name=f"{name}_constraint_4b" if name else None,
#     )
#     model.addConstr(
#         ts <= X.getAttr("ub") * zs, name=f"{name}_constraint_4c" if name else None
#     )
#     return ts


def add_relu_constraint(model, X, name=None, **kwargs):
    # Returns a matrix of decision variables constrained to ReLU(X), where X is also a matrix of decision variables
    model.update()
    print("    X UB < 0 COUNT:", np.less(X.getAttr("ub"), 0).sum())
    print("    X LB > 0 COUNT:", np.greater(X.getAttr("lb"), 0).sum())

    # if isinstance(X, gp.MVar):
    #     ts = model.addMVar(
    #         X.shape,
    #         lb=X.getAttr("lb").clip(min=0),
    #         ub=X.getAttr("ub").clip(min=0),
    #         name=f"{name}_ts",
    #     )

    #     X_list = np.array(X.tolist()).flatten()
    #     t_list = np.array(ts.tolist()).flatten()
    # else:
    #     ts = model.addVar(lb=min(0, X.lb), ub=max(0, X.ub), name=f"{name}_t")
    #     X_list = [X]
    #     t_list = [ts]

    ts = model.addMVar(
        X.shape,
        lb=X.getAttr("lb").clip(min=0),
        ub=X.getAttr("ub").clip(min=0),
        name=f"{name}_ts",
    )

    X_list = np.array(X.tolist()).flatten()
    t_list = np.array(ts.tolist()).flatten()

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
        X = add_fc_constraint(model, X, lin_weight, lin_bias, name=name + "projection_fc")
        X = add_relu_constraint(model, X, name=name + "projection_relu")

    # Create decision variables to store the aggregated features of each node's neighborhood
    aggregated_features = model.addMVar(X.shape, lb=-float("inf"), ub=float("inf"), name=f"{name}_aggregated_features")

    model.update()

    if aggr == "mean":
        # aggregated_features[i][j] is the sum of all node i's neighbors' feature j divided by the number of node i's neighbors
        # Ensure gp.quicksum(A) does not have any zeros
        aggregated_features.setAttr("lb", (A.getAttr("ub") @ X.getAttr("lb").clip(max=0)) / X.shape[0])
        aggregated_features.setAttr("ub", (A.getAttr("ub") @ X.getAttr("ub").clip(min=0)) / X.shape[0])
        model.addConstr(
            aggregated_features * gp.quicksum(A)[:, np.newaxis] == A @ X,
            name=f"{name}_averages_constraint" if name else None,
        )  # may need to transpose
    elif aggr == "sum":
        aggregated_features.setAttr("lb", A.getAttr("ub") @ X.getAttr("lb").clip(max=0))
        aggregated_features.setAttr("ub", A.getAttr("ub") @ X.getAttr("ub").clip(min=0))

        model.addConstr(
            aggregated_features == A @ X,
            name=f"{name}_sum_constraint" if name else None,
        )

    model.update()

    # The bounds for the output of the layer will be the sums of the bounds of its addends
    first_lower_bounds, first_upper_bounds = get_matmul_bounds(X, lin_r_weight.T)
    second_lower_bounds, second_upper_bounds = get_matmul_bounds(aggregated_features, lin_l_weight.T)

    # If node features are categorical, we can tighten the bounds on the output.
    if name == "Conv_0" and aggr == "sum" and model.getConstrByName("categorical_features[0]") is not None:
        print("Tightening bounds for first term in Conv_0 (mean) for categorical features")
        first_lower_bounds = np.min(lin_r_weight.T, 0, keepdims=True)
        first_upper_bounds = np.max(lin_r_weight.T, 0, keepdims=True)

    ts_lower_bounds = first_lower_bounds + second_lower_bounds + np.expand_dims(lin_l_bias, 0)
    ts_upper_bounds = first_upper_bounds + second_upper_bounds + np.expand_dims(lin_l_bias, 0)
    ts = model.addMVar(
        (X.shape[0], lin_r_weight.shape[0]),
        lb=ts_lower_bounds,
        ub=ts_upper_bounds,
        name=f"{name}_t" if name else None,
    )

    # assert ts_lower_bounds.shape == first_lower_bounds.shape, (
    #     ts_lower_bounds.shape,
    #     first_lower_bounds.shape,
    # )
    # assert first_lower_bounds.shape == (X @ lin_r_weight.T).shape, (
    #     first_lower_bounds.shape,
    #     (X @ lin_r_weight.T).shape,
    # )
    # assert second_lower_bounds.shape == (aggregated_features @ lin_l_weight.T).shape, (
    #     second_lower_bounds.shape,
    #     (aggregated_features @ lin_l_weight.T).shape,
    # )
    # assert first_lower_bounds.shape == second_lower_bounds.shape, (
    #     first_lower_bounds.shape,
    #     second_lower_bounds.shape,
    # )

    # Constrain outputs to correct values
    model.addConstr(
        ts == (X @ lin_r_weight.T) + (aggregated_features @ lin_l_weight.T) + np.expand_dims(lin_l_bias, 0),
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


def global_mean_pool(model, X, name=None, batch=None, **kwargs):
    # Outputs variables constrained to the mean of node features element-wise
    model.update()
    if batch is None:
        batch = [0] * X.shape[0]
    num_graphs = batch[-1] + 1
    averages = model.addMVar(
        (num_graphs, X.shape[1]),
        name=name,
    )
    # model.addConstr(
    #     averages == gp.quicksum(X) / X.shape[0],
    #     name=f"{name}_constraint" if name else None,
    # )
    # return averages[np.newaxis, :]

    for i in range(num_graphs):
        graph_vars = [X[j] for j in range(X.shape[0]) if batch[j] == i]
        model.addConstr(
            averages[i] == gp.quicksum(graph_vars) / len(graph_vars),
            name=f"{name}_constraint_{i}" if name else None,
        )
        averages[i].setAttr("lb", sum(v.getAttr("lb") for v in graph_vars) / len(graph_vars))
        averages[i].setAttr("ub", sum(v.getAttr("ub") for v in graph_vars) / len(graph_vars))
    return averages


# def global_mean_pool(model, X, name=None, **kwargs):
#     # Outputs variables constrained to the mean of node features element-wise
#     model.update()
#     averages = model.addMVar(
#         (X.shape[1],),
#         lb=X.getAttr("lb").mean(axis=0),
#         ub=X.getAttr("ub").mean(axis=0),
#         name=name,
#     )
#     model.addConstr(
#         averages == gp.quicksum(X) / X.shape[0],
#         name=f"{name}_constraint" if name else None,
#     )
#     return averages[np.newaxis, :]


def torch_fc_constraint(model, X, layer, name=None, max_output=None, **kwargs):
    # Encodes a PyTorch Linear layer based on the input X
    weight = layer.get_parameter("weight").cpu().detach().numpy()
    bias = layer.get_parameter("bias").cpu().detach().numpy()
    if max_output is not None:
        weight = weight[max_output][np.newaxis, :]
        bias = np.atleast_2d(bias[max_output])

    return add_fc_constraint(model, X, W=weight, b=bias, name=name)


def torch_sage_constraint(model, A, X, layer, name=None, **kwargs):
    # Encodes a PyTorch-Geometric GraphSAGE convolutional layer object based on the input X and A
    # TODO: Cleanup
    lin_r_weight = layer.get_parameter("lin_r.weight").cpu().detach().numpy()
    lin_l_weight = layer.get_parameter("lin_l.weight").cpu().detach().numpy()
    lin_l_bias = layer.get_parameter("lin_l.bias").cpu().detach().numpy()
    lin_weight, lin_bias = None, None
    if layer.project and hasattr(layer, "lin"):
        lin_weight = layer.get_parameter("lin.weight").cpu().detach().numpy()
        lin_bias = layer.get_parameter("lin.bias").cpu().detach().numpy()
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
    return cosine_similarity, lambda newvec: np.dot(newvec, vec) / (norm(newvec) * norm(vec))


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
    model.update()
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


def get_max(model, var_list, name=None):
    model.update()
    if name is None:
        name = "max_of_" + sum((v.getAttr("varName") for v in var_list), start="")
    max_var = model.addVar(
        lb=max(v.getAttr("lb") for v in var_list),
        ub=max(v.getAttr("ub") for v in var_list),
        vtype=var_list[0].vtype if all(v.vtype == var_list[0].vtype for v in var_list) else GRB.CONTINUOUS,
        name=name,
    )
    model.addGenConstrMax(max_var, var_list, name=f"{name}_constr")
    return max_var


def get_min(model, var_list, name=None):
    model.update()
    if name is None:
        name = "min_of_" + sum((v.getAttr("varName") for v in var_list), start="")
    min_var = model.addVar(
        lb=min(v.getAttr("lb") for v in var_list),
        ub=min(v.getAttr("ub") for v in var_list),
        vtype=var_list[0].vtype if all(v.vtype == var_list[0].vtype for v in var_list) else GRB.CONTINUOUS,
        name=name,
    )
    model.addGenConstrMin(min_var, var_list, name=f"{name}_constr")
    return min_var


def aleqb(a, b):
    sorted_a = sorted(list(a))
    sorted_b = sorted(list(b))
    for i in range(min(len(sorted_a), len(sorted_b))):
        if sorted_a[i] < sorted_b[i]:
            return -1
        if sorted_a[i] > sorted_b[i]:
            return 1
    if len(sorted_a) == len(sorted_b):
        return 0
    elif len(sorted_a) < len(sorted_b):
        return 1
    else:
        return -1


def get_ranks(d):
    initial_ranks = {i: sum(aleqb(d[i], d[j]) == 1 for j in d.keys()) for i in d.keys()}
    ranks = {k: v for v, k in enumerate(sorted(set(initial_ranks.values())))}
    return {k: ranks[initial_ranks[k]] for k in d.keys()}


def LO(d):
    # Lexicographic ordering of the keys of d based on their values, which are set
    return sorted(d.keys(), key=lambda k: cmp_to_key(aleqb)(d[k]))


def canonicalize_graph(data):
    # This function will reorder the nodes of a given graph (PyTorch Geometric "Data" Object) to a canonical (ish) version
    # From https://arxiv.org/pdf/2305.09420.pdf

    G = to_networkx(data)

    ## Test with example from reference above
    # import networkx as nx
    # G = nx.Graph()
    # for i in range(6):
    #     G.add_node(i)
    # G.add_edges_from(
    #     [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (2, 5), (3, 4)]
    # )

    # # Get node ordering
    # sorted_nodes = list(dfs_preorder_nodes(G, source=next(iter(G.nodes))))
    # sorted_nodes.reverse()
    # # Create a mapping of old labels to new labels
    # label_mapping = {node: i for i, node in enumerate(sorted_nodes)}

    unindexed_nodes = set(G.nodes)
    # first_node = unindexed_nodes.pop()
    # Set the first node to be the one with the highest lexicographic ordering
    F = data.x.shape[1]
    first_node = np.argmax(
        [sum(2 ** (F - f - 1) * data.x[i, f] for f in range(F)).item() for i in range(G.number_of_nodes())]
    )
    unindexed_nodes.remove(first_node)
    s = 1
    label_mapping = {first_node: 0}
    while s < len(G.nodes):
        indexed_neighbors = {
            node: {label_mapping[neighbor] for neighbor in G.neighbors(node) if neighbor in label_mapping}
            for node in unindexed_nodes
        }
        # print("Indexed Neighbors:", indexed_neighbors)
        ranks = get_ranks(indexed_neighbors)
        # print("Ranks:", ranks)
        temporary_indices = {v: (label_mapping[v] if v in label_mapping else ranks[v] + s) for v in G.nodes}
        # print("Temporary Indices:", temporary_indices)
        temp_neighbor_set = {v: [temporary_indices[u] for u in G.neighbors(v)] for v in unindexed_nodes}
        # print("Temporary Neighbor Set:", temp_neighbor_set)
        vs = LO(temp_neighbor_set)[0]
        # print("LO:", LO(temp_neighbor_set))
        # print("Selected Node:", vs)

        label_mapping[vs] = s
        unindexed_nodes.remove(vs)
        s += 1

    sorted_nodes = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])

    # Relabel the graph
    G = relabel_nodes(G, label_mapping)

    data.x = data.x[sorted_nodes, :]
    data.edge_index = Tensor(list(G.edges)).to(data.edge_index.dtype).T
