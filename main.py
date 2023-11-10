import torch
from gnn import GNN
import gurobipy as gp
from gurobipy import GRB
from invert import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_undirected
from torch_geometric.data import Data
import networkx as nx
import os
from torch.nn import Linear, ReLU
from torch_geometric.nn.aggr import Aggregation, SumAggregation, MeanAggregation, MaxAggregation
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import TUDataset
import pickle
from collections import OrderedDict
from utils import *
import matplotlib.pyplot as plt
import sys
from torch_geometric.utils import from_networkx, to_networkx
from arg_parser import parse_args

args = parse_args()

dataset_name = args.dataset_name
model_path = args.model_path
max_class = args.max_class
output_file = args.output_file
sim_weights = dict(zip(args.regularizers, args.regularizer_weights))
sim_methods = args.regularizers
num_nodes = args.num_nodes

if not model_path:
    model_path = f"models/{dataset_name}_model.pth"

torch.manual_seed(12345)
# TODO: Seed for Gurobi

if not os.path.isdir("solutions"): os.mkdir("solutions")

if dataset_name == "MUTAG":
    dataset = TUDataset(root='data/TUDatascet', name='MUTAG')
else:
    with open(f"data/{dataset_name}/dataset.pkl", "rb") as f: dataset = pickle.load(f)

# Count classes in dataset, node features
ys = [int(d.y) for d in dataset]
num_classes = len(set(ys))
num_node_features = dataset[0].x.shape[1]

def canonicalize_graph(graph):
    # This function will reorder the nodes of a given graph (PyTorch Geometric "Data" Object) to a canonical (maybe) version
    # TODO: Generalize to non one-hot vector node features

    G = to_networkx(init_graph)

    # Get source node for DFS, should be largest degree/first lexicographically (TODO)
    A = to_dense_adj(graph.edge_index).detach().numpy().squeeze()
    weighted_feature_sums = A @ (graph.x.detach().numpy() @ np.linspace(1,graph.num_node_features, num=graph.num_node_features))
    min_node=np.argmin(A.sum(axis=0)*num_node_features*num_nodes+weighted_feature_sums)

    # Get node ordering
    sorted_nodes = list(nx.dfs_preorder_nodes(G, source=min_node))
    sorted_nodes.reverse()

    # Create a mapping of old labels to new labels
    label_mapping = {i: node for i, node in enumerate(sorted_nodes)}

    # Relabel the graph
    G = nx.relabel_nodes(G, label_mapping)

    graph.x = init_graph.x[sorted_nodes, :]
    graph.edge_index = torch.Tensor(list(G.edges)).to(torch.int64).T

if __name__ == "__main__":
    # Load the model
    nn = torch.load(model_path, fix_imports=True)
    nn.eval()
    nn.to(torch.float64)
    
    # Track hyperparameters
    if args.log:
        import wandb
        wandb.login()
        config = {
            "architecture": str(nn),
            "big_number": big_number,
            "model_path": model_path, 
            }
        config.update(vars(args))
        wandb.init(
            project="GNN-Inverter", 
            config=config,
        )
        wandb.save(args.param_file, policy="now")
        wandb.save(output_file, policy="end")
        wandb.run.log_code(".")

    print("Args:", args)
    print("Number of Classes", num_classes)
    print("Number of Node Features", num_node_features)

    if args.init_with_data:
        if args.init_index is not None:
            print(f"Initializing from dataset with graph at index {args.init_index}")
            init_graph = dataset[args.init_index]
        elif num_nodes is not None:
            print(f"Initializing from dataset graph with {num_nodes} nodes")
            init_graph = random.choice([d for d in dataset if int(d.y) == max_class and d.num_nodes == num_nodes])
        A = to_dense_adj(init_graph.edge_index).detach().numpy().squeeze()
        print("Connected before reordering:", all([sum(A[i][j] + A[j][i] for j in range(i+1,A.shape[0])) >= 1 for i in range(A.shape[0]-1)]))

        canonicalize_graph(init_graph)

        A = to_dense_adj(init_graph.edge_index).detach().numpy().squeeze()
        assert all([sum(A[i][j] + A[j][i] for j in range(i+1,A.shape[0])) >= 1 for i in range(A.shape[0]-1)]), "Initialization graph was not connected"

        num_nodes = init_graph.num_nodes

    else:
        # By default, will generate a line graph with uniform random node features
        print(f"Initializing with dummy graph")
        # init_graph_adj = np.clip(init_graph_adj + np.eye(num_nodes, k=1), a_min=0, a_max=1)
        init_graph_adj = torch.diag_embed(torch.diag(torch.ones((num_nodes, num_nodes)), diagonal=-1), offset=-1)+torch.diag_embed(torch.diag(torch.ones((num_nodes, num_nodes)), diagonal=1), offset=1)
        if dataset_name in ["Is_Acyclic", "Shapes", "Shapes_Clean"]:
            init_graph_x = torch.unsqueeze(torch.sum(init_graph_adj, dim=-1), dim=-1)
        elif dataset_name in ["MUTAG", "OurMotifs"]:
            # init_graph_x = torch.eye(num_node_features)[torch.randint(num_node_features, (num_nodes,)),:]
            init_graph_x = torch.eye(num_node_features)[torch.randint(1, (num_nodes,)),:]
        elif dataset_name in ["Shapes_Ones", "Is_Acyclic_Ones"]:
            init_graph_x = torch.ones((num_nodes, num_node_features))

        # init_graph_adj = torch.randint(0, 2, (num_nodes, num_nodes))
        # init_graph_adj = torch.ones((num_nodes, num_nodes))
        init_graph = Data(x=init_graph_x,edge_index=dense_to_sparse(init_graph_adj)[0])

        canonicalize_graph(init_graph)

    # Each row of phi is the average embedding of the graphs in the corresponding class of the dataset
    phi = get_average_phi(dataset, nn, "Aggregation")

    print(nn)
    print("Model Parameters:", sum(param.numel() for param in nn.parameters()))

    m = gp.Model("GNN Inverse")

    # Add and constrain decision variables for adjacency matrix
    A = m.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY, name="A")
    force_connected(m, A)
    force_undirected(m, A)
    remove_self_loops(m, A)
    # m.addConstr(gp.quicksum(A) >= 1, name="non_isolatied") # Nodes need an edge. Need this for SAGEConv inverse to work. UNCOMMENT IF NO OTHER CONSTRAINTS DO THIS


    # Add and constrain decision variables for node feature matrix
    if dataset_name in ["MUTAG", "OurMotifs"]:
        X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X")
        m.addConstr(gp.quicksum(X.T) == 1, name="categorical_features")
    elif dataset_name in ["Is_Acyclic", "Shapes", "Shapes_Clean"]:
        X = m.addMVar((num_nodes, num_node_features), lb=0, ub=init_graph.num_nodes, name="X", vtype=GRB.INTEGER)
        m.addConstr(X == gp.quicksum(A)[:, np.newaxis], name="features_are_node_degrees")
    elif dataset_name in ["Shapes_Ones", "Is_Acyclic_Ones"]:
        X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X")
        m.addConstr(X == 1, name="features_are_ones")
    else:
        raise ValueError(f"Unknown Decision Variables for {dataset_name}")
    
    # # Enforce canonical (maybe) representation
    # # Only works for one-hot node features and assumes an undirected graph
    # m.update()
    # if X[0][0].vtype == GRB.BINARY:
    #     # messages [i][j][k] is 1 if node i is a neighbor of node j and node i's feature k is 1
    #     messages = m.addMVar((num_nodes, num_nodes, num_node_features), vtype=GRB.BINARY)
    #     messages = m.addVars([(i,j,k) for i in range(num_nodes) for j in range(num_nodes) for k in range(num_node_features)], vtype=GRB.BINARY, name="initial_messages_constraint")
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             for k in range(num_node_features):
    #                 m.addGenConstrIndicator(messages[i,j,k], 1, A[i,j]+X[i,k], GRB.EQUAL, 2, name=f"initial_message_{i}_{j}_{k}")

    #     # neighborhoods_match[i,j,k] is a binary decision variable, constrained to 1 if node i and node j have the same number of neighbors with feature k equal to 1, 0 otherwise
    #     neighborhoods_match = m.addVars([(i,j,k) for i in range(num_nodes-1) for j in range(i+1, num_nodes) for k in range(num_node_features)], vtype=GRB.BINARY, name="neighborhoods_match")
    #     # obeys_orderings is only 1 if two nodes have the correct partial ordering based on their node features
    #     # If they have the same neighborhood, then the node with the smaller features lexicographically should come first in the ordering
    #     obeys_orderings = m.addVars([(i,j) for i in range(num_nodes-1) for j in range(i+1, num_nodes)], vtype=GRB.BINARY, name="obeys_orderings")
    #     for i in range(num_nodes-1):
    #         for j in range(i+1, num_nodes):
    #             for k in range(num_node_features):
    #                 # Constrain the neighborhoods_match variables to the correct values
    #                 i_neighborhood_ks = messages.select(i,'*',k)
    #                 i_neighborhood_ks.pop(i)
    #                 j_neighborhood_ks = messages.select(j,'*',k)
    #                 j_neighborhood_ks.pop(j)
    #                 m.addGenConstrIndicator(neighborhoods_match[i,j,k], 1, sum(i_neighborhood_ks)-sum(j_neighborhood_ks), GRB.EQUAL, 0, name=f"neighborhood_match_constraint_{i}_{j}_{k}")

    #             # Constrain the obeys_orderings variables to the correct values
    #             # For each k in the number of features, the sum of the features of node i before or at k equal the sum of the features of node j at or after k
    #             m.addGenConstrIndicator(obeys_orderings[i,j], 1, sum(sum(X[i][:k+1])-sum(X[j][k:]) for k in range(num_node_features)), GRB.GREATER_EQUAL, 0, name=f"obeys_ordering_constraint_{i}_{j}")
                
    #             # If the neighborhoods match, the ordering must be obeyed
    #             m.addConstr(sum(neighborhoods_match.select(i, j, '*'))-obeys_orderings[i,j] <= num_node_features-1, name=f"nb_ordering_{i}_{j}")

    #     ## The first node will have the highest degree and come first lexicographically among nodes with the highest degree
    #     weighted_feature_sums = A @ (X @ np.linspace(1,num_node_features, num=num_node_features))
    #     for j in range(1, num_nodes):
    #         # TODO: Fix for directed graphs, integer features
    #         m.addConstr(sum(A[0])*num_node_features*num_nodes+sum(weighted_feature_sums[0]) <= sum(A[j])*num_node_features*num_nodes+sum(weighted_feature_sums[j]), name=f"node_0_smallest_{j}")
    #     m.update()

    ## Build a MIQCP for the trained neural network
    ## For each layer, create and constrain decision variables to represent the output
    output_vars = OrderedDict()
    output_vars["Input"] = X
    for name, layer in nn.layers.items():
        previous_layer_output = output_vars[next(reversed(output_vars))]
        if isinstance(layer, Linear):
            output_vars[name] = torch_fc_constraint(m, previous_layer_output, layer, name=name, max_output=max_class if args.trim_unneeded_outputs and name == "Output" else None)
        elif isinstance(layer, SAGEConv):
            output_vars[name] = torch_sage_constraint(m, A, previous_layer_output, layer, name=name)
        elif isinstance(layer, MeanAggregation):
            output_vars[name] = global_mean_pool(m, previous_layer_output, name=name)
        elif isinstance(layer, SumAggregation):
            output_vars[name] = global_add_pool(m, previous_layer_output, name=name)
        elif isinstance(layer, ReLU):
            output_vars[name] = add_relu_constraint(m, previous_layer_output, name=name)
        else:
            raise NotImplementedError(f"layer type {layer} has no MIQCP analog")

    # # Lexicographic ordering of node embeddings
    # embedding = output_vars["Aggregation"][0]
    # for i in range(num_nodes-1):
    #     emebdding_indices = (embedding/np.linalhg) @ np.linspace(1,graph.num_node_features, num=graph.num_node_features)
    #     for j in range(i+1, num_nodes):
    #         m.addConstr(emebdding_indices[i] <= emebdding_indices[j])
        
    ## Create decision variables to represent (unweighted) regularizer terms based on embedding similarity/distance
    ## These can also be used in constraints!!!
    embedding = output_vars["Aggregation"][0]
    regularizers = {}
    if "Cosine" in sim_methods:
        # Cosine Similarity
        cosine_similarity = m.addVar(lb=0, ub=1, name="embedding_similarity") # Add variables for cosine similarity
        embedding_magnitude = m.addVar(lb=0, ub=sum(embedding.getAttr("ub")**2)**0.5, name="embedding_magnitude") # Variables for embedding magnitude, intermediate value in calculation
        phi_magnitude = np.linalg.norm(phi[max_class])
        m.addGenConstrNorm(embedding_magnitude, embedding, which=2, name="embedding_magnitude_constraint")
        m.addConstr(gp.quicksum(embedding*phi[max_class]) == embedding_magnitude*phi_magnitude*cosine_similarity, name="cosine_similarity") # u^Tv=|u||v|cos_sim(u,v)
        regularizers["Cosine"] = cosine_similarity

        # Cosine Similarity
        # phi_magnitude = np.linalg.norm(phi[max_class])
        # embedding_magnitude = m.addVar(lb=0, name="embedding_magnitude")
        # m.addConstr(embedding_magnitude*embedding_magnitude == gp.quicksum(embedding*embedding), name="embedding_magnitude_constraint")
        # m.addConstr(gp.quicksum(embedding*phi[max_class]) == embedding_magnitude*phi_magnitude*embedding_similarity, name="embedding_similarity")
    if "L2" in sim_methods:
        # L2 Distance
        l2_similarity = m.addVar(lb=0, ub=sum(embedding.getAttr("ub")**2)**0.5, name="l2_distance") # Add variables for L2 Distance
        m.addConstr(l2_similarity*l2_similarity == gp.quicksum((phi[max_class]-embedding)*(phi[max_class]-embedding)), name="l2_similarity") # l2_dist(u,v)^2 = (u-v)^T(u-v)
        regularizers["L2"] = l2_similarity
    if "Squared L2" in sim_methods:
        # Squared L2 Distance
        squared_l2_similarity = m.addVar(lb=0, ub=sum(embedding.getAttr("ub")**2), name="l2_distance") # Add variables for Squared L2 Distance
        m.addConstr(squared_l2_similarity >= gp.quicksum((phi[max_class]-embedding)*(phi[max_class]-embedding)), name="l2_similarity") # l2_dist(u,v)^2 = (u-v)^T(u-v)
        regularizers["Squared L2"] = squared_l2_similarity
    m.update()

    # List of decision variables representing the logits that are not the max_class logit
    other_outputs_vars = [output_vars["Output"][0, j] for j in range(num_classes) if j!=max_class]

    # # Constrain other_outputs_vars to be at most the one we want to maximize
    # for var in other_outputs_vars:
    #     m.addConstr(var <= max_output_var)

    # # Create a decision variable and constrain it to the maximum of the non max_class logits
    other_outputs_max = m.addVar(name="other_outputs_max", lb=max(v.getAttr("lb") for v in other_outputs_vars), ub=max(v.getAttr("ub") for v in other_outputs_vars))
    m.addGenConstrMax(other_outputs_max, other_outputs_vars, name="max_of_other_outputs")

    max_output_var = output_vars["Output"][0] if args.trim_unneeded_outputs else output_vars["Output"][0, max_class]  
    ## MIQCP objective function
    m.setObjective(max_output_var-other_outputs_max+sum(sim_weights[sim_method]*regularizers[sim_method] for sim_method in sim_methods), GRB.MAXIMIZE)
    m.update()

    # Save a copy of the model
    m.write("model.lp")
    m.write("model.mps")
    if args.log:
        wandb.save("model.lp", policy="now")
        wandb.save("model.mps", policy="now")

    # Define the callback function for the solver to save intermediate solutions, other emtrics
    solutions = []
    previous_var_values = [None]*len(m.getVars())
    def callback(model, where):
        global A, X, solutions
        if where == GRB.Callback.MIPSOL:
            # A new incumbent solution has been found
            print(f"New incumbent solution found (ID {len(solutions)}) Objective value: {model.cbGet(GRB.Callback.MIPSOL_OBJ)}")

            # See which variables were changed from the previous solution
            n_changed = 0
            changed_vars = []
            vars = m.getVars()
            for i in range(len(vars)):
                prev_value = previous_var_values[i]
                var = vars[i]
                new_value = model.cbGetSolution(var)
                if prev_value != new_value:
                    n_changed += 1
                    changed_vars.append((var.VarName, prev_value, new_value))
                previous_var_values[i] = new_value
            print(f"{n_changed} variables different from previous solution.")

            X_sol = model.cbGetSolution(X)
            A_sol = model.cbGetSolution(A)
            output_var_value = model.cbGetSolution(output_vars["Output"])

            ## Sanity Check, ensure that the decision variables for the network's output actually match the network's output
            sol_output = nn.forwardXA(X_sol, A_sol).detach().numpy()
            assert np.allclose(sol_output, output_var_value), "uh oh :("

            # Manually calculate and save regularizer values
            # This way, decision variables for regularizers can just be bounded above/below by the correct value if that helps
            embedding_var_value = model.cbGetSolution(embedding)
            similarities = {}
            for sim_method in sim_methods:
                if sim_method == "Cosine":
                    sol_similarity = np.dot(embedding_var_value, phi[max_class])/(np.linalg.norm(embedding_var_value)*np.linalg.norm(phi[max_class]))
                elif sim_method == "L2":
                    sol_similarity = np.sqrt(sum((embedding_var_value - phi[max_class])*(embedding_var_value - phi[max_class])))
                elif sim_method == "Squared L2":
                    sol_similarity = sum((embedding_var_value - phi[max_class])*(embedding_var_value - phi[max_class]))
                similarities[sim_method] = sol_similarity
                embedding_sim_var_value = model.cbGetSolution(regularizers[sim_method])
                print(f"Predicted {sim_method} vs Actual:", embedding_sim_var_value, sol_similarity)

            solution = {
                "X": X_sol,
                "A": A_sol,
                "Output": sol_output,
                "Objective Value": model.cbGet(GRB.Callback.MIPSOL_OBJ),
                "Upper Bound": model.cbGet(GRB.Callback.MIPSOL_OBJBND),
                "Variables Changed": n_changed,
                # "Changed Variables": changed_vars,
            }
            solution.update(similarities)
            solutions.append(solution)

            if args.log:
                fig, _ = draw_graph(A_sol, X_sol)
                wandb.log(solutions[-1], commit=False)
                wandb.log({f"Output Logit {i}": sol_output[0, i] for i in range(sol_output.shape[1])}, commit=False)
                wandb.log({"fig": wandb.Image(fig)})
                plt.close()

            with open(output_file, "wb") as f:
                pickle.dump(solutions, f)

    ## Warm start - create an initial solution for the model
    # TODO: Initial values for canonicalization variables (not currently a problem, the solver completes the partial solution)
    m.NumStart = 1
    init_graph.x = init_graph.x.to(torch.float64)
    init_graph.edge_index = to_undirected(init_graph.edge_index) ## TODO: Remove

    A.Start = to_dense_adj(init_graph.edge_index).detach().numpy().squeeze()
    all_outputs = nn.get_all_layer_outputs(init_graph) # Get the outputs of each layer from the GNN given the init_graph
    all_ub, all_lb = [], []
    assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
    # Creates starts for each layer in the model, including the inputs
    for var, name_output in zip(output_vars.values(), all_outputs):
        layer_name = name_output[0]
        # var = output_vars[layer_name]
        output = name_output[1].detach().numpy()
        if args.trim_unneeded_outputs and layer_name == "Output": 
            output = output[:, max_class][np.newaxis, :]
        m.update()

        # Allows us to check ranges for bounds
        all_lb.extend(var.getAttr("lb").flatten().tolist())
        all_ub.extend(var.getAttr("ub").flatten().tolist())

        assert var.shape == output.shape, (layer_name, var.shape, output.shape)
        assert np.less_equal(var.getAttr("lb"), output).all(), (layer_name, var.shape, var.getAttr("lb").min(), output.min(), np.greater(var.getAttr("lb"), output).sum())
        assert np.greater_equal(var.getAttr("ub"), output).all(), (layer_name, var.shape, var.getAttr("ub").max(), output.max(), np.less(var.getAttr("ub"), output).sum())
        
        var.Start = output

        ## Fix outputs for debugging
        # var.setAttr("lb", output)
        # var.setAttr("ub", output)

        if layer_name == "Aggregation":
            for sim_method in sim_methods:
                if sim_method == "Cosine":
                    regularizers[sim_method].Start = np.dot(output[0], phi[max_class])/(np.linalg.norm(output[0])*np.linalg.norm(phi[max_class]))
                elif sim_method == "L2":
                    regularizers[sim_method].Start = np.sqrt(sum((output[0] - phi[max_class])*(output[0]- phi[max_class])))
                elif sim_method == "Squared L2":
                    regularizers[sim_method].Start = sum((output[0] - phi[max_class])*(output[0] - phi[max_class]))
    m.update()

    print(f"Lowest Lower Bound: {min(all_lb)} | Highest Upper Bound: {max(all_ub)}, | Min AV Bound: {min([b for b in np.abs(all_lb+all_ub) if b>0])}")

    # Check variables are bounded or binary
    for var in m.getVars():
        assert var.vtype==GRB.BINARY or (var.LB != float('-inf') and var.UB != float('inf')), f"Variable {var.VarName} is unbounded."

    # Get solver parameters
    m.read(args.param_file)
    
    # # Tune solver parameters
    # print("Tuning")
    # # m.setParam("TuneTimeLimit", 3600*48)
    # m.tune()
    # for i in range(m.tuneResultCount):
    #     m.getTuneResult(i)
    #     m.write('tune'+str(i)+'.prm')
    # print("Done Tuning")

    m.setParam("TimeLimit", 3600*6)

    m.optimize(callback)

    # Save all solutions
    with open(output_file, "wb") as f:
        pickle.dump(solutions, f)

    print("Model Status:", m.Status)

    if m.Status in [3, 4]: # If the model is infeasible, see why
        m.computeIIS()
        m.write("model.ilp")