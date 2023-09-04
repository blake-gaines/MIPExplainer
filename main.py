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
import time
from torch_geometric.datasets import TUDataset
import pickle
from collections import OrderedDict
from utils import save_graph, get_average_phi

print(time.strftime("\nStart Time: %H:%M:%S", time.localtime()))

if not os.path.isdir("solutions"): os.mkdir("solutions")

model_path = "models/MUTAG_model.pth"
# model_path = "models/OurMotifs_model_mean.pth"

dataset = TUDataset(root='data/TUDatascet', name='MUTAG')
# num_node_features = dataset.num_node_features

# with open("data/OurMotifs/dataset.pkl", "rb") as f:
#     dataset = pickle.load(f)
num_node_features = dataset[0].x.shape[1]
init_with_data = True
init_index = 0
num_nodes = 5
if init_with_data:
    print(f"Initializing with solution from graph {init_index}")
    init_graph = dataset[init_index]
    num_nodes = init_graph.num_nodes
else:
    print(f"Initializing with dummy graph")
    init_graph_x = torch.eye(num_node_features)[torch.randint(num_node_features, (num_nodes,)),:]
    init_graph_adj = torch.randint(0, 2, (num_nodes, num_nodes))
    init_graph = Data(x=init_graph_x,edge_index=dense_to_sparse(init_graph_adj)[0])

if __name__ == "__main__":
    nn = torch.load(model_path)
    nn.eval()
    nn.double()

    phi = get_average_phi(dataset, nn, "Aggregation")

    print(nn)

    m = gp.Model("GNN Inverse")

    A = m.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY, name="A")
    X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X") # vtype for BINARY node features
    m.update()

    m.addConstr(gp.quicksum(A) >= 1, name="non_isolatied") # Nodes need an edge. Need this for SAGEConv inverse to work UNCOMMENT IF NO OTHER CONSTRAINTS DO THIS
    force_undirected(m, A) # Generate an undirected graph
    m.addConstr(gp.quicksum(X.T) == 1, name="categorical") # REMOVE for non-categorical features

    # # Impose some ordering to keep graph connected
    # for i in range(num_nodes):
    #     m.addConstr(gp.quicksum(A[i][j+1] for j in range(i,num_nodes-1)) >= 1,name="node_i_connected")

    # X = m.addMVar((10, 7), lb=-5, ub=5, name="x")

    max_class = 1

    ## Build a MIQCP for the trained neural network
    output_vars = OrderedDict()
    output_vars["Input"] = X
    for name, layer in nn.layers.items():
        previous_layer_output = output_vars[next(reversed(output_vars))]
        if isinstance(layer, Linear):
            output_vars[name] = torch_fc_constraint(m, previous_layer_output, layer, name=name)
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
        
    # These constraints are lower bounds, needed because quadratic equality constraints are non-convex
    embedding = output_vars["Aggregation"]
    embedding_similarity = m.addVar(lb=0, ub=1, name="embedding_similarity")
    # embedding_similarity = 0

    # L2 Similarity
    # embedding_phi_diffs = m.addMVar(embedding.shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="embedding_phi_diffs")
    # m.addConstr(embedding_phi_diffs == embedding - phi[max_class])
    # m.addGenConstrNorm(embedding_similarity, embedding_phi_diffs, which=2, name="embedding_similarity_pnorm")

    # # L2 Similarity
    # m.addConstr(embedding_similarity*embedding_similarity >= gp.quicksum((embedding - phi[max_class])*(embedding - phi[max_class])), name="embedding_similarity_l2")

    # Cosine Similarity
    # embedding_magnitude = m.addVar(lb=0, ub=GRB.INFINITY, name="embedding_magnitude")
    # phi_magnitude = np.linalg.norm(phi[max_class])
    # m.addGenConstrNorm(embedding_magnitude, output_vars["Aggregation"], which=2, name="embedding_magnitude_constraint")
    # m.addConstr(gp.quicksum(embedding*phi) >= embedding_magnitude*phi_magnitude*embedding_similarity, name="embedding_similarity_cosine")

    # Cosine Similarity
    embedding_magnitude = m.addVar(lb=0, name="embedding_magnitude")
    m.addConstr(embedding_magnitude*embedding_magnitude >= gp.quicksum(embedding*embedding), name="embedding_magnitude_constraint")
    m.addConstr(np.linalg.norm(phi[max_class])*embedding_magnitude*embedding_similarity <= gp.quicksum(embedding*phi), name="embedding_similarity_cosine")
            

    ## MIQCP objective function
    m.setObjective(output_vars["Output"][max_class]+10*embedding_similarity, GRB.MAXIMIZE)

    m.update()
    m.write("model.mps") # Save model file

    # Define the callback function
    solution_count = 0
    def callback(model, where):
        global A, X, solution_count
        if where == GRB.Callback.MIPSOL:
            # A new incumbent solution has been found
            print(f"New incumbent solution found (ID {solution_count}) Objective value: {model.cbGet(GRB.Callback.MIPSOL_OBJ)}")

            solution_count += 1
            X_sol = model.cbGetSolution(X)
            A_sol = model.cbGetSolution(A)
            output_var_value = model.cbGetSolution(output_vars["Output"])

            ## Sanity Check
            print("NN output given X:", nn.forwardXA(X_sol, A_sol))
            print("predicted output:", output_var_value)

            embedding_var_value = model.cbGetSolution(embedding)
            embedding_sim_var_value = model.cbGetSolution(embedding_similarity)
            embedding_magnitude_var_value = model.cbGetSolution(embedding_similarity)
            print("Predicted Embedding Magnitude vs Actual:", embedding_magnitude_var_value, np.linalg.norm(embedding_var_value))
            print("Predicted Embedding Similarity vs Actual:", embedding_sim_var_value, np.dot(embedding_var_value, phi)/(np.linalg.norm(embedding_var_value)*np.linalg.norm(phi)))

            save_graph(A=A_sol, X=X_sol, index=solution_count)

    ## Start with a graph from the dataset
    m.NumStart = 1
    # m.update()
    # for s in range(m.NumStart):
    #     m.params.StartNumber = s
    init_graph.x = init_graph.x.double()
    init_graph.edge_index = to_undirected(init_graph.edge_index)
    # X.Start = G.x.detach().numpy()
    A.Start = to_dense_adj(init_graph.edge_index).detach().numpy().squeeze()
    all_outputs = nn.get_all_layer_outputs(init_graph)
    assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
    for var, name_output in zip(output_vars.values(), all_outputs):
        layer_name = name_output[0]
        output = name_output[1].detach().numpy().squeeze()
        assert var.shape == output.shape
        var.Start = output
        # if layer_name == "Aggregation":
        #     m.addConstr(gp.quicksum((output - var)*(output - var)) <= 0.1) #######################################
    m.update()
    save_graph(A.Start, X.Start, 0)

    # Use tuned parameters
    m.read("./tune0.prm")

    # # Tune
    # print("Tuning")
    # m.setParam("TuneTimeLimit", 3600*20)
    # m.tune()
    # for i in range(m.tuneResultCount):
    #     m.getTuneResult(i)
    #     m.write('tune'+str(i)+'.prm')
    # print("Done Tuning")

    m.setParam("TimeLimit", 3600*8)
    m.setParam("NonConvex", 2)
    m.optimize(callback)

    print("Status:", m.Status)

    if m.Status in [3, 4]: # If the model is infeasible, see why
        m.computeIIS()
        m.write("model.ilp")
    else:
        save_graph(A.X, X.X, "Final")
        print("NN output given X", nn.forwardXA(X.X, A.X))
        print("predicted output", output_vars["Output"].X)

print(time.strftime("\nEnd Time: %H:%M:%S", time.localtime()))