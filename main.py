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
from utils import *
import wandb
import matplotlib.pyplot as plt
import sys

log_run = False

torch.manual_seed(12345)

print(time.strftime("\nStart Time: %H:%M:%S", time.localtime()))

if not os.path.isdir("solutions"): os.mkdir("solutions")

# dataset_name = "MUTAG"
# model_path = "models/MUTAG_model.pth"
# dataset = TUDataset(root='data/TUDatascet', name='MUTAG')

dataset_name = "Shapes" # {"Shapes", "OurMotifs", "Is_Acyclic"}
model_path = f"models/{dataset_name}_model.pth"
with open(f"data/{dataset_name}/dataset.pkl", "rb") as f: dataset = pickle.load(f)


max_class = 0
output_file = "./solutions.pkl"
sim_methods = ["Cosine"]
sim_weights = {
    "Cosine": 10,
    "Squared L2": -0.01,
    "L2": -1,
}

num_node_features = dataset[0].x.shape[1]
init_with_data = False
init_index = 0
num_nodes = 10
if init_with_data:
    print(f"Initializing with solution from graph {init_index}")
    init_graph = dataset[init_index]
    num_nodes = init_graph.num_nodes
else:
    print(f"Initializing with dummy graph")
    # init_graph_adj = np.clip(init_graph_adj + np.eye(num_nodes, k=1), a_min=0, a_max=1)
    init_graph_adj = torch.diag_embed(torch.diag(torch.ones((num_nodes, num_nodes)), diagonal=-1), offset=-1)+torch.diag_embed(torch.diag(torch.ones((num_nodes, num_nodes)), diagonal=1), offset=1)
    if dataset_name in ["Is_Acyclic", "Shapes"]:
        init_graph_x = torch.unsqueeze(torch.sum(init_graph_adj, dim=-1), dim=-1)
    elif dataset_name in ["MUTAG", "OurMotifs"]:
        init_graph_x = torch.eye(num_node_features)[torch.randint(num_node_features, (num_nodes,)),:]
    # init_graph_adj = torch.randint(0, 2, (num_nodes, num_nodes))
    # init_graph_adj = torch.ones((num_nodes, num_nodes))
    init_graph = Data(x=init_graph_x,edge_index=dense_to_sparse(init_graph_adj)[0])

if __name__ == "__main__":
    nn = torch.load(model_path)
    nn.eval()
    # nn.double()
    nn = nn.to(torch.double)

    phi = get_average_phi(dataset, nn, "Aggregation")

    print(nn)

    if log_run:
        wandb.login()
        wandb.init(
            project="GNN-Inverter", 
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": str(nn),
            "dataset": dataset_name,
            "max_class": max_class,
            "output_file": output_file,
            "sim_weights": sim_weights,
            "sim_methods": sim_methods,
            "M": M,
            "big_number": big_number,
            "init_with_data": init_with_data,
            "model_path": model_path,  
        })
        wandb.save("model.lp", policy="end")
        wandb.save("model.mps", policy="end")
        wandb.save("solutions.pkl", policy="end")
        wandb.run.log_code(".")

    m = gp.Model("GNN Inverse")

    A = m.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY, name="A")
    force_connected(m, A)
    force_undirected(m, A) # Generate an undirected graph
    # remove_self_loops(m, A)
    m.addConstr(gp.quicksum(A) >= 1, name="non_isolatied") # Nodes need an edge. Need this for SAGEConv inverse to work UNCOMMENT IF NO OTHER CONSTRAINTS DO THIS

    if dataset_name in ["MUTAG", "OurMotifs"]:
        X = m.addMVar((num_nodes, num_node_features), vtype=GRB.BINARY, name="X")
        m.addConstr(gp.quicksum(X.T) == 1, name="categorical_features")
    elif dataset_name in ["Is_Acyclic", "Shapes"]:
        X = m.addMVar((num_nodes, num_node_features), lb=0, ub=init_graph.num_nodes, name="X", vtype=GRB.INTEGER)
        m.addConstr(X == gp.quicksum(A)[:, np.newaxis], name="features_are_node_degrees")
    else:
        raise ValueError(f"Unknown Decision Variables for {dataset_name}")

    m.update()

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
        
    embedding = output_vars["Aggregation"][0]
    regularizers = {}
    if "Cosine" in sim_methods:
        cosine_similarity = m.addVar(lb=0, ub=1, name="embedding_similarity")

        # Cosine Similarity
        embedding_magnitude = m.addVar(lb=0, ub=GRB.INFINITY, name="embedding_magnitude")
        phi_magnitude = np.linalg.norm(phi[max_class])
        m.addGenConstrNorm(embedding_magnitude, embedding, which=2, name="embedding_magnitude_constraint")
        m.addConstr(gp.quicksum(embedding*phi[max_class]) == embedding_magnitude*phi_magnitude*cosine_similarity, name="cosine_similarity")
        regularizers["Cosine"] = cosine_similarity

        # Cosine Similarity
        # phi_magnitude = np.linalg.norm(phi[max_class])
        # embedding_magnitude = m.addVar(lb=0, name="embedding_magnitude")
        # m.addConstr(embedding_magnitude*embedding_magnitude == gp.quicksum(embedding*embedding), name="embedding_magnitude_constraint")
        # m.addConstr(gp.quicksum(embedding*phi[max_class]) == embedding_magnitude*phi_magnitude*embedding_similarity, name="embedding_similarity")
    if "L2" in sim_methods:
        # L2 Distance
        l2_similarity = m.addVar(lb=0, name="l2_distance")
        m.addConstr(l2_similarity*l2_similarity == gp.quicksum((phi[max_class]-embedding)*(phi[max_class]-embedding)), name="l2_similarity")
        regularizers["L2"] = l2_similarity
    if "Squared L2" in sim_methods:
        # Squared L2 Distance
        squared_l2_similarity = m.addVar(lb=0, name="l2_distance")
        m.addConstr(squared_l2_similarity == gp.quicksum((phi[max_class]-embedding)*(phi[max_class]-embedding)), name="l2_similarity")
        regularizers["Squared L2"] = squared_l2_similarity

    

            
    ## MIQCP objective function
    m.setObjective(output_vars["Output"][0, max_class]+sum(sim_weights[sim_method]*regularizers[sim_method] for sim_method in sim_methods), GRB.MAXIMIZE)

    m.update()
    m.write("model.lp") # Save model file
    m.write("model.mps")

    # Define the callback function
    solutions = []
    def callback(model, where):
        global A, X, solutions
        if where == GRB.Callback.MIPSOL:
            # A new incumbent solution has been found
            print(f"New incumbent solution found (ID {len(solutions)}) Objective value: {model.cbGet(GRB.Callback.MIPSOL_OBJ)}")

            X_sol = model.cbGetSolution(X)
            A_sol = model.cbGetSolution(A)
            output_var_value = model.cbGetSolution(output_vars["Output"])

            ## Sanity Check
            sol_output = nn.forwardXA(X_sol, A_sol).detach().numpy()
            print("NN output given X:", sol_output.squeeze())
            print("predicted output:", output_var_value.squeeze())

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
                "time": time.time(),
                "Objective Value": model.cbGet(GRB.Callback.MIPSOL_OBJ),
                "Upper Bound": model.cbGet(GRB.Callback.MIPSOL_OBJBND),
            }
            solution.update(similarities)
            solutions.append(solution)

            if log_run:
                fig, _ = draw_graph(A_sol, X_sol)
                wandb.log(solutions[-1], commit=False)
                wandb.log({f"Output Logit {i}": sol_output[0, i] for i in range(sol_output.shape[1])}, commit=False)
                wandb.log({"fig": wandb.Image(fig)})
                plt.close()

            with open(output_file, "wb") as f:
                pickle.dump(solutions, f)

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
    all_ub, all_lb = [], []
    assert len(all_outputs) == len(output_vars), (len(all_outputs), len(output_vars))
    for var, name_output in zip(output_vars.values(), all_outputs):
        layer_name = name_output[0]
        output = name_output[1].detach().numpy()
        # var.setAttr("lb", var.getAttr("lb").clip(min=-128, max=128))
        # var.setAttr("ub", var.getAttr("ub").clip(min=-128, max=128))
        all_lb.extend(var.getAttr("lb").flatten().tolist())
        all_ub.extend(var.getAttr("ub").flatten().tolist())
        assert var.shape == output.shape, (layer_name, var.shape, output.shape)
        assert np.less_equal(var.getAttr("lb"), output).all(), (layer_name, var.getAttr("lb"), output, np.greater(var.getAttr("lb"), output).sum())
        assert np.greater_equal(var.getAttr("ub"), output).all(), (layer_name, var.getAttr("ub"), output, np.lesser(var.getAttr("ub"), output).sum())
        var.Start = output
        if layer_name == "Aggregation":
            for sim_method in sim_methods:
                if sim_method == "Cosine":
                    regularizers[sim_method].Start = np.dot(output[0], phi[max_class])/(np.linalg.norm(output[0])*np.linalg.norm(phi[max_class]))
                elif sim_method == "L2":
                    regularizers[sim_method].Start = np.sqrt(sum((output[0] - phi[max_class])*(output[0]- phi[max_class])))
                elif sim_method == "Squared L2":
                    regularizers[sim_method].Start = sum((output[0] - phi[max_class])*(output[0] - phi[max_class]))
        #     m.addConstr(gp.quicksum((output - var)*(output - var)) <= 0.1) #######################################
    m.update()
    print((np.array(all_ub) < 0).sum()/len(all_ub))
    print(min(all_lb), max(all_ub))

    # Use tuned parameters
    m.read("./tune0.prm")
    
    # # Tune
    # print("Tuning")
    # m.setParam("TuneTimeLimit", 3600*48)
    # m.tune()
    # for i in range(m.tuneResultCount):
    #     m.getTuneResult(i)
    #     m.write('tune'+str(i)+'.prm')
    # print("Done Tuning")

    m.setParam("TimeLimit", 3600*24)
    # m.setParam("PreQLinearize", 2) # TODO: Chose between 1 and 2
    m.optimize(callback)

    with open(output_file, "wb") as f:
        pickle.dump(solutions, f)

    print("Status:", m.Status)

    if m.Status in [3, 4]: # If the model is infeasible, see why
        m.computeIIS()
        m.write("model.ilp")
    else:
        print("NN output given X", nn.forwardXA(X.X, A.X))
        print("predicted output", output_vars["Output"].X)

print(time.strftime("\nEnd Time: %H:%M:%S", time.localtime()))