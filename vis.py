# %%
import pickle
import os
import pandas as pd
import numbers
import networkx as nx
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
from datasets import get_dataset

pio.templates.default = "plotly_white"

# %%
datasets = {dataset_name: get_dataset(dataset_name) for dataset_name in ["Is_Acyclic_Ones", "MUTAG", "Shapes_Ones"]}
dataset_name = "Is_Acyclic_Ones"

# %%
def create_graph(adjacency_matrix, node_features):
    """
    Create a NetworkX graph from a numpy adjacency matrix and node feature matrix.

    Parameters:
    - adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph.
    - node_features (numpy.ndarray): The matrix of node features.

    Returns:
    - nx.Graph: The created NetworkX graph.
    """

    # Create an empty graph
    G = nx.Graph()

    # Get the number of nodes in the graph
    num_nodes = adjacency_matrix.shape[0]

    # Add nodes to the graph with corresponding features
    for i in range(num_nodes):
        G.add_node(i, label=node_features[i])

    # Add edges to the graph based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(i, j)

    return G

def average_edit_distance(Gs):
    # Get the average edit distance between all pairs of graphs
    edit_distances = []
    for i in range(1,len(Gs)):
        for j in range(i):
            edit_distances.append(nx.graph_edit_distance(Gs[i], Gs[j], node_match=lambda x, y: np.isclose(x['label'], y['label']).all()))
    return np.mean(edit_distances), np.std(edit_distances)

consistency, runtimes, avg_consistency = None, None, None
for dataset_name in ["Is_Acyclic_Ones", "MUTAG", "Shapes_Ones"]:

    gdir = f"./results/runs_{dataset_name}/"
    d_list = []
    for filename in tqdm(os.listdir(gdir), desc="Loading Run Files"):
        try:
            d = pickle.load(open(gdir+filename, "rb"))
            if d["dataset_name"] != dataset_name:
                continue
            d["mip_information"] = d["mip_information"][::max(1, len(d["mip_information"])//1000)]
            pickle.dump(d, open(gdir+filename, "wb"))
            del d["mip_information"]
            d["run_id"] = filename.split(".")[0]
            d["max_class_name"] = datasets[d["dataset_name"]].GRAPH_CLS[d["max_class"]]
            d_list.append(d)
        except Exception as e:
            print(e)
            continue

    # %%
    # mip_info = None
    # for d in d_list:
    #     if not d["num_nodes"] == 7 or not d["max_class"] == 1:
    #         continue
    #     m= pd.DataFrame(d["mip_information"][::max(1, len(d["mip_information"])//100)])
    #     m["run_id"] = d["run_id"]
    #     if mip_info is None:
    #         mip_info = m
    #     else:
    #         mip_info = pd.concat([mip_info, m], axis=0)
    # mip_info.rename(columns={"BestBound": "Best Bound", "ObjBound": "Objective Bound", "WorkUnits": "Work Units", "ExploredNodeCount": "Explored Node Count", "UnexploredNodeCount": "Unexplored Node Count"}, inplace=True)
    # print("Generating Figures")
    # for fname, yaxisname, y_list in [("convergence", "Objective Value", ["Best Bound", "Objective Bound"]), ("nodecount", "Number of Nodes", ["Explored Node Count", "Unexplored Node Count"])]:
    #     fig = px.line(mip_info,
    #     x="Work Units", 
    #     y=y_list, 
    #     title="", 
    #     width=1000, 
    #     height=1000,
    #     log_y = False,
    #     facet_row="run_id")
    #     fig.update_layout(
    #         font=dict(
    #             family= "Nimbus Roman",
    #             size=32,
    #             color='rgb(82, 82, 82)',
    #         ),
    #     # legend=dict(
    #     #     visible=False
    #     # ),
    #     showlegend=False,
    #     autosize=False,
    #     margin=dict(
    #         autoexpand=True,
    #         l=120,
    #         r=20,
    #         t=20,
    #     ),
    #     )   
    #     fig.update_xaxes(
    #         ticks='outside',
    #         tickfont=dict(
    #             # size=18,
    #             color='rgb(82, 82, 82)',
    #     ),)
    #     fig.update_yaxes(
    #         title="",
    #         # dtick=2,
    #         tickfont=dict(
    #             # size=18,
    #             color='rgb(82, 82, 82)',
    #     ),)
    #     fig.update_layout(legend_title_text='Legend')
    #     # Save figure
    #     fig.for_each_annotation(lambda a: a.update(text=""))
    #     fig.for_each_yaxis(lambda y: y.update(title = ''))
    #     # and:
    #     fig.add_annotation(x=-0.1,y=0.5,
    #                     text=yaxisname, textangle=-90,
    #                         xref="paper", yref="paper")
    #     print("Writing Image")
    #     fig.write_image(f"./results/figures/{fname}_{d['dataset_name']}_class_{d['max_class']}_n_{d['num_nodes']}_id_{d['run_id']}.png", format="png")
    #     # fig.write_html(f"./results/figures/convergence_{d['dataset_name']}_class_{d['max_class']}_n_{d['num_nodes']}_id_{d['run_id']}.html") 
    #     print("Done Writing Image")
    #     # fig.show()
    # # import sys; sys.exit()

    # %%
    mipexplainer_df = pd.DataFrame([{key: value for key, value in d.items() if isinstance(value, numbers.Number) or key in {"run_id", "dataset_name"}} for d in d_list])

    mipexplainer_df["G"] = [create_graph(d["solutions"][-1]["A"], d["solutions"][-1]["X"]) for d in d_list]
    mipexplainer_df["init_G"] = [create_graph(d["solutions"][0]["A"], d["solutions"][0]["X"]) for d in d_list]
    mipexplainer_df["method"] = "MIPExplainer"

    mipexplainer_df = mipexplainer_df[mipexplainer_df["dataset_name"]==dataset_name]

    # %%
    gnninterpreter_df = pd.DataFrame(pickle.load(open(f"results/gnninterpreter_{dataset_name}.pkl", "rb")))
    xgnn_df = pd.DataFrame(pickle.load(open(f"results/XGNN_{dataset_name}.pkl", "rb")))

    df = pd.concat([mipexplainer_df, gnninterpreter_df, xgnn_df])
    index_names = ["dataset_name", "max_class", "num_nodes", "method"] 

    df = df[df["max_class"] != 4]
    df = df[df["num_nodes"] <= 8]

    df["method"] = df["method"].map({"xgnn": "XGNN", "MIPExplainer": "MIPExplainer", "gnninterpreter": "GNNInterpreter"})
    df["max_class"] = df["max_class"].map(datasets[dataset_name].GRAPH_CLS)

    df = df.set_index(index_names).sort_index()

    # %%
    df.rename(columns={f"Output Logit {i}": f"{datasets[dataset_name].GRAPH_CLS[i]} Output Logit" for i in range(datasets[dataset_name].num_classes)}, inplace=True)
    a = df[[c for c in df.columns if "Output Logit" in c]]
    # a = a.div(a.sum(axis=1)**2, axis=0)
    logit_table = a.groupby(index_names).mean()
    logit_std_table = a.groupby(index_names).std()
    combined_table = logit_table.map("{0:.3f}".format) + " $\\pm$ " + logit_std_table.map("{0:.3f}".format)
    combined_table = combined_table.reindex(["MIPExplainer", "GNNInterpreter", "XGNN"], level=3)
    with open(f"results/tables/output_logit_{dataset_name}.tex", "w") as f: 
        f.write(combined_table.loc[dataset_name].to_latex(index=True).replace("_", "\\_"))
    logit_table

    # %%
    runtime_table = df.groupby(index_names)["runtime"].mean()
    runtime_std_table = df.groupby(index_names)["runtime"].std()
    combined_table = runtime_table.map("{0:.3f}".format) + " $\\pm$ " + runtime_std_table.map("{0:.3f}".format)
    combined_table = combined_table.reindex(["MIPExplainer", "GNNInterpreter", "XGNN"], level=3)
    with open(f"results/tables/runtime_{dataset_name}.tex", "w") as f:
        f.write(combined_table.loc[dataset_name].to_latex(index=True).replace("_", "\\_"))
    runtime_table
    if runtimes is None:
        runtimes = combined_table.copy()
    else:
        runtimes = pd.concat([runtimes, combined_table.copy()])

    # %%
    distances = []
    distances_std = []
    for name, group in tqdm(df.groupby(index_names)["G"]):
        # Save the average edit distance of the group to a df
        group = list(group)
        m, std = average_edit_distance(group)
        distances.append({"Consistency": m} | dict(zip(index_names, name)))
        distances_std.append({"Consistency": std} | dict(zip(index_names, name)))
    distances_df = pd.DataFrame(distances).set_index(index_names).sort_index()
    distances_std_df = pd.DataFrame(distances_std).set_index(index_names).sort_index()

    combined_table = distances_df.map("{0:.3f}".format) + " $\\pm$ " + distances_std_df.map("{0:.3f}".format)
    combined_table = combined_table.reindex(["MIPExplainer", "GNNInterpreter", "XGNN"], level=3)
    # %%
    with open(f"results/tables/consistency_{dataset_name}.tex", "w") as f:
        f.write(combined_table.loc[dataset_name].to_latex(index=True).replace("_", "\\_"))
    
    if consistency is None:
        consistency = combined_table.copy()
    else:
        consistency = pd.concat([consistency, combined_table.copy()])

    # %%
    # Average over num_nodes
    averaged_distances_df = distances_df.groupby(["dataset_name", "max_class", "method"]).mean()#.loc[dataset_name]
    averaged_distances_df_std = distances_df.groupby(["dataset_name", "max_class", "method"]).std()

    combined_table = averaged_distances_df.map("{0:.2}".format) + " $\\pm$ " + averaged_distances_df_std.map("{0:.2f}".format)
    combined_table = combined_table.reindex(["MIPExplainer", "GNNInterpreter", "XGNN"], level=2)
    with open(f"results/tables/averaged_consistency_{dataset_name}.tex", "w") as f:
        f.write(combined_table.loc[dataset_name].to_latex(index=True).replace("_", "\\_"))

    if avg_consistency is None:
        avg_consistency = combined_table.copy()
    else:
        avg_consistency = pd.concat([avg_consistency, combined_table.copy()])

for name, table in zip(["all_runtimes", "all_consistency", "all_avg_consistency"], [runtimes, consistency, avg_consistency]):
    with open(f"results/tables/{name}.tex", "w") as f:
        f.write(table.to_latex(index=True).replace("_", "\\_"))
    with open(f"results/tables/{name}_flipped.tex", "w") as f:
        tempdf = pd.DataFrame(table)
        breakpoint()
        f.write(tempdf.unstack(level=tempdf.index.names.index("method")).to_latex(index=True).replace("_", "\\_"))