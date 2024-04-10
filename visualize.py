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
import random
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from PIL import Image
import io

pio.templates.default = "plotly_white"
random.seed(0)

# %%
methods = ["MIPExplainer", "GNNInterpreter", "XGNN", "PAGE"]
datasets = {
    dataset_name: get_dataset(dataset_name)
    for dataset_name in ["MUTAG", "Is_Acyclic_Ones", "Shapes_Ones"]
}

replace_d = {
    "runtime": "Runtime (s)",
    "consistency": "Average Edit Distance",
    "avg_consistency": "Average Edit Distance",
    "num_nodes": "\\# Nodes",
    "max_class": "Class",
    "Is_Acyclic_Ones": "Is_Acyclic",
    "Shapes_Ones": "Shapes",
    "method": "Method",
    "dataset_name": "Dataset",
    "_": "\\_",
    "NaN": "N/A",
}


def replace_all(text):
    for i, j in replace_d.items():
        text = text.replace(i, j)
    return text


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
    for i in range(1, len(Gs)):
        for j in range(i):
            edit_distances.append(
                nx.graph_edit_distance(
                    Gs[i],
                    Gs[j],
                    node_match=lambda x, y: np.isclose(x["label"], y["label"]).all(),
                )
            )
    return np.mean(edit_distances), np.std(edit_distances)


consistency, runtimes, avg_consistency = None, None, None
for dataset_name in ["MUTAG", "Is_Acyclic_Ones", "Shapes_Ones"]:

    def vis_nx(G):
        A = nx.to_numpy_array(G)
        # X = np.stack([np.eye(datasets[dataset_name].num_node_features)[G.nodes[i]["label"]] for i in G.nodes()])
        if G.number_of_nodes() == 0:
            return Image.new("RGBa", (800, 600), "white")
        X = np.stack([G.nodes[i]["label"] for i in G.nodes()])
        if X.ndim == 1:
            X = np.eye(datasets[dataset_name].num_node_features)[X]
        assert (
            X.shape[0] == G.number_of_nodes()
            and X.shape[1] == datasets[dataset_name].num_node_features
        )
        fig, ax = datasets[dataset_name].draw_graph(A=A, X=X)
        fig.canvas.draw()
        img = Image.frombytes(
            "RGBa", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")
        # img_byte_arr = img_byte_arr.getvalue()
        return img
        # plt.close()
        # return img

    print(f"Processing {dataset_name}")

    gdir = f"./results/old_runs_{dataset_name}/"
    d_list = []
    for filename in tqdm(os.listdir(gdir), desc="Loading Run Files"):
        try:
            d = pickle.load(open(gdir + filename, "rb"))
            if d["dataset_name"] != dataset_name:
                continue
            d["mip_information"] = d["mip_information"][
                :: max(1, len(d["mip_information"]) // 1000)
            ]
            # pickle.dump(d, open(gdir+filename, "wb"))
            # del d["mip_information"]
            d["run_id"] = filename.split(".")[0]
            d["max_class_name"] = datasets[d["dataset_name"]].GRAPH_CLS[d["max_class"]]
            d_list.append(d)
        except Exception as e:
            print(e)
            breakpoint()

        # for d in d_list:
        #     if not d["num_nodes"] == 7 or not d["max_class"] == 1:
        #         continue
        #     datasets[dataset_name].draw_graph(d["solutions"][-1]["A"], d["solutions"][-1]["X"], f"./results/figures/{dataset_name}_class_{d['max_class']}_n_7_id_{d['run_id']}.png")
    # %%
    mip_info = None
    for d in d_list:
        if not d["num_nodes"] == 7 or not d["max_class"] == 1:
            continue
        m = pd.DataFrame(
            d["mip_information"][:: max(1, len(d["mip_information"]) // 1000)]
        )
        m["run_id"] = d["run_id"]
        if mip_info is None:
            mip_info = m
        else:
            mip_info = pd.concat([mip_info, m], axis=0)
    mip_info.rename(
        columns={
            "Runtime": "Runtime (s)",
            "BestBound": "Objective Bound",
            "ObjBound": "Best Objective",
            "WorkUnits": "Work Units",
            "ExploredNodeCount": "Explored Node Count",
            "UnexploredNodeCount": "Unexplored Node Count",
        },
        inplace=True,
    )
    ## sample 3 run_ids
    # breakpoint()
    mip_info = mip_info[
        mip_info["run_id"].isin(random.choices(mip_info["run_id"].unique(), k=3))
    ]
    print("Generating Figures")
    for fname, yaxisname, y_list in [
        ("convergence", "Objective Value", ["Best Objective", "Objective Bound"]),
        (
            "nodecount",
            "Number of Nodes",
            ["Explored Node Count", "Unexplored Node Count"],
        ),
    ]:
        fig = px.line(
            mip_info,
            x="Runtime (s)",
            y=y_list,
            title="",
            width=700,
            height=700,
            log_y=False,  # (fname == "convergence"),
            facet_row="run_id",
            color_discrete_sequence=px.colors.qualitative.Plotly
            if fname == "convergence"
            else px.colors.qualitative.Dark2,
        )
        fig.update_layout(
            font=dict(
                family="Nimbus Roman",
                size=50,
                color="rgb(82, 82, 82)",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.01),
            showlegend=False,
            autosize=False,
            margin=dict(
                autoexpand=True,
                l=120,
                r=20,
                t=20,
            ),
        )
        fig.update_xaxes(
            ticks="outside",
            # range=[0, mip_info["Runtime (s)"].max()],
            tickfont=dict(
                size=24,
                color="rgb(82, 82, 82)",
            ),
        )
        fig.update_yaxes(
            title="",
            # range=[mip_info[y_list].min().min(), mip_info[y_list].max().max()],
            # dtick=2,
            tickfont=dict(
                size=24,
                color="rgb(82, 82, 82)",
            ),
        )
        fig.update_layout(legend_title_text="Legend")
        # Save figure
        fig.for_each_annotation(lambda a: a.update(text=""))
        fig.for_each_yaxis(lambda y: y.update(title=""))
        # and:
        fig.add_annotation(
            x=-0.13,
            y=0.5,
            text=yaxisname,
            textangle=-90,
            xref="paper",
            yref="paper",
            font=dict(
                family="Nimbus Roman",
                size=55,
                color="rgb(82, 82, 82)",
            ),
        )
        if fname == "convergence":
            fig.add_hline(
                y=mip_info["Best Objective"].max(),
                line_dash="dot",
                line_color="black",
                annotation_position="bottom right",
                annotation_text="",
            )
        fig.update_yaxes(matches=None)
        fig.write_image(
            f"./results/figures/{fname}_{d['dataset_name']}_class_{d['max_class']}_n_7.png",
            format="png",
        )

    index_names = ["dataset_name", "max_class", "num_nodes", "method"]

    # %%
    mipexplainer_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in d.items()
                if isinstance(value, numbers.Number)
                or key in {"run_id", "dataset_name"}
            }
            for d in d_list
        ]
    )

    mipexplainer_df["G"] = [
        create_graph(d["solutions"][-1]["A"], d["solutions"][-1]["X"]) for d in d_list
    ]
    mipexplainer_df["init_G"] = [
        create_graph(d["solutions"][0]["A"], d["solutions"][0]["X"]) for d in d_list
    ]
    mipexplainer_df["method"] = "MIPExplainer"

    mipexplainer_df = mipexplainer_df[mipexplainer_df["dataset_name"] == dataset_name]

    # %%
    gnninterpreter_df = pd.DataFrame(
        pickle.load(open(f"results/gnninterpreter_{dataset_name}.pkl", "rb"))
    )
    xgnn_df = pd.DataFrame(pickle.load(open(f"results/XGNN_{dataset_name}.pkl", "rb")))
    page_df = pd.DataFrame(pickle.load(open(f"results/PAGE_{dataset_name}.pkl", "rb")))
    if dataset_name == "Shapes_Ones":
        page_df = pd.concat(
            [
                page_df,
                pd.DataFrame(
                    pickle.load(open("results/PAGE_Shapes_Ones_star_reduced.pkl", "rb"))
                ),
            ]
        )
    page_df["G"] = page_df["G"].apply(
        lambda x: create_graph(to_dense_adj(x.edge_index).squeeze(), x.x)
    )
    print(page_df.groupby(index_names).count())
    if dataset_name == "Shapes_Ones":
        breakpoint()
    page_df = page_df.groupby(index_names, group_keys=False).apply(
        lambda x: x.sample(5, random_state=0)
    )  # TODO:: Move this to the whole df

    print(
        f"Dataframe Lengths:\n  MIPExplainer: {len(mipexplainer_df)}\n  GNNInterpreter: {len(gnninterpreter_df)}\n  XGNN: {len(xgnn_df)}\n  PAGE: {len(page_df)}"
    )

    df = pd.concat([mipexplainer_df, gnninterpreter_df, xgnn_df, page_df])

    df = df[df["max_class"] != 4]
    df = df[df["num_nodes"] <= 8]

    df["method"] = df["method"].map(
        {
            "xgnn": "XGNN",
            "MIPExplainer": "MIPExplainer",
            "gnninterpreter": "GNNInterpreter",
            "PAGE": "PAGE",
        }
    )
    df["max_class"] = df["max_class"].map(datasets[dataset_name].GRAPH_CLS)
    df.rename(
        columns={
            f"Output Logit {i}": f"{datasets[dataset_name].GRAPH_CLS[i]} Output Logit"
            for i in range(datasets[dataset_name].num_classes)
        },
        inplace=True,
    )

    writer = pd.ExcelWriter(f"results/all_{dataset_name}.xlsx", engine="xlsxwriter")
    df.dropna(axis=1).drop(["G"], axis=1).to_excel(
        writer, sheet_name="Sheet1", index=False
    )
    worksheet = writer.sheets["Sheet1"]
    img_col = worksheet.dim_colmax + 1

    for i in range(len(df)):
        img = vis_nx(df["G"].iloc[i])
        img = img.resize((img.size[0] // 5, img.size[1] // 5))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        worksheet.set_column_pixels(img_col, img.size[0])
        worksheet.set_row_pixels(i + 1, img.size[1])
        worksheet.insert_image(
            i + 1, img_col, f"Index: {i}", {"image_data": img_byte_arr}
        )
    worksheet.write(0, img_col, "Explanation")
    worksheet.autofit()
    writer.close()

    df = df.set_index(index_names).sort_index()
    df["Image"] = df["G"].map(vis_nx)
    df.to_pickle(f"results/all_{dataset_name}.pkl")

    # breakpoint()

    # %%
    a = df[[c for c in df.columns if "Output Logit" in c][:4]]
    # a = a.div(a.sum(axis=1)**2, axis=0)
    logit_table = a.groupby(index_names).mean()
    logit_std_table = a.groupby(index_names).std()
    combined_table = (
        logit_table.map("{0:.3f}".format)
        + " $\\pm$ "
        + logit_std_table.map("{0:.3f}".format)
    )
    combined_table = combined_table.reindex(methods, level=3)
    with open(f"results/tables/output_logit_{dataset_name}.tex", "w") as f:
        f.write(replace_all(combined_table.loc[dataset_name].to_latex(index=True)))
    with open(f"results/tables/output_logit_{dataset_name}_flipped.tex", "w") as f:
        tempdf = pd.DataFrame(combined_table)
        f.write(
            replace_all(
                tempdf.stack()
                .unstack(level=tempdf.index.names.index("method"))
                .loc[dataset_name]
                .to_latex(index=True)
            )
        )

    # %%
    runtime_table = df.groupby(index_names)["runtime"].mean()
    runtime_std_table = df.groupby(index_names)["runtime"].std()
    combined_table = (
        runtime_table.map("{0:.3f}".format)
        + " $\\pm$ "
        + runtime_std_table.map("{0:.3f}".format)
    )
    combined_table = combined_table.reindex(methods, level=3)
    with open(f"results/tables/runtime_{dataset_name}.tex", "w") as f:
        f.write(replace_all(combined_table.loc[dataset_name].to_latex(index=True)))
    runtime_table
    if runtimes is None:
        runtimes = combined_table.copy()
    else:
        runtimes = pd.concat([runtimes, combined_table.copy()])

    # %%
    distances = []
    distances_std = []
    for name, group in tqdm(
        df.groupby(index_names)["G"], desc="Calculating Consistency"
    ):
        # Save the average edit distance of the group to a df
        group = list(group)
        m, std = average_edit_distance(group)
        distances.append({"Consistency": m} | dict(zip(index_names, name)))
        distances_std.append({"Consistency": std} | dict(zip(index_names, name)))
    distances_df = pd.DataFrame(distances).set_index(index_names).sort_index()
    distances_std_df = pd.DataFrame(distances_std).set_index(index_names).sort_index()

    combined_table = (
        distances_df.map("{0:.3f}".format)
        + " $\\pm$ "
        + distances_std_df.map("{0:.3f}".format)
    )
    combined_table = combined_table.reindex(methods, level=3)
    # %%
    with open(f"results/tables/consistency_{dataset_name}.tex", "w") as f:
        f.write(replace_all(combined_table.loc[dataset_name].to_latex(index=True)))

    if consistency is None:
        consistency = combined_table.copy()
    else:
        consistency = pd.concat([consistency, combined_table.copy()])

    # %%
    # Average over num_nodes
    averaged_distances_df = distances_df.groupby(
        ["dataset_name", "max_class", "method"]
    ).mean()  # .loc[dataset_name]
    averaged_distances_df_std = distances_df.groupby(
        ["dataset_name", "max_class", "method"]
    ).std()

    combined_table = (
        averaged_distances_df.map("{0:.2}".format)
        + " $\\pm$ "
        + averaged_distances_df_std.map("{0:.2f}".format)
    )
    combined_table = combined_table.reindex(methods, level=2)
    with open(f"results/tables/averaged_consistency_{dataset_name}.tex", "w") as f:
        f.write(replace_all(combined_table.loc[dataset_name].to_latex(index=True)))

    if avg_consistency is None:
        avg_consistency = combined_table.copy()
    else:
        avg_consistency = pd.concat([avg_consistency, combined_table.copy()])

for name, table in zip(
    ["all_runtimes", "all_consistency", "all_avg_consistency"],
    [runtimes, consistency, avg_consistency],
):
    with open(f"results/tables/{name}.tex", "w") as f:
        f.write(replace_all(table.to_latex(index=True)))
    with open(f"results/tables/{name}_flipped.tex", "w") as f:
        tempdf = pd.DataFrame(table)
        f.write(
            replace_all(
                tempdf.unstack(level=tempdf.index.names.index("method")).to_latex(
                    index=True
                )
            )
        )
