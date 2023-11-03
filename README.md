
# Explain GNNs with Discrete Optimization

In-progress implementation of methods to generate explanitory graphs for trained GNN models by encoding them as mixed-integer programming problems. Requires a [Gurobi](https://www.gurobi.com/) license (free for academic use).

## Repository Structure
* [gnn.py](./gnn.py): GNN with methods needed for compatibility with explanation generation code. Running `python gnn.py` trains a GNN 
* [invert.py](./invert.py): Contains methods for adding encodings for various NN and GNN layers into a Gurobi model
* [main.py](./main.py): Loads in a GNN model, encodes it as a MIP, and solves.
* [generate_data.ipynb](./generate_data.ipynb): Generates datasets for testing explanation methods
* [utils.py](./utils.py) Contains small helper functions
* \*.prm: Files that store parameters controlling the behavior of the MIP solver

## Run Locally

Clone the project

```bash
  git clone https://github.com/blake-gaines/Invert-GNNs.git
```

Go to the project directory

```bash
  cd Invert-GNNs
```

Install dependencies listed in [environment.yml](./environment.yml). If using Conda you can run the following command:
```bash
  conda env create -f environment.yml
```
