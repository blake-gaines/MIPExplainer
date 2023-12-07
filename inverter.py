import gurobipy as gp
from gurobipy import GRB
import pickle
from collections import OrderedDict
import numpy as np
import warnings
from itertools import product


class ObjectiveTerm:
    def __init__(self, name, var, weight=1, calc=None, required_vars=[]):
        self.name = name
        self.var = var
        self.weight = weight
        self.hybrid_calc = calc
        self.required_vars = required_vars


class Inverter:
    def __init__(
        self,
        args,
        nn,
        dataset,
        env,
        convert_inputs_func=None,
        model_name="model",
    ):
        self.args = args
        self.nn = nn
        self.dataset = dataset
        self.env = env
        self.convert_inputs_func = convert_inputs_func
        self.model_name = model_name
        self.m = gp.Model(model_name, env)
        self.model = self.m
        self.output_vars = OrderedDict()
        self.objective = 0
        self.objective_terms = dict()
        self.all_vars = dict()
        self.solutions = []
        self.tracked_vars = dict()
        self.input_vars = dict()

    def convert_inputs(self, **kwargs):
        if self.convert_inputs_func is not None:
            return self.convert_inputs_func(**kwargs)
        else:
            return kwargs

    def set_tracked_vars(self, tracked_vars):
        self.tracked_vars = tracked_vars
        self.all_vars.update(tracked_vars)

    def set_input_vars(self, input_vars):
        self.input_vars = input_vars
        self.all_vars.update(input_vars)

    # def set_output_vars(self, output_vars):
    #     self.output_vars = output_vars
    #     self.all_vars.update(output_vars)

    def save_model(self, log_files=False, exts=["lp", "mps"]):
        if isinstance(exts, str):
            exts = [exts]
        file_names = [f"{self.model_name}.{ext}" for ext in exts]
        for file_name in file_names:
            self.m.write(file_name)
        return file_names if len(file_names) > 1 else file_names[0]

    def save_inverter(self, filename="inverter.pkl"):
        everything = dict()
        everything["output_keys"] = {
            key: var.varName for key, var in self.output_vars.items()
        }
        pickle.dump(everything, open(filename, "wb"))

    def load_inverter(self, filename="inverter.pkl"):
        everything = pickle.load(open(filename, "rb"))
        get_var_matrix = np.vectorize(self.m.getVarByName)
        for key, name_matrix in everything["output_keys"].items():
            self.output_vars[key] = gp.MVar.fromlist(
                get_var_matrix(name_matrix).tolist()
            )

    def load_model(self, file_name="model.mps", input_var_names=[]):
        self.m = gp.read(file_name)
        self.set_input_vars(
            {name: self.m.getVarByName(name) for name in input_var_names}
        )

    def get_mvar(self, name, shape):
        X = np.empty(shape, dtype=gp.Var)
        for index in product(*[range(d) for d in shape]):
            X[index] = self.m.getVarByName(f"{name}[{','.join(str(i) for i in index)}]")
        return gp.MVar.fromlist(X.tolist())

    def solve(self, callback=None, param_file=None, output_file=None, **kwargs):
        if callback is None:
            callback = self.get_default_callback()
        # # Check variables are bounded or binary
        # for var in self.m.getVars():
        #     assert var.vtype == GRB.BINARY or (
        #         var.LB != float("-inf") and var.UB != float("inf")
        #     ), f"Variable {var.VarName} is unbounded."

        for param_name, param_value in kwargs.items():
            self.m.setParam(param_name, param_value)

        self.solutions = []
        if param_file:
            self.m.read(param_file)

        self.m.optimize(callback)

        if output_file:
            with open(self.args.output_file, "wb") as f:
                pickle.dump(self.solutions, f)

    def computeIIS(self, output_fname=None):
        if output_fname is None:
            output_fname = f"{self.model_name}.ilp"
        self.m.computeIIS()
        self.m.write(output_fname)
        print(f"Wrote IIS to {output_fname}")
        return output_fname

    def get_default_callback(self):
        def solver_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                solution_inputs = {
                    name: model.cbGetSolution(var)
                    for name, var in self.input_vars.items()
                }

                last_output_key = next(reversed(self.output_vars))

                nn_output = (
                    dict(
                        self.nn.get_all_layer_outputs(
                            **self.convert_inputs(**solution_inputs)
                        )
                    )[last_output_key]
                    .detach()
                    .numpy()
                )
                output_var_value = model.cbGetSolution(
                    self.output_vars[last_output_key]
                )
                # breakpoint()
                if not np.allclose(nn_output, output_var_value):
                    "uh oh :("
                    warnings.warn(
                        f"Model outputs diverge: max difference is {np.abs(nn_output-output_var_value).max():.3e}"
                    )

                objective_term_values = {
                    name: self.m.cbGetSolution(term.var)
                    for name, term in self.objective_terms.items()
                }

                tracked_var_values = {
                    name: model.cbGetSolution(var)
                    for name, var in self.tracked_vars.items()
                }

                for name, var_value in objective_term_values.items():
                    term = self.objective_terms[name]
                    if not hasattr(term, "calc"):
                        continue
                    real_value = term.calc(
                        *[
                            self.m.cbGetSolution(req_var)
                            for req_var in term.required_vars
                        ]
                    )
                    if not np.allclose(var_value, real_value):
                        warnings.warn(
                            f"The value of the variable representing objective term {name} has diverged from the actual value",
                            action="once",
                        )
                    ## TODO: Something with this

                solution = (
                    solution_inputs
                    | tracked_var_values
                    | objective_term_values
                    | {
                        "Output": nn_output,
                        "Objective Value": self.m.cbGet(GRB.Callback.MIPSOL_OBJ),
                        "Upper Bound": self.m.cbGet(GRB.Callback.MIPSOL_OBJBND),
                    }
                )
                self.solutions.append(solution)

        return solver_callback

    def add_objective_term(self, term, weight=1):
        self.objective_terms[term.name] = term
        self.objective += term.var * term.weight
        self.m.setObjective(self.objective, GRB.MAXIMIZE)

    def warm_start(self, input_var_values, debug_mode=False):
        self.m.NumStart = 1
        for input_name, value in input_var_values.items():
            self.input_vars[input_name].Start = value.detach().numpy()
        all_outputs = dict(
            self.nn.get_all_layer_outputs(**self.convert_inputs(**input_var_values))[1:]
        )

        all_ub, all_lb = [], []
        # assert len(all_outputs) == len(self.output_vars), (
        #     len(all_outputs),
        #     len(self.output_vars),
        # )

        for layer_name in self.output_vars.keys():
            var = self.output_vars[layer_name]
            output = all_outputs[layer_name].detach().numpy()

            # Allows us to check ranges for bounds
            all_lb.extend(var.getAttr("lb").flatten().tolist())
            all_ub.extend(var.getAttr("ub").flatten().tolist())

            # Check variables and ouputs have the same shape
            assert var.shape == output.shape, (layer_name, var.shape, output.shape)
            # # Check initializations for all variables are geq the lower bounds
            # assert np.less_equal(var.getAttr("lb"), output).all(), (
            #     layer_name,
            #     f'Lower Bounds: {var.getAttr("lb")[np.greater(var.getAttr("lb"), output)]}, Outputs: {output[np.greater(var.getAttr("lb"), output)]}',
            #     np.greater(var.getAttr("lb"), output).sum(),
            # )
            # # Check initializations for all variables are leq the upper bounds
            # assert np.greater_equal(var.getAttr("ub"), output).all(), (
            #     layer_name,
            #     var.shape,
            #     var.getAttr("ub").max(),
            #     output.max(),
            #     np.less(var.getAttr("ub"), output).sum(),
            # )

            var.Start = output

            if debug_mode:
                self.m.addConstr(var == output, name=f"fixing_constraint_{layer_name}")

        self.m.update()

        for term in self.objective_terms:
            if hasattr(term, "calc"):
                term.calc(*[req_var.Start for req_var in term.required_vars])

        return {
            "Lowest Lower Bound": min(all_lb),
            "Highest Upper Bound": max(all_ub),
            "Min ABS Bound": min([b for b in np.abs(all_lb + all_ub) if b > 0]),
        }

    def tune(self, callback=None, **kwargs):
        for param_name, param_value in kwargs.items():
            self.m.setParam(param_name, param_value)
            self.m.tune()
            for i in range(self.m.tuneResultCount):
                self.m.getTuneResult(i)
                self.m.write("tune" + str(i) + ".prm")
