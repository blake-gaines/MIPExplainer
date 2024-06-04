import gurobipy as gp
from gurobipy import GRB
import pickle
from collections import OrderedDict
import numpy as np
import warnings
from itertools import product
from invert_utils import invert_torch_layer


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
        optimization_sense=GRB.MAXIMIZE,
        verbose=1,
    ):
        self.args = args
        self.nn = nn
        self.dataset = dataset
        self.env = env
        self.convert_inputs_func = convert_inputs_func
        self.model_name = model_name
        self.optimization_sense = optimization_sense
        self.verbose = verbose

        self.m = gp.Model(model_name, env)
        self.model = self.m
        self.output_vars = OrderedDict()
        self.objective = 0
        self.objective_terms = dict()
        self.all_vars = dict()
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
            if self.verbose:
                print(f"Writing {file_name}")
            self.m.write(file_name)
        return file_names if len(file_names) > 1 else file_names[0]

    def save_inverter(self, filename="inverter.pkl"):
        if self.verbose:
            print(f"Saving Inverter to {filename}")
        everything = dict()
        everything["output_keys"] = {
            key: var.varName for key, var in self.output_vars.items()
        }
        pickle.dump(everything, open(filename, "wb"))

    def load_inverter(self, filename="inverter.pkl"):
        if self.verbose:
            print(f"Loading Inverter from {filename}")
        everything = pickle.load(open(filename, "rb"))
        get_var_matrix = np.vectorize(self.m.getVarByName)
        for key, name_matrix in everything["output_keys"].items():
            self.output_vars[key] = gp.MVar.fromlist(
                get_var_matrix(name_matrix).tolist()
            )

    def load_model(self, file_name="model.mps", input_var_names=[]):
        if self.verbose:
            print(f"Loading model from {file_name}")
        self.model = gp.read(file_name)
        self.m = self.model
        self.set_input_vars(
            {name: self.m.getVarByName(name) for name in input_var_names}
        )

    def get_mvar(self, name, shape):
        X = np.empty(shape, dtype=gp.Var)
        for index in product(*[range(d) for d in shape]):
            X[index] = self.m.getVarByName(f"{name}[{','.join(str(i) for i in index)}]")
        return gp.MVar.fromlist(X.tolist())

    def solve(self, callback=None, param_file=None, **kwargs):
        if self.verbose:
            print("Beginning Solve")
        if callback is None:
            callback = self.get_default_callback()
        # Check variables are bounded or binary
        for var in self.m.getVars():
            assert var.vtype == GRB.BINARY or (
                var.LB != float("-inf") and var.UB != float("inf")
            ), f"Variable {var.VarName} is unbounded."

        for param_name, param_value in kwargs.items():
            self.m.setParam(param_name, param_value)

        if param_file:
            self.m.read(param_file)

        # TODO: Make these class properties, check if they are already set
        self.solutions = []
        self.mip_data = []
        self.m.optimize(callback)

        if self.verbose:
            print("Solve Ended with Status", self.m.Status)
            print("Solve Runtime:", self.m.Runtime)

        return {
            "Model Status": self.m.Status,
            "Node Count": self.m.NodeCount,
            "Open Node Count": self.m.OpenNodeCount,
            "Solve Runtime": self.m.Runtime,
            "MIPGap": self.m.MIPGap if hasattr(self.m, "MIPGap") else None,
        }

    def computeIIS(self, output_fname=None):
        if output_fname is None:
            output_fname = f"{self.model_name}.ilp"
        self.m.computeIIS()
        self.m.write(output_fname)
        if self.verbose:
            print(f"Wrote IIS to {output_fname}")
        return output_fname

    def get_default_callback(self):
        def solver_callback(model, where):
            if where == GRB.Callback.MIP:
                pass

            if where == GRB.Callback.MIPSOL:
                print("New Solution Found!")
                solution_inputs = {
                    name: model.cbGetSolution(var)
                    for name, var in self.input_vars.items()
                }

                last_output_key = next(reversed(self.output_vars))

                ## TODO: Get the output in a cleaner way
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
                divergence = np.abs(nn_output - output_var_value).max()
                if not np.allclose(nn_output, output_var_value):
                    warnings.warn(
                        f"Model outputs diverge: max difference is {divergence:.3e}",
                        category=RuntimeWarning,
                    )

                objective_term_values = {
                    name: model.cbGetSolution(term.var)
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

                solution = (
                    solution_inputs
                    | tracked_var_values
                    | objective_term_values
                    | {
                        "Output": nn_output,
                        "Objective Value": self.m.cbGet(GRB.Callback.MIPSOL_OBJ),
                        "Upper Bound": self.m.cbGet(GRB.Callback.MIPSOL_OBJBND),
                        "Divergence": divergence,
                    }
                )
                self.solutions.append(solution)
                print("Num Solutions:", len(self.solutions))
                return ("Solution", solution)
            elif where == GRB.Callback.MIP:
                # Access MIP information when upper bound is updated
                runtime = model.cbGet(GRB.Callback.RUNTIME)
                if self.mip_data and runtime - self.mip_data[-1]["Runtime"] < 1:
                    return

                # Save the information to a dictionary
                self.mip_data.append(
                    {
                        "Upper Bound": model.cbGet(GRB.Callback.MIP_OBJBND),
                        "Best Objective": model.cbGet(GRB.Callback.MIP_OBJBST),
                        "Node Count": model.cbGet(GRB.Callback.MIP_NODCNT),
                        "Explored Node Count": model.cbGet(GRB.Callback.MIP_NODCNT),
                        "Unexplored Node Count": model.cbGet(GRB.Callback.MIP_NODLFT),
                        "Cut Count": model.cbGet(GRB.Callback.MIP_CUTCNT),
                        "Work Units": model.cbGet(GRB.Callback.WORK),
                        "Runtime": runtime,
                    }
                )
                return ("MIP Data", self.mip_data[-1])

            else:
                return None

        return solver_callback

    def recalculate_objective(self):
        self.m.setObjective(
            sum(term.var * term.weight for term in self.objective_terms.values()),
            self.optimization_sense,
        )
        # self.objective = 0
        # for term in self.objective_terms.values():
        #     self.objective += term.var * term.weight
        # self.m.setObjective(self.objective, GRB.MAXIMIZE)

    def reset_objective(self):
        self.objective_terms = dict()
        self.m.setObjective(0, self.optimization_sense)  # self.recalculate_objective()

    def add_objective_term(self, term):
        if self.verbose:
            print(f"Adding Objective Term: {term.name}")
        self.objective_terms[term.name] = term
        self.recalculate_objective()

    def remove_objective_term(self, term_name):
        if self.verbose:
            print(f"Removing Objective Term: {term_name}")
        del self.objective_terms[term_name]
        self.recalculate_objective()

    def tune(self, **kwargs):
        if self.verbose:
            print("Beginning Tune")
        for param_name, param_value in kwargs.items():
            self.m.setParam(param_name, param_value)
            self.m.tune()
            for i in range(self.m.tuneResultCount):
                self.m.getTuneResult(i)
                self.m.write("tune" + str(i) + ".prm")

    def bounds_summary(self):
        summary = {
            "Lowest Lower Bound": min(
                var.getAttr("lb") for var in self.model.getVars()
            ),
            "Highest Upper Bound": max(
                var.getAttr("ub") for var in self.model.getVars()
            ),
            "Min ABS Bound": min(
                min(abs(var.getAttr("lb")), abs(var.getAttr("ub")))
                for var in self.model.getVars()
            ),
        }
        if self.verbose:
            print("Bounds Summary")
            for k, v in summary.items():
                print(f"{k}: {v}")
        return summary

    def encode_seq_nn(
        self, input_inits=None, add_layers=True, debug=False, max_bound=None
    ):
        if self.verbose:
            print("Encoding NN")

        assert hasattr(
            self.nn, "get_all_layer_outputs"
        ), "NN must have method get_all_layer_outputs"
        assert hasattr(
            self.nn, "layers"
        ), "NN must have attribute 'layers' containing an ordered dictionary of torch Modules"
        ## Encode the layers of a sequential
        ## For each layer, create and constrain decision variables to represent the output
        ## If in Debug Mode, we add layers one at a time and fix them to their starting values. If the model becomes infeasible, we can diagnose the problem by computing a minimal IIS
        fixing_constraints = []
        if input_inits is not None:
            if self.verbose:
                print("    Setting Warm Start")
            for var_name, init_value in input_inits.items():
                self.input_vars[var_name].Start = init_value
                if debug:
                    fixing_constraints.append(
                        self.model.addConstr(
                            self.input_vars[var_name] == init_value,
                            name=f"fix_{var_name}",
                        )
                    )
            all_layer_outputs = dict(
                self.nn.get_all_layer_outputs(**self.convert_inputs(**input_inits))
            )

        previous_layer_output = self.input_vars[
            "X"
        ]  ## TODO: Generalize to arbitrary side information
        self.model.update()
        old_numvars = self.model.NumVars
        old_numconstrs = self.model.NumConstrs

        for name, layer in self.nn.layers.items():
            self.model.update()
            if add_layers:
                print("    Encoding layer:", name)
                previous_layer_output = invert_torch_layer(
                    self.model,
                    layer,
                    name=name,
                    X=previous_layer_output,
                    A=self.input_vars[
                        "A"
                    ],  ## TODO: Generalize to arbitrary side information
                )
                self.model.update()
                if max_bound is not None:
                    for var in previous_layer_output.reshape((-1)).tolist():
                        if var.UB > max_bound:
                            if self.verbose:
                                print(
                                    f"        Lowering upper Bound of {var.varName} from {var.UB} to {max_bound}"
                                )
                            var.UB = max_bound
                        if var.LB < -max_bound:
                            if self.verbose:
                                print(
                                    f"        Raising lower Bound of {var.varName} from {var.LB} to {-max_bound}"
                                )
                            var.LB = -max_bound
                    self.model.update()
                unnamed_constraints = [
                    constr
                    for constr in self.model.getConstrs()
                    if constr.ConstrName is None
                ] + [
                    constr
                    for constr in self.model.getQConstrs()
                    if constr.QCName is None
                ]
                assert (
                    len(unnamed_constraints) == 0
                ), f"Unnamed Constraints: {[constr.ConstrName for constr in unnamed_constraints]}"
                self.output_vars[name] = previous_layer_output
            else:
                previous_layer_output = self.output_vars[name]
            if input_inits is not None:
                output = all_layer_outputs[name].detach().numpy()
                assert self.output_vars[name].shape == output.shape
                if not np.less_equal(
                    previous_layer_output.getAttr("lb"), output + 1e-8
                ).all():
                    print(
                        f'\nERROR: Layer Output Lower than Lower Bounds\nLayer: {name}\nTotal Bound Violations: {np.greater(previous_layer_output.getAttr("lb"), output).sum()} out of {output.size} elements\nLower Bounds:\n{previous_layer_output.getAttr("lb")[np.greater(previous_layer_output.getAttr("lb"), output)]}\nOutputs:\n{output[np.greater(previous_layer_output.getAttr("lb"), output)]}',
                    )
                    raise AssertionError(f"{name} Lower Bound Violation")

                # Check initializations for all variables are leq the upper bounds
                if not np.greater_equal(
                    previous_layer_output.getAttr("ub"), output - 1e-8
                ).all():
                    print(
                        f'\nERROR: Layer Output Greater than Upper Bounds\nLayer: {name}\nTotal Bound Violations: {np.greater(previous_layer_output.getAttr("lb"), output).sum()} out of {output.size} elements\nLower Bounds:\n{previous_layer_output.getAttr("lb")[np.less(previous_layer_output.getAttr("ub"), output)]}\nOutputs:\n{output[np.less(previous_layer_output.getAttr("ub"), output)]}',
                    )
                    raise AssertionError(f"{name} Upper Bound Violation")
                self.output_vars[name].Start = output
            if debug:
                fixing_constraints.append(
                    self.model.addConstr(
                        self.output_vars[name]
                        == all_layer_outputs[name].detach().numpy(),
                        name=f"fix_{name}",
                    )
                )
            self.model.update()
            numvars = self.model.NumVars
            numconstrs = self.model.NumConstrs
            print(
                f"    Added {numvars - old_numvars} variables and {numconstrs - old_numconstrs} constraints"
            )
            if debug:
                self.model.optimize()
                if not self.model.Status == GRB.OPTIMAL:
                    print("============ PROBLEM WITH LAYER:", name, "=================")
                    print(
                        "Fixed Variables:",
                        set(
                            v.varName.split("[")[0]
                            for v in self.model.getVars()[:old_numvars]
                        ),
                    )
                    print(
                        "Fixed Constraints:",
                        set(
                            c.ConstrName.split("[")[0]
                            for c in self.model.getConstrs()[:old_numconstrs]
                        ),
                    )
                    print(
                        "Relaxing Variables:",
                        set(
                            v.varName.split("[")[0]
                            for v in self.model.getVars()[old_numvars:]
                        ),
                    )
                    print(
                        "Relaxing Constraints:",
                        set(
                            c.ConstrName.split("[")[0]
                            for c in self.model.getConstrs()[old_numconstrs:]
                        ),
                    )
                    lbpen = [1.0] * (numvars - old_numvars)
                    ubpen = [1.0] * (numvars - old_numvars)
                    rhspen = [1.0] * (numconstrs - old_numconstrs)

                    print(
                        "feasRelax Result:",
                        self.model.feasRelax(
                            0,
                            False,
                            self.model.getVars()[old_numvars:],
                            lbpen,
                            ubpen,
                            self.model.getConstrs()[old_numconstrs:],
                            rhspen,
                        ),
                    )
                    self.model.optimize()
                    print("\nSlack values:")
                    slacks = self.model.getVars()[numvars:]
                    for sv in slacks:
                        if sv.X > 1e-9:
                            print("%s = %g" % (sv.VarName, sv.X))
                    raise ValueError("Infeasible Model")

            old_numvars = numvars
            old_numconstrs = numconstrs

        if debug:
            self.model.remove(fixing_constraints)
