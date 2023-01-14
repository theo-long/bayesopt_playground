import torch
import scipy

from hpobench.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
from hpobench.benchmarks.ml import NNBenchmark
from hpobench.abstract_benchmark import AbstractBenchmark

from .config import tkwargs

OPEN_ML_TASK_IDS = {
    "credit-g":31,
    "iris":59,
    "fashion-mnist":146825,
    "boston":3736,
    "vehicle_registration":53,
    "mnist":167216
}

# This has been difficult to set up imports for
def svm_benchmark():
    return SurrogateSVMBenchmark(rng=1), "dataset_fraction", None

def nn_benchmark(task):
    return NNBenchmark(task_id=OPEN_ML_TASK_IDS["task"], rng=1), "subsample", {"batch_size":64, "depth":3, "width":128}

def generate_benchmark_bounds(benchmark: AbstractBenchmark, fidelity_param, fixed_params=None):
    if not fixed_params:
        fixed_params = {}
    bounds = []
    for param_name in benchmark.get_configuration_space():
        if param_name not in fixed_params.keys():
            param = benchmark.get_configuration_space()[param_name]
            bounds.append([param.lower, param.upper])

    fidelity_param = benchmark.get_fidelity_space()[fidelity_param]
    bounds.append([fidelity_param.lower, fidelity_param.upper])
    return torch.tensor(bounds, **tkwargs).T

def generate_objective_function(benchmark: AbstractBenchmark, fidelity_param, fixed_params=None):
    if not fixed_params:
        fixed_params = {}
    param_names = list(benchmark.configuration_space)
    for p in fixed_params.keys():
        param_names.remove(p)

    config_dict = fixed_params
    def objective_wrapper(x, full_result=False):
        accs = []
        costs = []
        results = []
        for row in x:
            if len(row.shape) == 2:
                row = row.squeeze()
            for k, v in zip(param_names, row):
                config_dict[k] = v.item()

            res = benchmark.objective_function(config_dict, {fidelity_param:row[-1].item()})
            results.append(res)
            accs.append(res["function_value"])
            costs.append(res["cost"])
        if full_result:
            return results
        else:
            return torch.tensor(accs, **tkwargs), torch.tensor(costs, **tkwargs)
    return objective_wrapper

if __name__ == "__main__":
    from IPython import embed
    embed()

