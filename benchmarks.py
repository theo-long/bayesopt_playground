import torch
import scipy

from hpobench.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
from hpobench.benchmarks.ml import NNBenchmark
from hpobench.abstract_benchmark import AbstractBenchmark

from botorch.test_functions import SyntheticTestFunction, AugmentedBranin, AugmentedHartmann, Branin, Hartmann
from botorch.models.cost import AffineFidelityCostModel

from .config import tkwargs

OPEN_ML_TASK_IDS = {
    "credit-g":31,
    "iris":59,
    "fashion-mnist":146825,
    "boston":3736,
    "vehicle_registration":53,
    "mnist":3573
}

# This has been difficult to set up imports for
def svm_benchmark():
    return SurrogateSVMBenchmark(rng=1), "dataset_fraction", None

def nn_benchmark(task):
    return NNBenchmark(task_id=OPEN_ML_TASK_IDS[task], rng=1), "subsample", {"batch_size":64, "depth":3, "width":128}

nn_mnist_benchmark = lambda : nn_benchmark("mnist")
nn_mnist_benchmark.log_transform_indices = [0, 1]

nn_fashion_mnist_benchmark = lambda : nn_benchmark("fashion-mnist")
nn_fashion_mnist_benchmark.log_transform_indices = [0, 1]

def augmented_hartmann_benchmark():
    obj_function = AugmentedHartmann(negate=True)
    return obj_function, torch.tensor([[0.0] * obj_function.dim, [1.0] * obj_function.dim], **tkwargs)
augmented_hartmann_benchmark.synthetic = True

def augmented_branin_benchmark():
    return AugmentedBranin(negate=True),  torch.tensor([[-5., 10.], [0., 15.], [0., 1.]], **tkwargs).T
augmented_branin_benchmark.synthetic = True

def hartmann_benchmark():
    obj_function = Hartmann(negate=True)
    return obj_function, torch.tensor([[0.0] * obj_function.dim, [1.0] * obj_function.dim], **tkwargs)
hartmann_benchmark.synthetic = True

def branin_benchmark():
    return Branin(negate=True),  torch.tensor([[-5., 10.], [0., 15.]], **tkwargs).T
branin_benchmark.synthetic = True

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

def generate_objective_function(benchmark: AbstractBenchmark, fidelity_param, fixed_params=None, fidelity_kwargs=None):
    if not fixed_params:
        fixed_params = {}
    param_names = list(benchmark.configuration_space)
    for p in fixed_params.keys():
        param_names.remove(p)


    config_dict = fixed_params
    def objective_wrapper(x, project_to_max_fidelity=False):
        values = []
        costs = []
        results = []
        for row in x:
            if len(row.shape) == 2:
                row = row.squeeze()
            for k, v in zip(param_names, row):
                config_dict[k] = v.item()

            fidelity_dict = {fidelity_param:1.0} if project_to_max_fidelity else {fidelity_param:row[-1].item()}
            if fidelity_kwargs:
                fidelity_dict.update(fidelity_kwargs)
            res = benchmark.objective_function(config_dict, fidelity_dict)
            results.append(res)
            # multiply by -1 since botorch maximizes by default
            values.append(-1 * res["function_value"])
            costs.append(res["cost"])
        return torch.tensor(values, **tkwargs), torch.tensor(costs, **tkwargs), results
    return objective_wrapper

def generate_optimization_task(benchmark):
    '''Generate bounds and objective function.'''

    if getattr(benchmark, "synthetic", False):
        # Ensure same call signature as above
        benchmark, bounds = benchmark()
        synthetic_cost_model = AffineFidelityCostModel(fidelity_weights={-1: 1.0}, fixed_cost=5.0)
        def objective_function(*args, **kwargs):
            # synthetic functions have no cost, so need to artificially create one
            cost = synthetic_cost_model(*args).squeeze(1)
            value = benchmark(*args)
            return value, cost, {"function_value":value, "cost":cost}
        objective_function.synthetic = True
        return objective_function, bounds

    
    benchmark, fidelity_param, fixed_params = benchmark()
    objective_function = generate_objective_function(benchmark, fidelity_param, fixed_params)
    bounds = generate_benchmark_bounds(benchmark, fidelity_param, fixed_params)
    objective_function.synthetic = False
    return objective_function, bounds




if __name__ == "__main__":
    from IPython import embed
    embed()

