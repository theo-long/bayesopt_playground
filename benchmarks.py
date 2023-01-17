import torch
import scipy

from hpobench.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnTimeBenchmark,
    ParamNetHiggsOnTimeBenchmark,
    ParamNetLetterOnTimeBenchmark,
    ParamNetMnistOnTimeBenchmark,
    ParamNetOptdigitsOnTimeBenchmark,
)
from hpobench.benchmarks.ml import NNBenchmark
from hpobench.abstract_benchmark import AbstractBenchmark

from botorch.test_functions import (
    SyntheticTestFunction,
    AugmentedBranin,
    AugmentedHartmann,
    Branin,
    Hartmann,
)
from botorch.models.cost import AffineFidelityCostModel

from .config import tkwargs

OPEN_ML_TASK_IDS = {
    "credit-g": 31,
    "iris": 59,
    "fashion-mnist": 146825,
    "boston": 3736,
    "vehicle_registration": 53,
    "mnist": 3573,
}

PARAMNET_BENCHMARKS = {
    "mnist":ParamNetMnistOnTimeBenchmark,
    "adult":ParamNetAdultOnTimeBenchmark,
    "letter":ParamNetLetterOnTimeBenchmark,
    "higgs":ParamNetHiggsOnTimeBenchmark,
    "opt_digits":ParamNetOptdigitsOnTimeBenchmark,
}

# This has been difficult to set up imports for
def svm_benchmark():
    return SurrogateSVMBenchmark(rng=1), "dataset_fraction", None, None

def paramnet_benchmark(task):
    benchmark = PARAMNET_BENCHMARKS[task]
    return benchmark(rng=1), "budget", {'batch_size_log2': 7, 'average_units_per_layer_log2': 6, "num_layers":3}, None

def paramnet_mnist():
    return paramnet_benchmark("mnist")

def paramnet_adult():
    return paramnet_benchmark("adult")

def paramnet_letter():
    return paramnet_benchmark("letter")

def paramnet_opt_digits():
    return paramnet_benchmark("opt_digits")

def nn_benchmark(task):
    return (
        NNBenchmark(task_id=OPEN_ML_TASK_IDS[task], rng=1),
        "subsample",
        {"batch_size": 64, "depth": 3, "width": 128},
        {"iter": 25},
    )


def nn_mnist_benchmark():
    return nn_benchmark("mnist")


nn_mnist_benchmark.log_transform_indices = [0, 1]


def nn_fashion_mnist_benchmark():
    return nn_benchmark("fashion-mnist")


nn_fashion_mnist_benchmark.log_transform_indices = [0, 1]


def augmented_hartmann_benchmark():
    obj_function = AugmentedHartmann(negate=True)
    return obj_function, torch.tensor(
        [[0.0] * obj_function.dim, [1.0] * obj_function.dim], **tkwargs
    )


augmented_hartmann_benchmark.synthetic = True


def augmented_branin_benchmark():
    return (
        AugmentedBranin(negate=True),
        torch.tensor([[-5.0, 10.0], [0.0, 15.0], [0.0, 1.0]], **tkwargs).T,
    )


augmented_branin_benchmark.synthetic = True


def hartmann_benchmark():
    obj_function = Hartmann(negate=True)
    return obj_function, torch.tensor(
        [[0.0] * obj_function.dim, [1.0] * obj_function.dim], **tkwargs
    )


hartmann_benchmark.synthetic = True


def branin_benchmark():
    return Branin(negate=True), torch.tensor([[-5.0, 10.0], [0.0, 15.0]], **tkwargs).T


branin_benchmark.synthetic = True


def generate_benchmark_bounds(
    benchmark: AbstractBenchmark, fidelity_param, fixed_params=None
):
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


class ObjectiveFunction:
    def __init__(
        self,
        objective_function,
        param_names,
        config_dict,
        fidelity_param,
        fidelity_kwargs,
        max_fidelity,
    ) -> None:
        self.param_names = param_names
        self.config_dict = config_dict
        self.fidelity_param = fidelity_param
        self.fidelity_kwargs = fidelity_kwargs
        self.objective_function = objective_function
        if max_fidelity > 1.0:
          self.max_fidelity = int(max_fidelity)
        else:
          self.max_fidelity = max_fidelity

    def __call__(self, x, project_to_max_fidelity=False):
        values = []
        costs = []
        results = []
        for row in x:
            if len(row.shape) == 2:
                row = row.squeeze()
            for k, v in zip(self.param_names, row):
                self.config_dict[k] = v.item()
    
            fidelity_value = row[-1].item()
            if self.max_fidelity > 1.0:
                fidelity_value = int(fidelity_value)
            fidelity_dict = (
                {self.fidelity_param: self.max_fidelity}
                if project_to_max_fidelity
                else {self.fidelity_param: fidelity_value}
            )
            if self.fidelity_kwargs:
                fidelity_dict.update(self.fidelity_kwargs)
            res = self.objective_function(self.config_dict, fidelity_dict)
            results.append(res)
            # multiply by -1 since botorch maximizes by default
            values.append(-1 * res["function_value"])
            costs.append(res["cost"])
        return torch.tensor(values, **tkwargs), torch.tensor(costs, **tkwargs), results


class SyntheticObjectiveFunction:
    def __init__(self, benchmark) -> None:
        self.benchmark = benchmark
        self.synthetic = True
        self.synthetic_cost_model = AffineFidelityCostModel(
            fidelity_weights={-1: 1.0}, fixed_cost=5.0
        )

    def __call__(self, *args, **kwargs):
        # synthetic functions have no cost, so need to artificially create one
        cost = self.synthetic_cost_model(*args).squeeze(1)
        value = self.benchmark(*args)
        return (
            value,
            cost,
            [
                {"function_value": v, "cost": c}
                for v, c in zip(
                    value.detach().flatten().cpu().numpy(),
                    cost.detach().flatten().cpu().numpy(),
                )
            ],
        )


def generate_objective_function(
    benchmark: AbstractBenchmark,
    fidelity_param,
    fixed_params=None,
    fidelity_kwargs=None,
) -> ObjectiveFunction:
    if not fixed_params:
        fixed_params = {}
    param_names = list(benchmark.configuration_space)
    for p in fixed_params.keys():
        param_names.remove(p)

    config_dict = fixed_params
    max_fidelity = benchmark.fidelity_space[fidelity_param].upper

    objective_wrapper = ObjectiveFunction(
        benchmark.objective_function,
        param_names,
        config_dict,
        fidelity_param,
        fidelity_kwargs,
        max_fidelity,
    )

    return objective_wrapper


def generate_optimization_task(benchmark):
    """Generate bounds and objective function."""

    if getattr(benchmark, "synthetic", False):
        # Ensure same call signature as above
        benchmark, bounds = benchmark()
        objective_function = SyntheticObjectiveFunction(benchmark)
        return objective_function, bounds

    benchmark, fidelity_param, fixed_params, fidelity_kwargs = benchmark()
    objective_function = generate_objective_function(
        benchmark, fidelity_param, fixed_params, fidelity_kwargs
    )
    bounds = generate_benchmark_bounds(benchmark, fidelity_param, fixed_params)
    objective_function.synthetic = False
    return objective_function, bounds


if __name__ == "__main__":
    from IPython import embed

    embed()
