from .optimization_loops import optimization_loop, get_recommendation
from .benchmarks import generate_optimization_task

from functools import partial

from botorch.models.cost import AffineFidelityCostModel

import pandas as pd
import os
import multiprocessing


def run_experiment(
    experiment_name,
    benchmarks,
    model_factory,
    cost_model_factory,
    acquisition_factory,
    multi_fidelity,
    num_runs=20,
    num_iters=50,
    initial_samples=5,
    parallelized=True,
    optimizer=None,
):
    """Run a single experiment on multiple benchmarks."""
    if not os.path.exists("./data"):
        os.mkdir("./data")

    all_results = []
    for benchmark in benchmarks:
        print(f"Starting {benchmark.__name__}")
        objective_function, bounds = generate_optimization_task(benchmark)

        log_transform_indices = getattr(benchmark, "log_transform_indices", None)
        if log_transform_indices:
            model_factory = partial(
                model_factory, log_transform_indices=log_transform_indices
            )

            if not isinstance(cost_model_factory, AffineFidelityCostModel):
                cost_model_factory = partial(
                    cost_model_factory, log_transform_indices=log_transform_indices
                )

        pass_current_best = getattr(acquisition_factory, "pass_current_best", False)
        full_fidelity = not multi_fidelity

        if multi_fidelity:
            max_fidelity = bounds[1, -1]
            fidelity_samples = [
                0.1 * max_fidelity,
                0.2 * max_fidelity,
                0.4 * max_fidelity,
            ]
            if max_fidelity > 1.0:
                fidelity_samples = [int(s) for s in fidelity_samples]
        else:
            fidelity_samples = None

        def args_iterable():
            for run in range(num_runs):
                yield (
                    run,
                    model_factory,
                    cost_model_factory,
                    acquisition_factory,
                    objective_function,
                    bounds,
                    num_iters,
                    initial_samples,
                    fidelity_samples,
                    log_transform_indices,
                    full_fidelity,
                    pass_current_best,
                    benchmark,
                    optimizer,
                )

        if parallelized:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                results = pool.starmap(single_run, args_iterable())
        else:
            results = []
            iterable = args_iterable()
            for args in iterable:
                results.append(single_run(*args))

        all_results += results

        # Save after every benchmark in case of partial run
        all_results_df = pd.concat(all_results)
        all_results_df.to_csv(f"./data/{experiment_name}.csv")

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(f"./data/{experiment_name}.csv")


def single_run(
    run,
    model_factory,
    cost_model_factory,
    acquisition_factory,
    objective_function,
    bounds,
    num_iters,
    initial_samples,
    fidelity_samples,
    log_transform_indices,
    full_fidelity,
    pass_current_best,
    benchmark,
    optimizer,
):
    print(f"Run {run+1}", end="\r")

    model, train_x, train_obj, full_results = optimization_loop(
        model_factory,
        cost_model_factory,
        acquisition_factory,
        objective_function,
        bounds,
        n_iter=num_iters,
        initial_samples=initial_samples,
        fidelity_samples=fidelity_samples,
        log_transform_indices=log_transform_indices,
        full_fidelity=full_fidelity,
        pass_current_best=pass_current_best,
        optimizer=optimizer,
    )
    final_rec, objective_value, results = get_recommendation(
        model, objective_function, bounds, full_fidelity=full_fidelity
    )
    full_results += results
    evaluations_df = pd.concat(
        [
            pd.DataFrame(train_x.detach().cpu().numpy()),
            pd.DataFrame(final_rec.detach().cpu().numpy()),
        ]
    ).reset_index(drop=True)
    results_df = pd.DataFrame(full_results)
    results_df["run"] = run
    results_df["benchmark"] = benchmark.__name__
    results_df = results_df.reset_index()
    results_df = pd.concat([results_df, evaluations_df], axis=1)
    return results_df
