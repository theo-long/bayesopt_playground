from .optimization_loops import optimization_loop, get_recommendation
from .benchmarks import generate_optimization_task

import pandas as pd


def run_experiment(
    experiment_name,
    benchmarks,
    model_factory,
    cost_model_factory,
    acquisition_factory,
    multi_fidelity,
    num_runs=20,
    num_iters=50,
):
    """Run a single """

    all_results = []
    for benchmark in benchmarks:
        print(f"Starting {benchmark.__name__}")
        objective_function, bounds = generate_optimization_task(benchmark)
        log_transform_indices = getattr(benchmark, "log_transform_indices", None)
        if log_transform_indices:
            old_mf = model_factory
            old_cmf = cost_model_factory
            model_factory = lambda *args: old_mf(
                *args, log_transform_indices=log_transform_indices
            )
            cost_model_factory = lambda *args: old_cmf(
                *args, log_transform_indices=log_transform_indices
            )

        pass_current_best = getattr(acquisition_factory, "pass_current_best", False)
        full_fidelity = not multi_fidelity
        if multi_fidelity:
            initial_samples = 10
            fidelity_samples = [0.1, 0.2, 0.4]
        else:
            initial_samples = 30
            fidelity_samples = None

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}", end="\r")

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
            )
            final_rec, objective_value, results = get_recommendation(
                model, objective_function, bounds, full_fidelity=full_fidelity
            )
            full_results += results
            evaluations_df = pd.concat(
                [
                    pd.DataFrame(train_x.detach().numpy()),
                    pd.DataFrame(final_rec.detach().numpy()),
                ]
            ).reset_index(drop=True)
            results_df = pd.DataFrame(full_results)
            results_df["run"] = run
            results_df["benchmark"] = benchmark.__name__
            results_df = results_df.reset_index()
            # results_df = pd.concat([results_df, evaluations_df], axis=1)
            all_results.append(results_df)

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(f"./data/{experiment_name}.csv")

    return evaluations_df, results_df
