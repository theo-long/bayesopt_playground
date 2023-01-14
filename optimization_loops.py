from botorch.utils import sampling
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import FixedFeatureAcquisitionFunction, PosteriorMean
from botorch.exceptions import ModelFittingError

import gpytorch.settings as gpt_settings

import torch
from tqdm import trange

from .acquistion_functions import optimize_acqf_and_get_observation, NUM_RESTARTS, RAW_SAMPLES


def generate_initial_data(objective_function, bounds, fidelity_samples=None, n=16, q=1):
    """generate inital training data before optimization starts."""
    if fidelity_samples is not None:
        bounds = bounds[:, :-1]

    train_x = sampling.draw_sobol_samples(bounds, n=n, q=q)
    if fidelity_samples:
        train_x = train_x.repeat(len(fidelity_samples), 1, 1)
        ext_samples = torch.tensor(fidelity_samples)[:, None, None].repeat(n, 1, 1)
        train_x = torch.cat([train_x, ext_samples], dim=-1)

    train_obj, train_cost = objective_function(train_x)
    return train_x, train_obj, train_cost


def optimization_loop(
    model_factory,
    cost_model_factory,
    acquistion_factory,
    objective_function,
    bounds,
    initial_samples=20,
    fidelity_samples=None,
    n_iter=10,
    q=1,
):
    # Initialize cost
    cumulative_cost = 0.0

    # Initial random sampling to seed the model
    train_x, train_obj, train_cost = generate_initial_data(
        objective_function,
        bounds,
        fidelity_samples=fidelity_samples,
        n=initial_samples,
        q=q,
    )
    train_obj = train_obj.unsqueeze(-1)
    train_cost = train_cost.unsqueeze(-1)
    train_x = train_x.squeeze(1)

    cumulative_cost += train_cost.sum()

    with gpt_settings.cholesky_max_tries(6):
        # Initialize objective model
        mll, model = model_factory(train_x, train_obj, bounds)
        fit_gpytorch_mll(mll)

        # Initialize cost model
        cost_mll, cost_model = cost_model_factory(train_x, train_cost, bounds)
        fit_gpytorch_mll(cost_mll)

        for _ in trange(n_iter):
            # Fetch new evaluation points
            acquisition_function = acquistion_factory(model, cost_model, bounds)
            new_x, new_obj, new_cost = optimize_acqf_and_get_observation(
                acquisition_function, bounds, objective_function, q=q
            )

            # Update dataset
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            train_cost = torch.cat([train_cost, new_cost])
            cumulative_cost += new_cost.sum()

            # Fit models with new data
            try:
                mll, model = model_factory(train_x, train_obj, bounds)
                fit_gpytorch_mll(mll)
                cost_mll, cost_model = cost_model_factory(train_x, train_cost, bounds)
                fit_gpytorch_mll(cost_mll)
            except ModelFittingError:
                print("ERROR FITTING MODEL")
                return model, train_x, train_obj


    return model, train_x, train_obj

def get_recommendation(model, objective_function, bounds):
    '''Generate a recommended hyperparameter setting from a trained model.'''

    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=bounds.shape[-1],
        columns=[-1],
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    final_rec = rec_acqf._construct_X_full(final_rec)
    
    objective_value, objective_cost = objective_function(final_rec)
    print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec


if __name__ == "__main__":
    from IPython import embed

    embed()
