from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient, qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy, qMaxValueEntropy
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import (
    PosteriorMean,
    FixedFeatureAcquisitionFunction,
    ExpectedImprovement,
)
from botorch.acquisition.objective import MCAcquisitionObjective

from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim import optimize_acqf, optimize_acqf_mixed
import torch

from .config import SMOKE_TEST

torch.set_printoptions(precision=3, sci_mode=False)

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

# Need to make this a class for pickle-ability
class ProjectionOperator:

    def __init__(self, target_fidelities=None) -> None:
        self.target_fidelities = target_fidelities
    
    def __call__(self, X):
        return project_to_target_fidelity(X, self.target_fidelities)


class FixedCostObjective(MCAcquisitionObjective):
    def __init__(self, fixed_cost) -> None:
        super().__init__()
        self.fixed_cost = fixed_cost

    def forward(self, samples: torch.Tensor, X=None):
        return self.fixed_cost + samples.squeeze(-1)


def optimize_acqf_and_get_observation(
    acqf, bounds, objective_function, q=1, verbose=False
):
    """Optimizes acquisition_function and returns a new candidate, observation, and cost."""

    fidelity_lower = bounds[0, -1].item()
    fidelity_upper = bounds[1, -1].item()

    if fidelity_upper != 1.0:
        fixed_features_list = [{-1:i} for i in range(int(fidelity_lower), int(fidelity_upper) + 1)]
    else:
        fixed_features_list = None
    
    if isinstance(acqf, qMultiFidelityKnowledgeGradient):
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            fixed_features=fixed_features_list[-1]
        )
    else:
        X_init = None

    if fidelity_upper != 1.0:
        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 500},
            fixed_features_list=fixed_features_list
        )
    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
        )
    # observe new values
    new_x = candidates.detach()
    new_obj, new_cost, full_results = objective_function(new_x)
    new_obj = new_obj.unsqueeze(-1)
    new_cost = new_cost.unsqueeze(-1)

    if verbose:
        print(f"candidates:\n{new_x}\n")
        print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, new_cost, full_results


def multi_fidelity_kg(model, cost_model, bounds):
    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=cost_model, cost_objective=FixedCostObjective(fixed_cost=0.0)
    )

    target_fidelity = bounds[1, -1].cpu().item()
    projection_operator = ProjectionOperator(target_fidelities={-1:target_fidelity})

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=bounds.shape[-1],
        columns=[-1],
        values=[target_fidelity],
    )

    # We need to calculate the current best value since we want to optimize the *improvement over current best* per cost
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )

    
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=projection_operator,
    )


def mutli_fidelity_entropy_search(model, cost_model, bounds):
    """Generate multi-fidelity ES acqisition function."""

    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=cost_model, cost_objective=FixedCostObjective(fixed_cost=0.0)
    )

    candidate_set = torch.rand(
        1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype
    )
    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set

    target_fidelity = bounds[1, -1].cpu().item()
    projection_operator = ProjectionOperator(target_fidelities={-1:target_fidelity})
    return qMultiFidelityMaxValueEntropy(
        model=model,
        cost_aware_utility=cost_aware_utility,
        project=projection_operator,
        num_fantasies=128 if not SMOKE_TEST else 2,
        candidate_set=candidate_set,
    )


def expected_improvement(model, cost_model, bounds, best_f):
    return ExpectedImprovement(model=model, best_f=best_f)


expected_improvement.pass_current_best = True
