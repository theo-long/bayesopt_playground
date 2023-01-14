from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean, FixedFeatureAcquisitionFunction, qExpectedImprovement
from botorch.acquisition.objective import MCAcquisitionObjective

from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim import optimize_acqf
import torch

from .config import SMOKE_TEST

torch.set_printoptions(precision=3, sci_mode=False)

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

class FixedCostObjective(MCAcquisitionObjective):
    def __init__(self, fixed_cost) -> None:
        super().__init__()
        self.fixed_cost = fixed_cost

    def forward(self, samples:torch.Tensor, X=None):
        return self.fixed_cost + samples.squeeze(-1)


def optimize_acqf_and_get_observation(acqf, bounds, objective_function, q=1, verbose=False):
    """Optimizes acquisition_function and returns a new candidate, observation, and cost."""
    
    if isinstance(acqf, qMultiFidelityKnowledgeGradient):
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function = acqf,
            bounds=bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
    else:
        X_init = None

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200}
    )
    # observe new values
    new_x = candidates.detach()
    new_obj, new_cost = objective_function(new_x)
    new_obj = new_obj.unsqueeze(-1)
    new_cost = new_cost.unsqueeze(-1)

    if verbose:
        print(f"candidates:\n{new_x}\n")
        print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, new_cost

def multi_fidelity_kg(model, cost_model, bounds):
    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=cost_model,
        cost_objective=FixedCostObjective(fixed_cost=0.0)
        )
    

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=bounds.shape(-1),
        columns=[-1],
        values=[1.0],
    )

    # We need to calculate the current best value since we want to optimize the *improvement over current best* per cost
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
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
        project=project_to_target_fidelity
    )

def mutli_fidelity_entropy_search(model, cost_model, bounds):
    """Generate multi-fidelity ES acqisition function."""

    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=cost_model,
        cost_objective=FixedCostObjective(fixed_cost=0.0)
        )

    candidate_set = torch.rand(1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
            
    return qMultiFidelityMaxValueEntropy(
        model=model,
        cost_aware_utility=cost_aware_utility,
        project=project_to_target_fidelity,
        num_fantasies=128 if not SMOKE_TEST else 2,
        candidate_set=candidate_set,
    )

def expected_improvement(model, cost_model, bounds, best_f):
    return FixedFeatureAcquisitionFunction(
        acq_function=qExpectedImprovement(model=model, best_f=best_f),
        d=bounds.shape[-1],
        columns=[-1],
        values=[1],
    )