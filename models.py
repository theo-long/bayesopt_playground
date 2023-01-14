from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def multi_fidelity_gp(train_x, train_obj, bounds):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    model = SingleTaskMultiFidelityGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        data_fidelity=-1,
        linear_truncated=True,
        nu=5/2
    )   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model