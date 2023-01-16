from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize, Log10, ChainedInputTransform
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel


def multi_fidelity_gp(train_x, train_obj, bounds, log_transform_indices=None):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    if not log_transform_indices:
        input_transform = Normalize(d=train_x.shape[-1], bounds=bounds)
    else:
        input_transform = ChainedInputTransform(
            tf1=Log10(log_transform_indices), 
            tf2=Normalize(d=train_x.shape[-1], bounds=Log10(log_transform_indices)(bounds)))

    model = SingleTaskMultiFidelityGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=1),
        input_transform=input_transform,
        data_fidelity=-1,
        linear_truncated=True,
        nu=5/2
    )   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def simple_gp(train_x, train_obj, bounds, log_transform_indices=None, output_dim=1):
        # define a surrogate model suited for a "training data"-like fidelity parameter
    if not log_transform_indices:
        input_transform = Normalize(d=train_x.shape[-1], bounds=bounds)
    else:
        input_transform = ChainedInputTransform(
            tf1=Log10(log_transform_indices), 
            tf2=Normalize(d=train_x.shape[-1], bounds=Log10(log_transform_indices)(bounds)))

    model = SingleTaskGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=output_dim),
        input_transform=input_transform,
        covar_module=MaternKernel(nu=5/2)
    )   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model