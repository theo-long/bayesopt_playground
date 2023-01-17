
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize, Log10, ChainedInputTransform
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from gpytorch.kernels import MaternKernel, RBFKernel, ProductKernel, ScaleKernel, LinearKernel
from botorch.models.kernels import ExponentialDecayKernel, LinearTruncatedFidelityKernel, DownsamplingKernel
from gpytorch.priors import GammaPrior

import torch
from typing import Optional, Dict, Tuple, List, Any

from botorch.exceptions import UnsupportedError


def multi_fidelity_gp(train_x, train_obj, bounds, kernel=MaternKernel, linear_truncated=False, log_transform_indices=None):
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
        linear_truncated=linear_truncated,
        nu=5/2,
        kernel=kernel
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

def _setup_custom_multifidelity_covar_module(
    dim: int,
    aug_batch_shape: torch.Size,
    iteration_fidelity: Optional[int],
    data_fidelity: Optional[int],
    linear_truncated: bool,
    nu: float,
    kernel = RBFKernel,
    data_fidelity_kernel = None,
) -> Tuple[ScaleKernel, Dict]:
    """Helper function to get the covariance module and associated subset_batch_dict
    for the multifidelity setting.

    Args:
        dim: The dimensionality of the training data.
        aug_batch_shape: The output-augmented batch shape as defined in
            `BatchedMultiOutputGPyTorchModel`.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelity: The column index for the downsampling fidelity parameter
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.

    Returns:
        The covariance module and subset_batch_dict.
    """

    if iteration_fidelity is not None and iteration_fidelity < 0:
        iteration_fidelity = dim + iteration_fidelity
    if data_fidelity is not None and data_fidelity < 0:
        data_fidelity = dim + data_fidelity

    if linear_truncated:
        fidelity_dims = [
            i for i in (iteration_fidelity, data_fidelity) if i is not None
        ]
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=fidelity_dims,
            dimension=dim,
            nu=nu,
            batch_shape=aug_batch_shape,
            power_prior=GammaPrior(3.0, 3.0),
        )
    else:
        active_dimsX = [
            i for i in range(dim) if i not in {iteration_fidelity, data_fidelity}
        ]
        kernel = kernel(
            ard_num_dims=len(active_dimsX),
            batch_shape=aug_batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            active_dims=active_dimsX,
        )
        additional_kernels = []
        if iteration_fidelity is not None:
            exp_kernel = ExponentialDecayKernel(
                batch_shape=aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
                offset_prior=GammaPrior(3.0, 6.0),
                power_prior=GammaPrior(3.0, 6.0),
                active_dims=[iteration_fidelity],
            )
            additional_kernels.append(exp_kernel)
        if data_fidelity is not None:
            if data_fidelity_kernel:
                ds_kernel = data_fidelity_kernel(
                    batch_shape=aug_batch_shape,
                    offset_prior=GammaPrior(3.0, 6.0),
                    power_prior=GammaPrior(3.0, 6.0),
                    active_dims=[data_fidelity],
                )
            else:
                ds_kernel = DownsamplingKernel(
                    batch_shape=aug_batch_shape,
                    offset_prior=GammaPrior(3.0, 6.0),
                    power_prior=GammaPrior(3.0, 6.0),
                    active_dims=[data_fidelity],
                )
            additional_kernels.append(ds_kernel)
        kernel = ProductKernel(kernel, *additional_kernels)

    covar_module = ScaleKernel(
        kernel, batch_shape=aug_batch_shape, outputscale_prior=GammaPrior(2.0, 0.15)
    )

    if linear_truncated:
        subset_batch_dict = {
            "covar_module.base_kernel.raw_power": -2,
            "covar_module.base_kernel.covar_module_unbiased.raw_lengthscale": -3,
            "covar_module.base_kernel.covar_module_biased.raw_lengthscale": -3,
        }
    else:
        subset_batch_dict = {
            "covar_module.base_kernel.kernels.0.raw_lengthscale": -3,
            "covar_module.base_kernel.kernels.1.raw_power": -2,
            "covar_module.base_kernel.kernels.1.raw_offset": -2,
        }
        if iteration_fidelity is not None:
            subset_batch_dict = {
                "covar_module.base_kernel.kernels.1.raw_lengthscale": -3,
                **subset_batch_dict,
            }
            if data_fidelity is not None:
                subset_batch_dict = {
                    "covar_module.base_kernel.kernels.2.raw_power": -2,
                    "covar_module.base_kernel.kernels.2.raw_offset": -2,
                    **subset_batch_dict,
                }

    return covar_module, subset_batch_dict


class SingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the data fidelity
    parameter (if present) and an ExponentialDecayKernel for the iteration
    fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelity=3)
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        iteration_fidelity: Optional[int] = None,
        data_fidelity: Optional[int] = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        likelihood = None,
        outcome_transform = None,
        input_transform = None,
        kernel=RBFKernel,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x (d + s)` tensor of training features,
                where `s` is the dimension of the fidelity parameters (either one
                or two).
            train_Y: A `batch_shape x n x m` tensor of training observations.
            iteration_fidelity: The column index for the training iteration fidelity
                parameter (optional).
            data_fidelity: The column index for the downsampling fidelity parameter
                (optional).
            linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
                of the default kernel.
            nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
                5/2. Only used when `linear_truncated=True`.
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                    training data during instantiation and to the posterior during
                    inference (that is, the `Posterior` obtained by calling
                    `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                    forward pass.
        """
        self._init_args = {
            "iteration_fidelity": iteration_fidelity,
            "data_fidelity": data_fidelity,
            "linear_truncated": linear_truncated,
            "nu": nu,
            "outcome_transform": outcome_transform,
        }
        if iteration_fidelity is None and data_fidelity is None:
            raise UnsupportedError(
                "SingleTaskMultiFidelityGP requires at least one fidelity parameter."
            )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self._set_dimensions(train_X=transformed_X, train_Y=train_Y)
        covar_module, subset_batch_dict = _setup_custom_multifidelity_covar_module(
            dim=transformed_X.size(-1),
            aug_batch_shape=self._aug_batch_shape,
            iteration_fidelity=iteration_fidelity,
            data_fidelity=data_fidelity,
            linear_truncated=linear_truncated,
            nu=nu,
            kernel=kernel,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
            **subset_batch_dict,
        }
        self.to(train_X)

    @classmethod
    def construct_inputs(
        cls,
        training_data,
        fidelity_features: List[int],
        **kwargs,
    ) -> Dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `SupervisedDataset`.

        Args:
            training_data: Dictionary of `SupervisedDataset`.
            fidelity_features: Index of fidelity parameter as input columns.
        """
        if len(fidelity_features) != 1:
            raise UnsupportedError("Multiple fidelity features not supported.")

        inputs = super().construct_inputs(training_data=training_data, **kwargs)
        inputs["data_fidelity"] = fidelity_features[0]
        return inputs