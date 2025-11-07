import math

import torch

from hy2dl.utils.distributions import Distribution


def nse_basin_averaged(y_sim: torch.tensor, y_obs: torch.tensor, per_basin_target_std: torch.tensor) -> torch.Tensor:
    """Basin-averaged Nash--Sutcliffe Efficiency.

    Loss function where the squared errors are weighed by the standard deviation of each basin. A description of this
    function is available at [#]_.

    Parameters
    ----------
    y_sim : torch.Tensor
        Simulated discharges.
    y_obs : torch.Tensor
        Observed discharges.
    per_basin_target_std : torch.Tensor
        Standard deviation of the discharge (during training period) for the respective basins.

    Returns
    -------
    loss: torch.Tensor
        Value of the basin-averaged NSE

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
       universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
       *Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019

    """
    # calculate mask to avoid nan in observation to affect the loss
    mask = ~torch.isnan(torch.flatten(y_obs))
    y_sim_masked = torch.flatten(y_sim)[mask]
    y_obs_masked = torch.flatten(y_obs)[mask]
    basin_std_masked = torch.flatten(per_basin_target_std)[mask]

    squared_error = (y_sim_masked - y_obs_masked) ** 2
    weights = 1 / (basin_std_masked + 0.1) ** 2  # The 0.1 is a small constant for numerical stability
    loss = weights * squared_error

    return torch.mean(loss)


def weighted_rmse(y_sim: torch.tensor, y_obs: torch.tensor) -> torch.Tensor:
    """Weighted root mean squared error.

    Weighted root mean squared error between measured and observed discharges. However it uses both the discharge and
    the logarithm of the discharge. The second part aims at improving low flows representation. A description of this
    function is available at [#]_.

    Parameters
    ----------
    y_sim : torch.Tensor
        Simulated discharges.
    y_obs : torch.Tensor
        Observed discharges.

    Returns
    -------
    loss: torch.Tensor
        Weighted root mean squared error

    References
    ----------
    .. [#] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources
        Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404

    """
    # calculate mask to avoid nan in observation to affect the loss
    mask = ~torch.isnan(torch.flatten(y_obs))

    y_sim_masked = torch.flatten(y_sim)[mask]
    y_obs_masked = torch.flatten(y_obs)[mask]
    y_sim_transformed = torch.log10(torch.sqrt(y_sim_masked + 1e-6) + 0.1)
    y_obs_transformed = torch.log10(torch.sqrt(y_obs_masked + 1e-6) + 0.1)

    loss = 0.75 * torch.sqrt(torch.mean((y_sim_masked - y_obs_masked) ** 2)) + 0.25 * torch.sqrt(
        torch.mean((y_sim_transformed - y_obs_transformed) ** 2)
    )

    return loss


def loss_nll(
    params: dict[str, torch.Tensor],
    weights: torch.Tensor,
    dist: Distribution,
    y_obs: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood loss

    Calculate negative log likelihood i.e. the log probability of `y_obs` given a mixture distribution defined
    by `dist`, `params` and `weights`. See specific distribution details below.
    Note: this function returns the mean NLL for each target.

    Parameters
    ----------
    params : dict[str, torch.Tensor]
        Dictionary of distribution parameters. The keys and shapes depend on `dist`
    weights : torch.Tensor
        Mixture weights. Shape [B, N, T]
    dist : Distribution
        Distribution type. See `Distribution` enum for options.
    y_obs : torch.Tensor
        Observed values. Shape [B, N, T]

    Returns
    -------
    torch.Tensor
        NLL for each target. Shape [T]
    """
    y = y_obs.unsqueeze(-2)  # [B, N, 1, T]

    match dist:
        case Distribution.GAUSSIAN:
            loc, scale = params.values()
            scale = torch.clamp(scale, min=1e-6)
            p = (y - loc) / scale
            log_p = -0.5 * p.pow(2) - torch.log(scale) - 0.5 * torch.log(torch.tensor(2 * math.pi))

        case Distribution.LAPLACIAN:
            loc, scale, kappa = params.values()
            scale = torch.clamp(scale, min=1e-6)
            kappa = torch.clamp(kappa, min=1e-6)

            p = (y - loc) / scale

            mask = p >= 0

            log_p = torch.zeros_like(p)

            log_p[mask] = -1 * p[mask] * kappa[mask]
            log_p[~mask] = p[~mask] / kappa[~mask]

            log_p = log_p - torch.log(kappa + 1 / kappa) - torch.log(scale)

    log_w = torch.log(torch.clamp(weights, min=1e-10))
    loss = -torch.logsumexp(log_p + log_w, dim=-2)
    return loss.mean(dim=(0, 1))
