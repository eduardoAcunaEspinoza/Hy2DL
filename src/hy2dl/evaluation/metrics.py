import torch
import xarray as xr

from hy2dl.evaluation.registry import registry
from hy2dl.utils import get_distribution


# --------------------------------------------------------------
#  Deterministic functions
# --------------------------------------------------------------
@registry.register("nse")
def nse(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    """Calculate Nash-Sutcliffe Efficiency."""
    numerator = ((sim - obs) ** 2).sum(dim="date", skipna=True)
    obs_mean = obs.mean(dim="date", skipna=True)
    denominator = ((obs - obs_mean) ** 2).sum(dim="date", skipna=True)
    return 1.0 - (numerator / denominator)


@registry.register("rmse")
def rmse(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    """Calculate Root Mean Squared Error."""
    mse = ((sim - obs) ** 2).mean(dim="date", skipna=True)
    return mse**0.5


# --------------------------------------------------------------
#  Deterministic functions, applicable only to forecast models
# --------------------------------------------------------------
@registry.register("pnse", is_forecast_only=True)
def pnse(obs: xr.DataArray, sim: xr.DataArray, persistent: xr.DataArray) -> xr.DataArray:
    """Calculate Persistence Nash-Sutcliffe Efficiency."""
    numerator = ((sim - obs) ** 2).sum(dim="date", skipna=True)
    denominator = ((obs - persistent) ** 2).sum(dim="date", skipna=True)
    return 1.0 - (numerator / denominator)


# --------------------------------------------------------------
#  Probabilistic functions
# --------------------------------------------------------------
@registry.register("nll", is_probabilistic=True)
def nll(dist: str, obs: xr.DataArray, sim: xr.DataArray, mdn_weight: xr.DataArray, **kwargs) -> xr.DataArray:
    distribution = get_distribution(dist)
    params_xr = {k: kwargs[k] for k in distribution.parameters}

    # To avoid repeating code we use the logpdf function from hy2dl.utils.distributions. However there are defined in
    # pytorch, so we need to, temporarily, convert the Xarray DataArrays to tensors.
    obs_t = torch.as_tensor(obs.values, dtype=torch.float32)
    weights_t = torch.as_tensor(mdn_weight.values, dtype=torch.float32)
    params_t = {k: torch.as_tensor(v.values, dtype=torch.float32) for k, v in params_xr.items()}
    with torch.no_grad():
        # Dummy mask of nans in observations
        log_p = distribution.calc_logpdf(params=params_t, weights=weights_t, x=torch.nan_to_num(obs_t, nan=0.0))
        nll_tensor = torch.where(torch.isnan(obs_t), torch.tensor(float("nan")), -log_p)

    nll_da = xr.DataArray(nll_tensor.numpy(), coords=obs.coords, dims=obs.dims)

    return nll_da.mean(dim="date", skipna=True)
