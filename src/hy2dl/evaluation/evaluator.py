import inspect
from pathlib import Path

import xarray as xr

import hy2dl.evaluation.metrics as _
from hy2dl.evaluation.registry import registry


def _apply_forecast_metric(
    ds: xr.Dataset, metric_name: str, filter_mask: xr.DataArray = None, distribution: str = None
) -> xr.DataArray:
    """Applies the specified metric across all lead times, handling the necessary data shifting and masking."""

    # Get metric name and function from the registry
    metric_meta = registry.get(metric_name)
    metric_func = metric_meta.func
    sig = inspect.signature(metric_func)  # finds out which arguments the metric needs

    lead_times = ds["lead_time"].values
    # Reindex filter_mask to match the dataset's date and gauge_id dimensions, filling missing values with False
    if filter_mask is not None:
        if set(filter_mask.dims) != {"gauge_id", "date", "feature"}:
            raise ValueError(
                f"filter_mask must strictly have dimensions gauge_id, date, feature.Got: {set(filter_mask.dims)}"
            )
        filter_mask = filter_mask.reindex(date=ds.date, gauge_id=ds.gauge_id, fill_value=False)

    results = []
    for lt in lead_times:
        y_sim = ds["y_sim"].sel(lead_time=lt)
        y_obs = ds["y_obs"].shift(date=-int(lt))

        # For persistence-based metrics (like PNSE)
        if lt > 0:  # forecast baseline: locked at emission time
            last_available_q = ds["y_obs"]
        else:  # hindcast baseline: steps back with the target
            last_available_q = ds["y_obs"].shift(date=-int(lt - 1))

        # Build "pool" of possible arguments any metric might want
        args_pool = {
            "obs": y_obs,
            "sim": y_sim,
            "persistent": last_available_q,
            "dist": distribution,
        }
        # If applicable, inject other variables (e.g, probabilistic variables: mdn_weight, mu, sigma, etc.)
        for var in ds.data_vars:
            if var not in ["y_obs", "y_sim"]:
                args_pool[var] = ds[var].sel(lead_time=lt)

        selected_args = {}  # filter to include only what this metrics needs
        for param_name, param in sig.parameters.items():
            if param_name in args_pool:
                selected_args[param_name] = args_pool[param_name]
            elif param.kind == inspect.Parameter.VAR_KEYWORD:  # metric has a **kwargs catch-all
                # assigned everything that is remaining in the pool
                selected_args.update({k: v for k, v in args_pool.items() if k not in selected_args})

        # find notnull mask across all arguments
        mask = filter_mask.shift(date=-int(lt)) if filter_mask is not None else True
        for k, v in selected_args.items():
            if k in ["obs", "sim", "persistent"]:
                mask = mask & v.notnull()

        # apply mask to selected arguments
        kwargs = {}
        for key, val in selected_args.items():
            if isinstance(val, xr.DataArray):
                masked_val = val.where(mask).transpose(*val.dims)
                kwargs[key] = masked_val
            else:
                kwargs[key] = val

        # Apply metric.
        metric_val = metric_func(**kwargs)
        results.append(metric_val)

    return xr.concat(results, dim=xr.Variable("lead_time", lead_times))


def _apply_simulation_metric(
    ds: xr.Dataset, metric_name: str, filter_mask: xr.DataArray = None, distribution: str = None
) -> xr.DataArray:
    """Applies the specified metric, handling the necessary masking."""

    # Get metric name and function from the registry
    metric_meta = registry.get(metric_name)
    metric_func = metric_meta.func
    sig = inspect.signature(metric_func)  # finds out which arguments the metric needs

    # Reindex filter_mask to match the dataset's date and gauge_id dimensions, filling missing values with False
    if filter_mask is not None:
        if set(filter_mask.dims) != {"gauge_id", "date", "feature"}:
            raise ValueError(
                f"filter_mask must strictly have dimensions gauge_id, date, feature.Got: {set(filter_mask.dims)}"
            )
        filter_mask = filter_mask.reindex(date=ds.date, gauge_id=ds.gauge_id, fill_value=False)

    # Build "pool" of possible arguments any metric might want
    args_pool = {
        "obs": ds["y_obs"],
        "sim": ds["y_sim"],
        "dist": distribution,
    }
    # If applicable, inject other variables (e.g, probabilistic variables: mdn_weight, mu, sigma, etc.)
    for var in ds.data_vars:
        if var not in ["y_obs", "y_sim"]:
            args_pool[var] = ds[var]

    selected_args = {}  # filter to include only what this metrics needs
    for param_name, param in sig.parameters.items():
        if param_name in args_pool:
            selected_args[param_name] = args_pool[param_name]
        elif param.kind == inspect.Parameter.VAR_KEYWORD:  # metric has a **kwargs catch-all
            # assigned everything that is remaining in the pool
            selected_args.update({k: v for k, v in args_pool.items() if k not in selected_args})

    # find notnull mask
    mask = filter_mask if filter_mask is not None else True
    for k, v in selected_args.items():
        if k in ["obs", "sim", "persistent"]:
            mask = mask & v.notnull()

    # apply mask to selected arguments
    kwargs = {}
    for key, val in selected_args.items():
        if isinstance(val, xr.DataArray):
            masked_val = val.where(mask).transpose(*val.dims)
            kwargs[key] = masked_val
        else:
            kwargs[key] = val

    # Apply metric.
    metric_val = metric_func(**kwargs)

    return metric_val


def calculate_metrics(
    ds_results: xr.Dataset | Path,
    metric_name: str | list[str] = "all",
    forecast_mode: bool = False,
    distribution=None,
    filter_mask: xr.DataArray = None,
) -> xr.DataArray:
    """
    Calculate a list of metrics

    Parameters
    ----------
    ds_results : xr.Dataset | Path
        Dataset loaded from the evaluation Zarr.
    metric_name : str | list[str]
        List of metric names to calculate, or "all" to run all valid metrics.
    forecast_mode : bool
        True if the dataset is from a forecast model (with lead_time dimension).
    distribution : str, optional
        Distribution string required for calculating probabilistic metrics like NLL.
    filter_mask : xr.DataArray, optional
        Boolean DataArray to filter values during evaluation. Expected dims (gauge_id, date).

    Returns
    -------
    xr.DataArray
        DataArray containing the calculated metrics.

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    is_probabilistic = distribution is not None
    if isinstance(metric_name, str):
        if metric_name.lower() == "all":
            # The registry safely filters out probabilistic metrics if no distribution is provided, and filters out
            # forecast metrics if forecast_mode is False.
            requested_metrics = registry.get_available(forecast_mode=forecast_mode, probabilistic=is_probabilistic)
        else:
            requested_metrics = [metric_name.lower()]
    else:
        requested_metrics = [m.lower() for m in metric_name]

    # Safety Check: See that we have all the necessary inputs for the requested metrics
    for m_name in requested_metrics:
        meta = registry.get(m_name)
        if meta.is_probabilistic and not is_probabilistic:
            raise ValueError(f"Metric '{m_name}' is probabilistic. You must specify a 'distribution'.")

    array_metrics = []
    for m_name in requested_metrics:
        if forecast_mode:
            m_value = _apply_forecast_metric(
                ds=ds_results, metric_name=m_name, filter_mask=filter_mask, distribution=distribution
            )
        else:
            m_value = _apply_simulation_metric(
                ds=ds_results, metric_name=m_name, filter_mask=filter_mask, distribution=distribution
            )

        array_metrics.append(m_value)

    # Concatenate the list of DataArrays along a new 'metric' dimension
    metrics = xr.concat(array_metrics, dim=xr.Variable("metric", requested_metrics))
    metrics = metrics.compute()

    # Clear encodings to avoid issues when saving to Zarr
    metrics.encoding.clear()
    for coord_name in list(metrics.coords):
        if metrics[coord_name].dtype == "O":
            metrics[coord_name] = metrics[coord_name].astype(str)
        metrics[coord_name].encoding.clear()

    # return
    return metrics
