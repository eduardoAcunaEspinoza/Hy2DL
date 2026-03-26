from pathlib import Path

import xarray as xr


# ------------------------------------------
#  Loss functions
# ------------------------------------------
def nse(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    """Calculate Nash-Sutcliffe Efficiency."""
    numerator = ((sim - obs) ** 2).sum(dim="date", skipna=True)
    obs_mean = obs.mean(dim="date", skipna=True)
    denominator = ((obs - obs_mean) ** 2).sum(dim="date", skipna=True)
    return 1.0 - (numerator / denominator)


def rmse(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    """Calculate Root Mean Squared Error."""
    mse = ((sim - obs) ** 2).mean(dim="date", skipna=True)
    return mse**0.5


def pnse(obs: xr.DataArray, sim: xr.DataArray, persistent: xr.DataArray) -> xr.DataArray:
    """Calculate Persistence Nash-Sutcliffe Efficiency."""
    numerator = ((sim - obs) ** 2).sum(dim="date", skipna=True)
    denominator = ((obs - persistent) ** 2).sum(dim="date", skipna=True)
    return 1.0 - (numerator / denominator)


# Map string names to the actual functions
forecast_metric_registry = {"nse": nse, "rmse": rmse, "pnse": pnse}
simulation_metric_registry = {"nse": nse, "rmse": rmse}


# ------------------------------------------
#  Prepare data and apply metrics
# ------------------------------------------
def _apply_forecast_metric(ds: xr.Dataset, metric_name: str, filter_mask: xr.DataArray = None) -> xr.DataArray:
    """Applies the specified metric across all lead times, handling the necessary data shifting and masking."""

    # Get the metric function based on the provided name
    metric_func = forecast_metric_registry.get(metric_name.lower())
    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(forecast_metric_registry.keys())}")

    lead_times = ds["lead_time"].values

    # Reindex filter_mask to match the dataset's date and gauge_id dimensions, filling missing values with False
    if filter_mask is not None:
        filter_mask = filter_mask.reindex(date=ds.date, gauge_id=ds.gauge_id, fill_value=False)

    results = []
    for lt in lead_times:
        y_sim = ds["y_sim"].sel(lead_time=lt)
        y_obs = ds["y_obs"].shift(date=-int(lt))

        # Create valid mask
        valid_mask = y_sim.notnull() & y_obs.notnull()

        # Apply masks and gather the necessary data for the metric function
        if metric_name.lower() == "pnse":
            if lt > 0:  # Forecast baseline: locked at emission time
                last_available_q = ds["y_obs"]
            else:  # Hindcast baseline: steps back with the target
                last_available_q = ds["y_obs"].shift(date=-int(lt - 1))

            valid_mask = valid_mask & last_available_q.notnull()

        if filter_mask is not None:  # additional custom filter mask (e.g. to evaluate only on certain dates or gauges)
            mask_shifted = filter_mask.shift(date=-int(lt))
            valid_mask = valid_mask & mask_shifted

        kwargs = {"obs": y_obs.where(valid_mask), "sim": y_sim.where(valid_mask)}
        if metric_name.lower() == "pnse":
            kwargs["persistent"] = last_available_q.where(valid_mask)

        # If applicable, inject other variables (e.g, probabilistic variables: mdn_weight, mu, sigma, etc.)
        for var in ds.data_vars:
            if var not in ["y_obs", "y_sim"]:
                var_sliced = ds[var].sel(lead_time=lt)
                kwargs[var] = var_sliced.where(valid_mask)

        # Apply metric
        metric_val = metric_func(**kwargs)
        results.append(metric_val)

    # Concatenate back into a single DataArray
    return xr.concat(results, dim=xr.Variable("lead_time", lead_times))


def _apply_simulation_metric(ds: xr.Dataset, metric_name: str, filter_mask: xr.DataArray = None) -> xr.DataArray:
    """Applies the specified metric, handling the necessary masking."""

    # Get the metric function based on the provided name
    metric_func = simulation_metric_registry.get(metric_name.lower())
    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(simulation_metric_registry.keys())}")

    # Reindex filter_mask to match the dataset's date and gauge_id dimensions, filling missing values with False
    if filter_mask is not None:
        filter_mask = filter_mask.reindex(date=ds.date, gauge_id=ds.gauge_id, fill_value=False)

    y_sim = ds["y_sim"]
    y_obs = ds["y_obs"]

    # Apply masks, gather the necessary data for the metric function and apply metric
    valid_mask = y_sim.notnull() & y_obs.notnull()
    if filter_mask is not None:  # additional custom filter mask (e.g. to evaluate only on certain dates or gauges)
        valid_mask = valid_mask & filter_mask

    kwargs = {"obs": y_obs.where(valid_mask), "sim": y_sim.where(valid_mask)}
    # If applicable, inject other variables (e.g, probabilistic variables: mdn_weight, mu, sigma, etc.)
    for var in ds.data_vars:
        if var not in ["y_obs", "y_sim"]:
            kwargs[var] = ds[var].where(valid_mask)

    metric_val = metric_func(**kwargs)

    return metric_val


def calculate_metrics(
    ds_results: xr.Dataset | Path,
    metrics: str | list[str] = "all",
    forecast_mode: bool = False,
    distribution=None,
    filter_mask: xr.DataArray = None,
    collapse: bool = False,
) -> xr.DataArray:
    """
    Calculate a list of metrics

    Parameters
    ----------
    ds_results : xr.Dataset | Path
        Dataset loaded from the evaluation Zarr store containing 'y_sim' and 'y_obs'.
        Expected dimensions:
        - y_sim: (gauge_id, date, lead_time, feature)
        - y_obs: (gauge_id, date, feature)
    metrics : str | List[str]
        List of metric names to calculate.
    forecast_mode : bool
        True if the dataset is from a forecast model (with lead_time dimension), False if from a simulation model.
    filter_mask : xr.DataArray, optional
        Boolean DataArray to filter values during evaluation. Expected dimensions (gauge_id, date).
    collapse : bool
        True to return a single value across all dimensions

    Returns
    -------
    xr.DataArray
        DataArray containing the calculated metrics If collapse=True, returns a single value across all dimensions.

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    if isinstance(metrics, str):
        if metrics.lower() == "all":
            metrics = (
                list(forecast_metric_registry.keys()) if forecast_mode else list(simulation_metric_registry.keys())
            )
        else:
            metrics = [metrics]  # Convert single string to a list

    da_list = []
    for metric in metrics:
        if forecast_mode:
            da_list.append(_apply_forecast_metric(ds_results, metric, filter_mask).compute())
        else:
            da_list.append(_apply_simulation_metric(ds_results, metric, filter_mask).compute())

    # Concatenate the list of DataArrays along a new 'metric' dimension
    da_combined = xr.concat(da_list, dim=xr.Variable("metric", metrics))

    # Clear encodings to avoid issues when saving to Zarr
    da_combined.encoding.clear()
    for coord_name in list(da_combined.coords):
        if da_combined[coord_name].dtype == "O":
            da_combined[coord_name] = da_combined[coord_name].astype(str)
        da_combined[coord_name].encoding.clear()

    # return
    return da_combined.median(skipna=True) if collapse else da_combined
