from pathlib import Path

import xarray as xr


def NSE(ds_results: xr.Dataset | Path, average: bool = False) -> xr.Dataset | float:
    """Nash--Sutcliffe Efficiency.

    Parameters
    ----------
    ds_results : xr.Dataset | Path
         xarray Dataset or path to a zarr file containing the results of the evaluation.
        Expected dimensions: ("gauge_id", "date", "feature").
    average : bool
        True to return a single median NSE value across all gauges and features.
        False to return the unaggregated NSE for each basin and feature.

    Returns
    -------
    float | xr.DataArray
        If average==True: returns a single float (the global median NSE).
        If average==False: returns an xr.DataArray with dimensions ("gauge_id", "feature").

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    y_sim = ds_results["y_sim"]
    y_obs = ds_results["y_obs"]

    # mask-out nans
    valid_mask = y_sim.notnull() & y_obs.notnull()
    y_sim = y_sim.where(valid_mask)
    y_obs = y_obs.where(valid_mask)

    # Calculate metric
    numerator = ((y_sim - y_obs) ** 2).sum(dim="date", skipna=True)
    obs_mean = y_obs.mean(dim="date", skipna=True)
    denominator = ((y_obs - obs_mean) ** 2).sum(dim="date", skipna=True)
    nse = 1.0 - (numerator / denominator)

    return nse.compute().median(skipna=True).item() if average else nse.compute()


def RMSE(ds_results: xr.Dataset | Path, average: bool = False) -> xr.DataArray | float:
    """Root Mean Square Error.

    Parameters
    ----------
    ds_results : xr.Dataset | Path
        xarray Dataset containing 'y_sim' and 'y_obs' variables.
        Expected dimensions: ("gauge_id", "date", "feature").
    average : bool
        True to return a single median RMSE value across all basins and features.
        False to return the unaggregated RMSE for each basin and feature.

    Returns
    -------
    float | xr.DataArray
        If average==True: returns a single float (the global median RMSE).
        If average==False: returns an xr.DataArray with dimensions ("gauge_id", "feature").

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    y_sim = ds_results["y_sim"]
    y_obs = ds_results["y_obs"]

    # mask-out nans
    valid_mask = y_sim.notnull() & y_obs.notnull()
    y_sim = y_sim.where(valid_mask)
    y_obs = y_obs.where(valid_mask)

    # Calculate metric
    squared_error = (y_sim - y_obs) ** 2
    mean_squared_error = squared_error.mean(dim="date", skipna=True)
    rmse_vals = mean_squared_error**0.5

    return rmse_vals.compute().median(skipna=True).item() if average else rmse_vals.compute()


def forecast_NSE(ds_results: xr.Dataset | Path, filter_mask: xr.DataArray = None) -> xr.DataArray:
    """Calculate the Nash--Sutcliffe Efficiency for each forecasted lead time.

    Parameters
    ----------
    ds_results : xr.Dataset | Path
        Dataset loaded from the evaluation Zarr store containing 'y_sim' and 'y_obs'.
        Expected dimensions:
        - y_sim: (gauge_id, date, lead_time, feature)
        - y_obs: (gauge_id, date, feature)
    filter_mask : xr.DataArray, optional
        Boolean DataArray to filter values during evaluation. Expected dimensions (gauge_id, date).

    Returns
    -------
    da_nse: xr.DataArray
        DataArray indexed by the gauge_id. The columns are the NSE for each lead time and feature.

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    nse_per_leadtime = []
    lead_times = ds_results["lead_time"].values

    for lt in lead_times:
        # simulated values for the current lead time
        y_sim = ds_results["y_sim"].sel(lead_time=lt)

        # shift observations backwards to align with the initialization date. A forecast emitted on 'date' with lead
        # time 'lt' verifies against observation at 'date + lt'
        y_obs = ds_results["y_obs"].shift(date=-int(lt))

        # create a valid mask (ensure neither sim nor obs is NaN)
        valid_mask = y_sim.notnull() & y_obs.notnull()
        if filter_mask is not None:
            valid_mask = valid_mask & filter_mask

        # apply mask
        y_sim_v = y_sim.where(valid_mask)
        y_obs_v = y_obs.where(valid_mask)

        # 4. Calculate vectorized NSE
        numerator = ((y_sim_v - y_obs_v) ** 2).sum(dim="date", skipna=True)
        obs_mean = y_obs_v.mean(dim="date", skipna=True)
        denominator = ((y_obs_v - obs_mean) ** 2).sum(dim="date", skipna=True)
        nse_per_leadtime.append(1 - (numerator / denominator))

    # concatenate all lead times back into a single DataArray
    da_nse = xr.concat(nse_per_leadtime, dim=xr.Variable("lead_time", lead_times))

    return da_nse.compute()


def forecast_PNSE(ds_results: xr.Dataset | Path, filter_mask: xr.DataArray = None) -> xr.DataArray:
    """Calculate the persistence Nash--Sutcliffe Efficiency for each forecasted lead time.

    Parameters
    ----------
    ds_results : xr.Dataset | Path
        Dataset loaded from the evaluation Zarr store containing 'y_sim' and 'y_obs'.
        Expected dimensions:
        - y_sim: (gauge_id, date, lead_time, feature)
        - y_obs: (gauge_id, date, feature)
    filter_mask : xr.DataArray, optional
        Boolean DataArray to filter values during evaluation. Expected dimensions (gauge_id, date).

    Returns
    -------
    da_pnse: xr.DataArray
        DataArray indexed by the gauge_id. The columns are the PNSE for each lead time and feature.

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    pnse_per_leadtime = []
    lead_times = ds_results["lead_time"].values

    # The persistent value is the observation at the time the forecast was emitted
    y_persistent = ds_results["y_obs"]

    for lt in lead_times:
        # simulated values for the current lead time
        y_sim = ds_results["y_sim"].sel(lead_time=lt)

        # shift observations backwards to align with the initialization date. A forecast emitted on 'date' with lead
        # time 'lt' verifies against observation at 'date + lt'
        y_obs = ds_results["y_obs"].shift(date=-int(lt))

        # create a valid mask (ensure sim, target, and persistent are not NaN)
        valid_mask = y_sim.notnull() & y_obs.notnull() & y_persistent.notnull()
        if filter_mask is not None:
            valid_mask = valid_mask & filter_mask

        # apply the mask
        y_sim_v = y_sim.where(valid_mask)
        y_obs_v = y_obs.where(valid_mask)
        y_persistent_v = y_persistent.where(valid_mask)

        # 4. Calculate vectorized PNSE
        numerator = ((y_sim_v - y_obs_v) ** 2).sum(dim="date", skipna=True)
        denominator = ((y_obs_v - y_persistent_v) ** 2).sum(dim="date", skipna=True)

        pnse_per_leadtime.append(1 - (numerator / denominator))

    # concatenate all lead times back into a single DataArray
    da_pnse = xr.concat(pnse_per_leadtime, dim=xr.Variable("lead_time", lead_times))

    return da_pnse.compute()


def forecast_RMSE(ds_results: xr.Dataset | Path, filter_mask: xr.DataArray = None) -> xr.DataArray:
    """Calculate the Root Mean Squared Error (RMSE) for each forecasted lead time.

    Parameters
    ----------
    ds_results : xr.Dataset | Path
        Dataset loaded from the evaluation Zarr store containing 'y_sim' and 'y_obs'.
        Expected dimensions:
        - y_sim: (gauge_id, date, lead_time, feature)
        - y_obs: (gauge_id, date, feature)
    filter_mask : xr.DataArray, optional
        Boolean DataArray to filter values during evaluation. Expected dimensions (gauge_id, date).

    Returns
    -------
    da_rmse: xr.DataArray
        DataArray indexed by the gauge_id. The columns are the RMSE for each lead time and feature.

    """
    if isinstance(ds_results, Path):
        ds_results = xr.open_zarr(ds_results)

    rmse_per_leadtime = []
    lead_times = ds_results["lead_time"].values

    for lt in lead_times:
        # simulated values for the current lead time
        y_sim = ds_results["y_sim"].sel(lead_time=lt)

        # shift observations backwards to align with the initialization date. A forecast emitted on 'date' with lead
        # time 'lt' verifies against observation at 'date + lt'
        y_obs = ds_results["y_obs"].shift(date=-int(lt))

        # create a valid mask (ensure neither sim nor obs is NaN)
        valid_mask = y_sim.notnull() & y_obs.notnull()
        if filter_mask is not None:
            valid_mask = valid_mask & filter_mask

        # apply the mask
        y_sim_v = y_sim.where(valid_mask)
        y_obs_v = y_obs.where(valid_mask)

        # 4. Calculate vectorized RMSE
        # Calculate the mean of the squared errors along the date dimension, then take the square root
        mse = ((y_sim_v - y_obs_v) ** 2).mean(dim="date", skipna=True)
        rmse = mse**0.5

        rmse_per_leadtime.append(rmse)

    # 5. Concatenate all lead times back into a single DataArray
    da_rmse = xr.concat(rmse_per_leadtime, dim=xr.Variable("lead_time", lead_times))

    return da_rmse.compute()
