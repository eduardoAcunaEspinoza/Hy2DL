import numpy as np
import pandas as pd
import xarray as xr


def nse(df_results: dict[str, pd.DataFrame], average: bool = True) -> np.array:
    """Nash--Sutcliffe Efficiency.

    Parameters
    ----------
    df_results : dict[str, pd.DataFrame]
        Dictionary, where each key is associated with a basin_id and each item is a pandas DataFrame.
        Each dataframe should contained at least two columns: y_sim for the simulated values and y_obs for the observed
        values.
    average : bool
        True if one wants to average the NSE over all the basin (items of the dictionary), or False
        if one wants the value for each one

    Returns
    -------
    loss: np.array
        If average==True returns one value for all basins. If average==False returns the NSE for each
        element.

    """
    loss = []
    # Go for each element (basin) of the dictionary
    for basin in df_results.values():
        # Read values
        y_sim = basin["y_sim"].values
        y_obs = basin["y_obs"].values

        # Mask values based on NaN from y_sim (this occurs in validation and testing if there are NaN in the inputs)
        mask_y_sim = ~np.isnan(y_sim)
        y_sim = y_sim[mask_y_sim]
        y_obs = y_obs[mask_y_sim]

        # Mask values based on NaN from y_obs (this occurs in validation and testing if there are NaN in the output)
        mask_y_obs = ~np.isnan(y_obs)
        y_sim = y_sim[mask_y_obs]
        y_obs = y_obs[mask_y_obs]

        # Calculate NSE
        if y_sim.size > 1 and y_obs.size > 1:
            loss.append(1.0 - np.sum((y_sim - y_obs) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2))
        else:
            loss.append(np.nan)

    return np.nanmedian(loss) if average else np.asarray(loss)


def forecast_NSE(results: dict[str, pd.DataFrame], filter: dict[str, pd.DataFrame] = None) -> dict[str, pd.DataFrame]:
    """Calculate the Nash--Sutcliffe Efficiency for each forecasted lead time.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dictionary, where each key is associated with a basin_id and each item is a datetime indexed pandas DataFrame.


    Returns
    -------
    df_loss: pd.DataFrame
        Daframe indexed by the basin_id and the columns are the NSE for each lead time.

    """
    nse_per_basin = []
    for basin, df in results.items():
        nrow, ncol = df.shape
        n_lead_times = ncol - 1  # first column is the observed value
        last_forecast_row = nrow - n_lead_times  # row where the last forecast is emmited

        nse_per_leadtime = []
        # Iterate through the different lead times
        for i in range(n_lead_times):
            # The simulated values for each lead time are located in the different columns
            y_sim = df.iloc[:last_forecast_row, i + 1]
            # The observed values are always located in the first column. To extract the observed value associated with
            # each lead time, we select the observed values shifted by the number of lead times.
            y_obs = df.iloc[i + 1 : last_forecast_row + i + 1, 0]

            # If there is an additional filter for the values considered during evaluation, we apply it
            if filter is not None:
                y_sim = y_sim[filter[basin][:last_forecast_row].values].values
                y_obs = y_obs[filter[basin][:last_forecast_row].values].values
            else:
                y_sim = y_sim.values
                y_obs = y_obs.values

            # Calcule NSE
            numerator = np.nansum((y_sim - y_obs) ** 2, axis=0)
            denominator = np.nansum((y_obs - np.nanmean(y_obs, axis=0)) ** 2, axis=0)
            nse_per_leadtime.append(1 - numerator / denominator)

        nse_per_basin.append(nse_per_leadtime)

    df_loss = pd.DataFrame(nse_per_basin, index=list(results.keys()), columns=df.columns[1:])
    df_loss.index.name = "gauge_id"

    return df_loss


def forecast_PNSE(results: dict[str, pd.DataFrame], filter: dict[str, pd.DataFrame] = None) -> dict[str, pd.DataFrame]:
    """Calculate the persistence Nash--Sutcliffe Efficiency for each forecasted lead time.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dictionary, where each key is associated with a basin_id and each item is a datetime indexed pandas DataFrame.


    filter :

    Returns
    -------
    df_loss: pd.DataFrame
        Daframe indexed by the basin_id and the columns are the NSE for each lead time.

    """
    nse_per_basin = []
    for basin, df in results.items():
        nrow, ncol = df.shape
        n_lead_times = ncol - 1  # first column is the observed value
        last_forecast_row = nrow - n_lead_times  # row where the last forecast is emmited

        nse_per_leadtime = []
        # Iterate through the different lead times
        for i in range(n_lead_times):
            # The simulated values for each lead time are located in the different columns
            y_sim = df.iloc[:last_forecast_row, i + 1]
            # The observed values are always located in the first column. To extract the observed value associated with
            # each lead time, we select the observed values shifted by the number of lead times.
            y_obs = df.iloc[i + 1 : last_forecast_row + i + 1, 0]
            # To calculate the persistent NSE, we normalize by the difference between the observed value and time t and
            # the observed value at the time the forecast was emmited.
            persistent = df.iloc[0:last_forecast_row, 0]

            # If there is an additional filter for the values considered during evaluation, we apply it
            if filter is not None:
                y_sim = y_sim[filter[basin][:last_forecast_row].values].values
                y_obs = y_obs[filter[basin][:last_forecast_row].values].values
                persistent = persistent[filter[basin][:last_forecast_row].values].values
            else:
                y_sim = y_sim.values
                y_obs = y_obs.values
                persistent = persistent.values

            # Calcule PNSE
            numerator = np.nansum((y_sim - y_obs) ** 2, axis=0)
            denominator = np.nansum((y_obs - persistent) ** 2, axis=0)
            nse_per_leadtime.append(1 - numerator / denominator)

        nse_per_basin.append(nse_per_leadtime)

    df_loss = pd.DataFrame(nse_per_basin, index=list(results.keys()), columns=df.columns[1:])
    df_loss.index.name = "gauge_id"

    return df_loss
