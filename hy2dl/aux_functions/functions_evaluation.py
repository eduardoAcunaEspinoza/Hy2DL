from typing import Dict

import numpy as np
import pandas as pd


def nse(df_results: Dict[str, pd.DataFrame], average: bool = True) -> np.array:
    """Nash--Sutcliffe Efficiency.

    Parameters
    ----------
    df_results : Dict[str, pd.DataFrame]
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
