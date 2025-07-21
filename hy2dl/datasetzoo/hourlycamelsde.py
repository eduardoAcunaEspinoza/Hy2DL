# import necessary packages
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from hy2dl.datasetzoo.camelsde import CAMELS_DE
from hy2dl.utils.config import Config


class Hourly_CAMELS_DE(CAMELS_DE):
    """
     Class to process hourly data in similar format as the CAMELS DE dataset.

     The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
     code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS DE.

    Parameters
     ----------
     cfg : Config
         Configuration file.
     time_period : {'training', 'validation', 'testing'}
         Defines the period for which the data will be loaded.
     check_NaN : Optional[bool], default=True
         Whether to check for NaN values while processing the data. This should typically be True during training,
         and can be set to False during evaluation (validation/testing).
     entity : Optional[str], default=None
         ID of the entity (e.g., single catchment's ID) to be analyzed

     References
     ----------
     .. [1] Loritz, R., Dolich, A., Acuña Espinoza, E., Ebeling, P., Guse, B., Götte, J., Hassler, S. K., Hauffe,
         C., Heidbüchel, I., Kiesel, J., Mälicke, M., Müller-Thomy, H., Stölzle, M., & Tarasova, L. (2024).
         CAMELS-DE: Hydro-meteorological time series and attributes for 1555 catchments in Germany.
         Earth System Science Data Discussions, 2024, 1–30. https://doi.org/10.5194/essd-2024-318
     .. [2] Dolich, A., Espinoza, E. A., Ebeling, P., Guse, B., Götte, J., Hassler, S., Hauffe, C., Kiesel, J.,
         Heidbüchel, I., Mälicke, M., Müller-Thomy, H., Stölzle, M., Tarasova, L., & Loritz, R. (2024).
         CAMELS-DE: hydrometeorological time series and attributes for 1582 catchments in Germany (1.0.0) [Data set].
         Zenodo. https://doi.org/10.5281/zenodo.13837553

    """

    def __init__(
        self,
        cfg: Config,
        time_period: str,
        check_NaN: Optional[bool] = True,
        entities_ids: Optional[Union[str, List[str]]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(Hourly_CAMELS_DE, self).__init__(
            cfg=cfg, time_period=time_period, check_NaN=check_NaN, entities_ids=entities_ids
        )

    def _read_data(self, catch_id: str) -> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        catch_id : str
            identifier of the basin.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        # Read hourly data
        path_timeseries = self.cfg.path_data / "hourly" / f"CAMELS_DE_1h_hydromet_timeseries_{catch_id}.csv"
        # load time series
        df_hourly = pd.read_csv(path_timeseries, index_col="time", parse_dates=["time"])

        # Fill gaps in the precipitation column
        df_hourly = self._fill_precipitation_gaps(df=df_hourly)

        # Load variables from CAMELS DE (daily) and resample
        path_daily_timeseries = self.cfg.path_data / "timeseries" / f"CAMELS_DE_hydromet_timeseries_{catch_id}.csv"
        df_resampled = pd.read_csv(path_daily_timeseries, index_col="date", parse_dates=["date"])
        df_resampled = df_resampled.loc[:, "precipitation_mean"].resample("1h").ffill() / 24
        df_resampled = df_resampled.loc[df_hourly.index.intersection(df_resampled.index)]

        # Create new column where gaps in hourly precipitation are filled with the daily resampled version
        df_hourly["precipitation_resampled"] = df_hourly["precipitation_sum_mean"].combine_first(df_resampled)

        return df_hourly

    def _fill_precipitation_gaps(self, df):
        """
        Fills gaps in the hourly precipitation column using the following rules:
        - If the gap is less than 3 hours: apply linear interpolation.
        - If the gap is between 3 and 6 hours and the values before & after the gap are 0: fill with 0.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries with the gaps filled
        """

        df_filled = df.copy()  # Create a copy to avoid modifying the original DataFrame
        col = "precipitation_sum_mean"

        # Identify missing values
        df_filled["is_missing"] = df_filled[col].isna()

        # Identify missing groups
        df_filled["missing_group"] = (df_filled["is_missing"] != df_filled["is_missing"].shift()).cumsum() * df_filled[
            "is_missing"
        ]

        # Count length of each missing group
        gap_lengths = df_filled[df_filled["is_missing"]].groupby("missing_group").size()

        # Iterate over missing groups
        for group_id, gap_length in gap_lengths.items():
            if gap_length <= 3:
                # Apply linear interpolation for gaps < 3 hours
                df_filled.loc[df_filled["missing_group"] == group_id, col] = df_filled[col].interpolate(method="linear")

            elif 3 < gap_length <= 6:
                # Get the index range of the missing group
                missing_indices = df_filled[df_filled["missing_group"] == group_id].index

                # Get values before and after the gap
                before_value = (
                    df_filled.loc[missing_indices[0] - pd.Timedelta(hours=1), col]
                    if missing_indices[0] - pd.Timedelta(hours=1) in df_filled.index
                    else np.nan
                )
                after_value = (
                    df_filled.loc[missing_indices[-1] + pd.Timedelta(hours=1), col]
                    if missing_indices[-1] + pd.Timedelta(hours=1) in df_filled.index
                    else np.nan
                )

                # Filled if before and after values are 0
                if before_value == 0 and after_value == 0:
                    df_filled.loc[df_filled["missing_group"] == group_id, col] = 0

        # Drop helper columns
        df_filled.drop(columns=["is_missing", "missing_group"], inplace=True)

        return df_filled
