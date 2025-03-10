# import necessary packages
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from hy2dl.datasetzoo.camelsde import CAMELS_DE


class HourlyCAMELS_DE(CAMELS_DE):
    """
    Class to process hourly data in similar format as the CAMELS DE dataset.

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS DE.

    Parameters
    ----------
    dynamic_input : Union[List[str], Dict[str, List[str]]]
        Name of variables used as dynamic series input in the data driven model. In most cases is a single list. If we
        are using multiple frequencies, it is a dictionary where the key is the frequency and the value is a list of
        variables.
    target : List[str]
        Target variable that will be used to train the model
    sequence_length : int
        Sequence length used for the model
    time_period : List[str]
        Initial and final date of the time period of interest. It should be consistent with the resolution of the data.
        e.g.  daily: ["1990-10-01","2003-09-30"], hourly: ["1990-10-01 00:00:00","2003-09-30 23:00:00"]
    path_data : str
        Path to the folder were the data is stored
    path_entities : str = None
        Path to a txt file that contain the id of the entities (e.g. catchment`s ids) that will be analyzed. An
        alternative option is to specify the entity directly with the "entity" parameter.
    entity : str = None
        id of the entity (e.g. single catchment`s id) that will be analyzed. An alternative option is to specify the
        path to a txt file containing multiple entities, using the "path_entities" parameter.
    check_NaN : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data. This parameter should be
        True during training, and it can be switch to False during evaluation (validation/testing)
    path_addional_features : Optional[str] = None
        Allows the option to add any arbitrary data that is not included in the standard data sets. Path to a pickle
        file, containing a dictionary with each key corresponding to one basin id  and the value is a date-time indexed
        pandas DataFrame, where the columns are the additional features. Default value is None.
        For deliberate exclusion of samples, the dictionary's value is a date-time indexed pandas DataFrame with one
        single column named "ablation_flag", containing 0/1 flags (0 for exclusion).
    predict_last_n : Optional[int] = 1
        Number of timesteps (e.g. days, hours) that will be output by the model as predictions. Default value is 1.
    static_input : Optional[List[str]] = None
        Name of static attributes used as input in the lstm (e.g. catchment attributes). Default value is None.
    conceptual_input : Optional[List[str]] = None
        Name of variables used as dynamic series input in the conceptual model. Mandatory when using hybrid models.
    custom_freq_processing : Optional[Dict[str, int]] = None
        Necessary when using multiple frequencies (e.g. daily/hourly). We specify the information as a nested
        dictionary. The key of the first dictionary is the frequency name (e.g "1D", "1h"...). The values of the
        first dictionary (keys of the second one) are "freq_factor" and "n_steps". The values of the second dictionary
        are the respective values. The order of the frequencies should be consistent with the way we want to process
        the data (e.g. first daily, then hourly). In case we have more than two frequencies, the frequency factor is
        always relative to the higher frequency. Default value is None.
        e.g. custom_freq_processing = {"1D": {"freq_factor": 24, "n_steps": 351}, "1h": {"freq_factor": 1, "n_steps": 14*24}}
    dynamic_embedding : Optional[bool] = False
        True if we want to use embeddings (custom linear layers) for the different frequencies. This is necessary when
        one has different number of inputs for each frequency.
    unique_prediction_blocks : Optional[bool] = False
        If predicting_last_n>1, we can decide to use unique_prediction_blocks = True if we do not want to have
        overlapping blocks of training data.

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
        dynamic_input: Union[List[str], Dict[str, List[str]]],
        target: List[str],
        sequence_length: int,
        time_period: List[str],
        path_data: str,
        path_entities: str = None,
        entity: str = None,
        check_NaN: bool = True,
        path_additional_features: Optional[str] = "",
        predict_last_n: Optional[int] = 1,
        static_input: Optional[List[str]] = None,
        conceptual_input: Optional[List[str]] = None,
        custom_freq_processing: Optional[Dict[str, int]] = None,
        dynamic_embedding: Optional[bool] = False,
        unique_prediction_blocks: Optional[bool] = False,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(HourlyCAMELS_DE, self).__init__(
            dynamic_input=dynamic_input,
            target=target,
            sequence_length=sequence_length,
            time_period=time_period,
            path_data=path_data,
            path_entities=path_entities,
            entity=entity,
            check_NaN=check_NaN,
            path_additional_features=path_additional_features,
            predict_last_n=predict_last_n,
            static_input=static_input,
            conceptual_input=conceptual_input,
            custom_freq_processing=custom_freq_processing,
            dynamic_embedding=dynamic_embedding,
            unique_prediction_blocks=unique_prediction_blocks,
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
        path_timeseries = Path(self.path_data) / "hourly" / f"CAMELS_DE_1h_hydromet_timeseries_{catch_id}.csv"
        # load time series
        df_hourly = pd.read_csv(path_timeseries, index_col="time", parse_dates=["time"])

        # Fill gaps in the precipitation column
        df_hourly = self._fill_precipitation_gaps(df=df_hourly)

        # Load variables from CAMELS DE (daily) and resample
        path_daily_timeseries = Path(self.path_data) / "timeseries" / f"CAMELS_DE_hydromet_timeseries_{catch_id}.csv"
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
