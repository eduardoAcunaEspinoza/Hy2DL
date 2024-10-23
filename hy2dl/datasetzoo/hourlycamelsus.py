# import necessary packages
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from hy2dl.datasetzoo.camelsus import CAMELS_US


class HourlyCamelsUS(CAMELS_US):
    """Class to process hourly data in similar format as the CAMELS US dataset.

    This class process hourly data stored in the same format as CAMELS US [1]_ [2]_. It also allows to load daily
    information from the CAMELS_US dataset and upsample it to hourly. Moreover, we can read the static attributes
    using the function defined in the CAMELS_US class.

    This class and its methods were adapted from Neural Hydrology [3]_.

    Parameters
    ----------
    dynamic_input : Union[List[str], Dict[str, List[str]]]
        Name of variables used as dynamic series input in the data driven model. In most cases is a single list. If we
        are using multiple frequencies, it is a dictionary where the key is the frequency and the value is a list of
        variables.
    forcing : List[str]
        specificy which forcing data will be used (e.g. daymet, maurer, ndlas, ndlas_hourly, etc.)
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
    .. [1] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett,
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223,
        doi:10.5194/hess-19-209-2015, 2015
    .. [2] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    .. [3] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    def __init__(
        self,
        dynamic_input: Union[List[str], Dict[str, List[str]]],
        forcing: List[str],
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
        # Run the __init__ method of CAMELS_US class
        super(HourlyCamelsUS, self).__init__(
            dynamic_input=dynamic_input,
            forcing=forcing,
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
        """Read a specific catchment timeseries into a dataframe.

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        dfs = []
        for forcing in self.forcing:
            if forcing[-7:] == "_hourly":
                df = self._load_hourly_data(catch_id=catch_id, forcing=forcing)
            else:
                # load daily CAMELS forcings and upsample to hourly
                df, _ = self._load_camelsus_data(catch_id=catch_id, forcing=forcing)
                df = df.resample("1h").ffill()
            if len(self.forcing) > 1:
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # Read discharges and add them to current dataframe
        df = df.join(self._load_hourly_discharge(catch_id=catch_id))

        return df

    def _load_hourly_data(self, catch_id: str, forcing: str) -> pd.DataFrame:
        """Read a specific catchment forcing timeseries

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.
        forcing : str
            e.g. ndlas_hourly'

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        path_timeseries = Path(self.path_data) / "hourly" / f"{forcing}" / f"{catch_id}_hourly_nldas.csv"
        # load time series
        df = pd.read_csv(path_timeseries, index_col=["date"], parse_dates=["date"])

        return df

    def _load_hourly_discharge(self, catch_id: str) -> pd.DataFrame:
        """Read a specific catchment discharge timeseries

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.

        Returns
        -------
        df: pd.Series
            Time-index pandas.Series of the discharge values (mm/h)

        """
        # Create a path to read the data
        streamflow_path = Path(self.path_data) / "hourly/usgs_streamflow" / f"{catch_id}-usgs-hourly.csv"

        # load time series
        df = pd.read_csv(streamflow_path, index_col=["date"], parse_dates=["date"])

        # Replace invalid discharge values by NaN
        df["QObs(mm/h)"] = df["QObs(mm/h)"].apply(lambda x: np.nan if x < 0 else x)

        return df
