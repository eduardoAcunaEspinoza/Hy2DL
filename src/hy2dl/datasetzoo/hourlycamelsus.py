# import necessary packages
from typing import Optional

import numpy as np
import pandas as pd

from hy2dl.datasetzoo.camelsus import CAMELS_US
from hy2dl.utils.config import Config


class Hourly_CAMELS_US(CAMELS_US):
    """Class to process hourly data in similar format as the CAMELS US dataset.

    This class process hourly data stored in the same format as CAMELS US [1]_ [2]_. It also allows to load daily
    information from the CAMELS_US dataset and upsample it to hourly. Moreover, we can read the static attributes
    using the function defined in the CAMELS_US class.

    This class was adapted from NeuralHydrology  [3]_.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    time_period : {'training', 'validation', 'testing'}
        Defines the period for which the data will be loaded.
    gauge_id : Optional[str | list[str]], default=None
        Id of gauge(s) to be loaded.

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
        cfg: Config,
        time_period: str,
        gauge_id: Optional[str | list[str]] = None,
    ):
        # Run the __init__ method of CAMELS_US class
        super(Hourly_CAMELS_US, self).__init__(cfg=cfg, time_period=time_period, gauge_id=gauge_id)

    def _read_data(self, gauge_id: str) -> pd.DataFrame:
        """Read a specific catchment timeseries into a dataframe.

        Parameters
        ----------
        gauge_id : str
            8-digit USGS identifier of the basin.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        dfs = []
        for forcing in self.cfg.forcings:
            if forcing[-7:] == "_hourly":
                df = self._load_hourly_data(gauge_id=gauge_id, forcing=forcing)
            else:
                # load daily CAMELS forcings and upsample to hourly
                df, _ = self._load_camelsus_data(gauge_id=gauge_id, forcing=forcing)
                df = df.resample("1h").ffill()
            if len(self.cfg.forcings) > 1:
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # Read discharges and add them to current dataframe
        df = df.join(self._load_hourly_discharge(gauge_id=gauge_id))

        return df

    def _load_hourly_data(self, gauge_id: str, forcing: str) -> pd.DataFrame:
        """Read a specific catchment forcing timeseries

        Parameters
        ----------
        gauge_id : str
            8-digit USGS identifier of the basin.
        forcing : str
            e.g. ndlas_hourly'

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        path_timeseries = self.cfg.path_data / "hourly" / f"{forcing}" / f"{gauge_id}_hourly_nldas.csv"
        # load time series
        df = pd.read_csv(path_timeseries, index_col=["date"], parse_dates=["date"])

        return df

    def _load_hourly_discharge(self, gauge_id: str) -> pd.DataFrame:
        """Read a specific catchment discharge timeseries

        Parameters
        ----------
        gauge_id : str
            8-digit USGS identifier of the basin.

        Returns
        -------
        df: pd.Series
            Time-index pandas.Series of the discharge values (mm/h)

        """
        # Create a path to read the data
        streamflow_path = self.cfg.path_data / "hourly/usgs_streamflow" / f"{gauge_id}-usgs-hourly.csv"

        # load time series
        df = pd.read_csv(streamflow_path, index_col=["date"], parse_dates=["date"])

        # Replace invalid discharge values by NaN
        df["QObs(mm/h)"] = df["QObs(mm/h)"].apply(lambda x: np.nan if x < 0 else x)

        return df
