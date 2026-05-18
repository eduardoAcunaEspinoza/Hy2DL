# import necessary packages
from typing import Optional

import pandas as pd

from hy2dl.datasetzoo.camelsde import CAMELS_DE
from hy2dl.utils.config import Config


class Hourly_CAMELS_DE(CAMELS_DE):
    """
     Class to process hourly data in similar format as the CAMELS DE dataset.


    Parameters
    ----------
     cfg : Config
         Configuration file.
     time_period : {'training', 'validation', 'testing'}
         Defines the period for which the data will be loaded..
    gauge_id : Optional[str | list[str]], default=None
        Id of gauge(s) to be loaded.

    """

    def __init__(
        self,
        cfg: Config,
        time_period: str,
        gauge_id: Optional[str | list[str]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(Hourly_CAMELS_DE, self).__init__(cfg=cfg, time_period=time_period, gauge_id=gauge_id)

    def _read_data(self, gauge_id: str) -> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        gauge_id : str
            identifier of the basin.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        # Read hourly data
        path_timeseries = self.cfg.path_data / "timeseries" / f"CAMELS_DE_1h_hydromet_timeseries_{gauge_id}.csv"

        # load time series
        df = pd.read_csv(path_timeseries, index_col="date", parse_dates=["date"])

        return df

    def _fill_precipitation_gaps(self, df):
        """
        CAMELS-DE-1h already contains gap-filled precipitation data, instead of using the entirely gap-filled data, we could
        also fill the gaps with different strategies, e.g. based on gap length.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries with the gaps filled
        """
        raise NotImplementedError(
            "This method is not implemented yet. Would be possible to fill gaps with different strategies, e.g. based on gap length."
        )
