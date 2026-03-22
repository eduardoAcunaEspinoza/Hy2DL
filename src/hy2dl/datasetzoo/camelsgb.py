# import necessary packages
from typing import Optional

import pandas as pd

from collections import defaultdict
from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.utils.config import Config


class CAMELS_GB(BaseDataset):
    """Class to process data from the CAMELS GB dataset [1]_ .

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS GB.

    This class was adapted from NeuralHydrology [2]_

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
    .. [1] Coxon, G., Addor, N., Bloomfield, J. P., Freer, J., Fry, M., Hannaford, J., Howden, N. J. K., Lane, R.,
        Lewis, M., Robinson, E. L., Wagener, T., and Woods, R.: CAMELS-GB: Hydrometeorological time series and
        landscape attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data Discuss.,
        https://doi.org/10.5194/essd-2020-49, in review, 2020.
    .. [2] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    def __init__(
        self,
        cfg: Config,
        time_period: str,
        gauge_id: Optional[str | list[str]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(CAMELS_GB, self).__init__(cfg=cfg, time_period=time_period, gauge_id=gauge_id)
        # super(CAMELS_GB, self).__init__(cfg=cfg, time_period=time_period)

    def _read_attributes(self) -> pd.DataFrame:
        """Read the catchments` attributes

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` attributes

        """
        # files that contain the attributes
        path_attributes = self.cfg.path_data
        read_files = list(path_attributes.glob("*_attributes.csv"))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=",", header=0, dtype={"gauge_id": str}).set_index("gauge_id")
            dfs.append(df)

        # Join all dataframes
        df_attributes = pd.concat(dfs, axis=1)

        # if possible, try to convert object columns to real numbers
        for col in df_attributes.select_dtypes(include=['object']).columns:
            df_attributes[col] = pd.to_numeric(df_attributes[col], errors='coerce')
            
        # encoding loop
        categorical_cols = df_attributes.select_dtypes(exclude=['number']).columns
        for column in categorical_cols:
            df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)

        # Filter attributes and basins of interest
        df_attributes = df_attributes.loc[self.gauge_id, self.cfg.static_input]

        return df_attributes

    def _read_data(self, gauge_id: str) -> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        gauge_id : str
            identifier of the basin.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        path_timeseries = (
            self.cfg.path_data / "timeseries" / f"CAMELS_GB_hydromet_timeseries_{gauge_id}_19701001-20150930.csv"
        )
        # load time series
        df = pd.read_csv(path_timeseries, index_col="date", parse_dates=["date"], dtype= defaultdict(lambda: "float32", date=str))
        return df
