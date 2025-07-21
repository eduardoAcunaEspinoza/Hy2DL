# import necessary packages
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.utils.config import Config


class CAMELS_US(BaseDataset):
    """Class to process data from the CAMELS US dataset [1]_ [2]_.

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS-US.

    This class and its methods were adapted from Neural Hydrology [3]_.

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
        check_NaN: Optional[bool] = True,
        entities_ids: Optional[Union[str, List[str]]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(CAMELS_US, self).__init__(
            cfg=cfg, time_period=time_period, check_NaN=check_NaN, entities_ids=entities_ids
        )

    def _read_attributes(self) -> pd.DataFrame:
        """Read the catchments` attributes

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` attributes

        """
        # files that contain the attributes
        path_attributes = self.cfg.path_data / "camels_attributes_v2.0"
        read_files = list(path_attributes.glob("camels_*.txt"))

        # Read one by one the attributes files
        dfs = []
        for file in read_files:
            df_temp = pd.read_csv(file, sep=";", header=0, dtype={"gauge_id": str}).set_index("gauge_id")
            dfs.append(df_temp)

        # Concatenate all the dataframes into a single one
        df = pd.concat(dfs, axis=1)
        # convert huc column to double digit strings
        df["huc"] = df["huc_02"].apply(lambda x: str(x).zfill(2))
        df = df.drop("huc_02", axis=1)

        # Encode categorical attributes in case there are any
        for column in df.columns:
            if df[column].dtype not in ["float64", "int64"]:
                df[column], _ = pd.factorize(df[column], sort=True)

        # Filter attributes and basins of interest
        df = df.loc[self.entities_ids, self.cfg.static_input]

        return df

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
        # Read forcings
        dfs = []
        for forcing in self.cfg.forcings:  # forcings can be daymet, maurer or nldas
            df, area = self._load_camelsus_data(catch_id=catch_id, forcing=forcing)
            # rename columns in case there are multiple forcings
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            # Append to list
            dfs.append(df)

        df = pd.concat(dfs, axis=1)  # dataframe with all the dynamic forcings

        # Read discharges and add them to current dataframe
        df["QObs(mm/d)"] = self._load_camelsus_discharge(catch_id=catch_id, area=area)

        # replace invalid discharge values by NaNs
        df["QObs(mm/d)"] = df["QObs(mm/d)"].apply(lambda x: np.nan if x < 0 else x)

        return df

    def _load_camelsus_data(self, catch_id: str, forcing: str) -> Tuple[pd.DataFrame, int]:
        """Read a specific catchment forcing timeseries

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.
        forcing : str
            Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries
        area : int
            Catchment area (m2), specified in the header of the forcing file.

        """
        # Create a path to read the data
        forcing_path = self.cfg.path_data / "basin_mean_forcing" / forcing
        file_path = list(forcing_path.glob(f"**/{catch_id}_*_forcing_leap.txt"))
        file_path = file_path[0]
        # Read dataframe
        with open(file_path, "r") as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
            # load the dataframe from the rest of the stream
            df = pd.read_csv(fp, sep=r"\s+")
            df["date"] = pd.to_datetime(
                df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d"
            )

            df = df.set_index("date")

        return df, area

    def _load_camelsus_discharge(self, catch_id: str, area: int) -> pd.DataFrame:
        """Read a specific catchment discharge timeseries

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.
        area : int
            Catchment area (m2), used to normalize the discharge.

        Returns
        -------
        df: pd.Series
            Time-index pandas.Series of the discharge values (mm/day)

        """
        # Create a path to read the data
        streamflow_path = self.cfg.path_data / "usgs_streamflow"
        file_path = list(streamflow_path.glob(f"**/{catch_id}_streamflow_qc.txt"))
        file_path = file_path[0]

        col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
        df["date"] = pd.to_datetime(
            df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d"
        )
        df = df.set_index("date")

        # normalize discharge from cubic feet per second to mm per day
        df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

        return df.QObs
