# import necessary packages
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.utils.config import Config


class Luxemburg(BaseDataset):
    """Class to process data from the Luxemburg dataset

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from Luxemburg dataset.


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



    """

    def __init__(
        self,
        cfg: Config,
        time_period: str,
        check_NaN: Optional[bool] = True,
        entities_ids: Optional[Union[str, List[str]]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(Luxemburg, self).__init__(
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
        path_attributes = self.cfg.path_data
        read_files = list(path_attributes.glob("*_attributes.csv"))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=",", header=0, dtype={"gauge_id": str}).set_index("gauge_id")
            dfs.append(df)

        # Join all dataframes
        df_attributes = pd.concat(dfs, axis=1)

        # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ["float64", "int64"]:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)

        # Filter attributes and basins of interest
        df_attributes = df_attributes.loc[self.entities_ids, self.cfg.static_input]

        return df_attributes

    def _read_data(self, catch_id: str) -> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        catch_id : str
            identifier of the basin.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        path_timeseries = self.cfg.path_data / "timeseries" / f"hydromet_timeseries_{catch_id}.csv"
        
        # load time series
        df = pd.read_csv(path_timeseries, index_col="date", parse_dates=["date"])

        # Check for problematic columns
        for column in df.columns:
            if df[column].dtype == "object":  # Likely mixed types or strings
                df[column] = pd.to_numeric(df[column], errors="coerce")  # Coerce non-numeric to NaN

        # calculate specific discharge [mm/h] -> Q[m3/s]*[3600s/1h]*[1000mm/1m]/[area in m2] = Q[mm/h]
        area = self._read_attributes().loc[catch_id, "area"] * 1000 * 1000  # area in m2
        df["discharge"] = df["discharge"] * 3600 * 1000 / area

        return df
