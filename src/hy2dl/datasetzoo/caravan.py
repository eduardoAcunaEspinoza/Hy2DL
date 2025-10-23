# import necessary packages
from typing import Optional
import pandas as pd
from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.utils.config import Config


class CARAVAN(BaseDataset):
    """Class to process data from the Caravan dataset [1]_ .

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from Caravan.
    The code would also run with user created datasets which conform to the Caravan style convention.

    This class and its methods were adapted from Neural Hydrology [2]_.

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
    .. [1] 1. Kratzert, F. et al. Caravan - A global community dataset for large-sample hydrology. Sci. Data 10, 61
        (2023). https://doi.org/10.1038/s41597-023-01975-w
    .. [2] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    def __init__(
        self,
        cfg: Config,
        time_period: str,
        check_NaN: Optional[bool] = True,
        entities_ids: Optional[str | list[str]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(CARAVAN, self).__init__(
            cfg=cfg,
            time_period=time_period,
            check_NaN=check_NaN,
            entities_ids=entities_ids,
        )

    def _read_attributes(self) -> pd.DataFrame:
        """Read the catchments` attributes from Caravan

        Parameters
        ----------
        path_data : Path to the root directory of Caravan that has to include a sub-directory
                    called 'attributes' which contain the attributes of all sub-datasets in separate folders.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments attributes
        """
        # files that contain the attributes
        data_dir = self.cfg.path_data
        # Take care of the subset directories in Caravans
        subdataset_dirs = [d for d in (data_dir / "attributes").glob("*") if d.is_dir()]

        # Load all required attribute files.
        dfs = []

        for subdataset_dir in subdataset_dirs:  # Loop over each sub directory
            dfr_list = []
            for csv_file in subdataset_dir.glob("*.csv"):  # Loop over each csv file
                dfr_list.append(pd.read_csv(csv_file, index_col="gauge_id"))
            dfr = pd.concat(dfr_list, axis=1)
            dfs.append(dfr)

        # Merge all DataFrames along the basin index.
        df_attributes = pd.concat(dfs, axis=0)

        # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ["float64", "int64"]:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)

        # Filter attributes and basins of interest
        df_attributes = df_attributes.loc[self.entities_ids, self.cfg.static_input]

        return df_attributes

    def _read_data(self, catch_id: str) -> pd.DataFrame:
        """Loads the timeseries data of one basin from the Caravan dataset.

        Parameters
        ----------
        data_dir : Path
            Path to the root directory of Caravan that has to include a sub-directory called 'timeseries'. This
            sub-directory has to contain another sub-directory called 'csv'.
        basin : str
            The Caravan gauge id string in the form of {subdataset_name}_{gauge_id}.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        data_dir = self.cfg.path_data
        basin = catch_id

        # Get the subdataset name from the basin string.
        subdataset_name = basin.split("_")[0].lower()
        filepath = data_dir / "timeseries" / "csv" / subdataset_name / f"{basin}.csv"
        df = pd.read_csv(filepath, index_col="date", parse_dates=["date"])

        return df
