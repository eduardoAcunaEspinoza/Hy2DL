# import necessary packages
from typing import Optional

import pandas as pd

from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.utils.config import Config


class CAMELS_DE(BaseDataset):
    """
    Class to process data from [version 1.0.0 of the CAMELS Germany dataset]
    (https://doi.org/10.5281/zenodo.13837553) by [1]_, [2]_ .

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS DE.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    period : {'training', 'validation', 'testing'}
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
        entities_ids: Optional[str | list[str]] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(CAMELS_DE, self).__init__(
            cfg=cfg,
            time_period=time_period,
            check_NaN=check_NaN,
            entities_ids=entities_ids,
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
        df: pd.DataFrame
            Dataframe with the catchments` timeseries

        """
        path_timeseries = self.cfg.path_data / "timeseries" / f"CAMELS_DE_hydromet_timeseries_{catch_id}.csv"
        # load time series
        df = pd.read_csv(path_timeseries, index_col="date", parse_dates=["date"])
        return df
