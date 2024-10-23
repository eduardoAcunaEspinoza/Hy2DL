# import necessary packages
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from hy2dl.datasetzoo.basedataset import BaseDataset


class CAMELS_DE(BaseDataset):
    """Class to process data from the CAMELS Germany dataset by [1]_ .

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
        super(CAMELS_DE, self).__init__(
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

    def _read_attributes(self) -> pd.DataFrame:
        """Read the catchments` attributes

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` attributes

        """
        # files that contain the attributes
        path_attributes = Path(self.path_data)
        read_files = list(path_attributes.glob("*_attributes.csv"))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=",", header=0, dtype={"gauge_id": str})
            df.set_index("gauge_id", inplace=True)
            dfs.append(df)

        # Join all dataframes
        df_attributes = pd.concat(dfs, axis=1)

        # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ["float64", "int64"]:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)

        # Filter attributes and basins of interest
        df_attributes = df_attributes.loc[self.entities_ids, self.static_input]

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
        path_timeseries = Path(self.path_data) / "timeseries" / f"CAMELS_DE_hydromet_timeseries_{catch_id}.csv"
        # load time series
        df = pd.read_csv(path_timeseries, index_col="date", parse_dates=["date"])
        return df
