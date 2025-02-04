import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import warnings
from numba import njit, prange
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Class to read and process data.

    This class is inherited by other subclasses (e.g. CAMELS_US, CAMELS_GB, ...) to read and process the data. The
    class contains all the common operations that need to be done, independently of which database is being used.

    This class and its methods are based on Neural Hydrology [#]_ and adapted for our specific case.

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
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    # Function to initialize the data
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
        # Store the information
        self.dynamic_input = dynamic_input
        self.conceptual_input = conceptual_input if conceptual_input is not None else []
        self.target = target

        self.sequence_length = sequence_length
        self.predict_last_n = predict_last_n

        self.path_data = path_data
        self.path_additional_features = path_additional_features
        self.time_period = time_period

        # One can specifiy a txt file with single/multiple entities ids, or use directly the entity_id
        if path_entities:
            entities_ids = np.loadtxt(path_entities, dtype="str").tolist()
            self.entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids
        elif entity:
            self.entities_ids = [entity]

        # Check if we want to have overlapping blocks in training data when predicting_last_n>1
        self.unique_prediction_blocks = unique_prediction_blocks

        # Variables used with multiple frequencies (e.g. daily/hourly)
        self.custom_freq_processing = custom_freq_processing
        self.dynamic_embedding = dynamic_embedding

        # Initialize variables
        self.sequence_data = {}  # store information that will be used to run the model
        self.df_ts = {}  # store processed dataframes for all basins
        self.scaler = {}  # information to standardize the data
        self.basin_std = {}  # std of the target variable of each basin (can be used later in the loss function)
        self.valid_entities = []  # index of valid samples that will be used for training

        # Process the attributes
        self.static_input = static_input if static_input is not None else []
        if self.static_input:
            self.df_attributes = self._read_attributes()

        # Process additional features that will be included as inputs
        if path_additional_features:
            self.additional_features = self._load_additional_features()

        # Remove duplicates in case we use the same variable in different frequencies (e.g. daily and hourly)
        self.unique_dynamic_input = self._unique_dynamic_input()
        # If we are using multiple frequencies, and for each frequency different input variables, we need the index
        # of those variables, in self.unique_dynamic_input, to construct the batches.
        if self.custom_freq_processing and isinstance(self.dynamic_input, dict):
            self.dynamic_input_index = {}
            for key, variables in self.dynamic_input.items():
                self.dynamic_input_index[key] = [self.unique_dynamic_input.index(elem) for elem in variables]

        # In case we are using a hybrid model, we also include the unique variables from conceptual model.
        unique_input = list(dict.fromkeys(self.unique_dynamic_input + self.conceptual_input))

        # This loop goes through all the catchments. For each catchment in creates an entry in the dictionary
        # self.sequence_data, where we will store the information that will be sent to the lstm
        for id in self.entities_ids:
            # Load time series for specific catchment id
            df_ts = self._read_data(catch_id=id)
            # Add additional features (optional)
            if path_additional_features:
                df_ts = pd.concat([df_ts, self.additional_features[id]], axis=1)
                if "ablation_flag" in df_ts.columns:
                    additional_is_flag = ["ablation_flag"]
                else:
                    additional_is_flag = []
            else:
                additional_is_flag = []

            # Defines the start date considering the offset due to sequence length. We want that, if possible, the start
            # date is the first date of prediction.
            freq = pd.infer_freq(df_ts.index)
            start_date = self._parse_datetime(date_str=self.time_period[0], freq=freq)
            end_date = self._parse_datetime(date_str=self.time_period[1], freq=freq)
            warmup_start_date = start_date - (
                self.sequence_length - self.predict_last_n
            ) * pd.tseries.frequencies.to_offset(freq)

            # Filter dataframe for the period and variables of interest. In case we are doing forecasting, and the
            # target is also used as input in the hindcast period, then we remove the duplicate columns.
            df_ts = df_ts.loc[
                warmup_start_date:end_date, list(dict.fromkeys(unique_input + self.target + additional_is_flag))
            ]

            # Reindex the dataframe to assure continuos data between the start and end date of the time period. Missing
            # data will be filled with NaN, so this will be taken care of later by the valid_samples function.
            full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=freq)
            df_ts = df_ts.reindex(full_range)

            # We can decide to use unique_prediction_blocks if we are predicting_last_n>1 and we do not want to have
            # overlapping samples. This alternative speed up the code by reducing the number of samples one has to
            # evaluate to complete an epoch. This is particularly useful if one is using hourly prediction.
            if self.unique_prediction_blocks and self.predict_last_n > 1:
                self.block_id = np.arange(len(df_ts) // self.predict_last_n) * self.predict_last_n + (
                    self.predict_last_n - 1
                )
            else:
                self.block_id = np.arange(len(df_ts))

            # Checks for invalid samples due to NaN or insufficient sequence length
            flag = validate_samples(
                x=df_ts.loc[:, unique_input].values,
                y=df_ts.loc[:, self.target].values,
                ablation_flag=df_ts.loc[:, "ablation_flag"].values if len(additional_is_flag) != 0 else None,
                attributes=self.df_attributes.loc[id].values if self.static_input else None,
                seq_length=self.sequence_length,
                predict_last_n=self.predict_last_n,
                block_id=self.block_id,
                check_NaN=check_NaN,
            )
            # Create a list that contain the indexes (basin, time_index) of the valid samples
            valid_samples = np.argwhere(flag == 1)
            self.valid_entities.extend([(id, self.block_id[int(f[0])]) for f in valid_samples])

            # Store the processed information of the basin, in basin-indexed dictionaries.
            if valid_samples.size > 0:
                # Store processed dataframes
                self.df_ts[id] = df_ts

                # Create dictionary entry for the basin
                self.sequence_data[id] = {}

                # Store dynamic input
                self.sequence_data[id]["x_d"] = torch.tensor(
                    df_ts.loc[:, self.unique_dynamic_input].values, dtype=torch.float32
                )

                # Store target data
                self.sequence_data[id]["y_obs"] = torch.tensor(df_ts.loc[:, self.target].values, dtype=torch.float32)

                # Store conceptual input (hybrid model case)
                if self.conceptual_input:
                    self.sequence_data[id]["x_conceptual"] = torch.tensor(
                        df_ts.loc[:, self.conceptual_input].values, dtype=torch.float32
                    )
                # Store static input (e.g. catchment attributes)
                if self.static_input:
                    self.sequence_data[id]["x_s"] = torch.tensor(self.df_attributes.loc[id].values, dtype=torch.float32)

    def __len__(self):
        return len(self.valid_entities)

    def __getitem__(self, id):
        """Function used to construct the batches"""
        basin, i = self.valid_entities[id]
        sample = {}
        # Information about dynamic input -----------------------------------------------------------------
        if self.custom_freq_processing is None:  # Case in which we have only one frequency
            sample["x_d"] = self.sequence_data[basin]["x_d"][i - self.sequence_length + 1 : i + 1, :]
        else:  # Case in which we have multiple frequencies
            current_index = 0  # index to keep track of the current position in the x_d tensor
            for index, (key, freq_info) in enumerate(self.custom_freq_processing.items()):
                if isinstance(self.dynamic_input, list):  # Case we have the same variables for all frequencies
                    x_lstm = self.sequence_data[basin]["x_d"][i - self.sequence_length + 1 : i + 1, :]
                elif isinstance(self.dynamic_input, dict):  # Case we have different variables for each frequency
                    x_lstm = self.sequence_data[basin]["x_d"][
                        i - self.sequence_length + 1 : i + 1, self.dynamic_input_index[key]
                    ]
                # Select timesteps of interest
                x_lstm = x_lstm[current_index : current_index + freq_info["n_steps"] * freq_info["freq_factor"], :]
                # Process values using the frequency factor
                x_lstm = x_lstm.reshape(freq_info["n_steps"], freq_info["freq_factor"], x_lstm.shape[1]).mean(dim=1)
                # Add a flag if we do not have custom embeddings for the different frequencies
                if not self.dynamic_embedding:
                    x_lstm = torch.cat([x_lstm, torch.ones(freq_info["n_steps"], 1) * index], dim=1)
                # Store the values
                sample["x_d_" + key] = x_lstm
                # Update index = start position for next frequency
                current_index += freq_info["n_steps"] * freq_info["freq_factor"]

        # Information about the static input ---------------------------------------------------------------
        if self.static_input:
            sample["x_s"] = self.sequence_data[basin]["x_s"]

        # Information about target variable -----------------------------------------------------------------
        sample["y_obs"] = self.sequence_data[basin]["y_obs"][i - self.predict_last_n + 1 : i + 1, :]

        # Information about the conceptual (just for hybrid model) -----------------------------------------
        if self.conceptual_input:
            sample["x_conceptual"] = self.sequence_data[basin]["x_conceptual"][i - self.sequence_length + 1 : i + 1, :]

        # Standard deviation of the discharge for the specific basin (use in the basin-averaged NSE loss function)
        if self.basin_std:
            sample["basin_std"] = self.basin_std[basin].repeat(sample["y_obs"].size(0)).unsqueeze(1)

        # Information about the basin and the dates to which predictions will be made. This facilitates
        # evaluating and ploting the results.
        sample["basin"] = np.array(basin, dtype=np.str_)
        sample["date"] = self.df_ts[basin].index[i - self.predict_last_n + 1 : i + 1].to_numpy()

        return sample

    def _read_attributes(self) -> pd.DataFrame:
        # This function is specific for each dataset
        raise NotImplementedError

    def _read_data(self) -> pd.DataFrame:
        # This function is specific for each dataset
        raise NotImplementedError

    def _load_additional_features(self) -> Dict[str, pd.DataFrame]:
        """Read pickle dictionary containing additional features.

        Returns
        -------
        additional_features : Dict[str, pd.DataFrame]
            Dictionary where each key is a basin and each value is a date-time indexed pandas DataFrame with the
            additional features

        """
        with open(self.path_additional_features, "rb") as file:
            additional_features = pickle.load(file)
        return additional_features

    def _parse_datetime(self, date_str: str, freq: str) -> pd.Timestamp:
        """Convert string date into pandas Timestamp object.

        Parameters
        ----------
        date_str : str
            string date
        freq : str
            frequency of the date (e.g. "D", "h")

        Returns
        -------
        pd.Timestamp
            pandas Timestamp object

        """
        if freq == "D":
            return pd.to_datetime(date_str, format="%Y-%m-%d")
        elif freq == "h":
            return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S")

    def _unique_dynamic_input(self) -> List[str]:
        """Remove duplicates in case we use the same variable in different frequencies.

        Returns
        -------
        List[str]
            List of unique dynamic input variables

        """
        if isinstance(self.dynamic_input, list):
            return self.dynamic_input
        elif isinstance(self.dynamic_input, dict):
            return list(dict.fromkeys([item for sublist in self.dynamic_input.values() for item in sublist]))

    def calculate_basin_std(self):
        """Fill the self.basin_std dictionary with the standard deviation of the target variables for each basin.

        This information is necessary if we use the basin-averaged NSE loss function during training [#]_.

        References
        ----------
        .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
            universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
            Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019

        """
        for id, data in self.sequence_data.items():
            self.basin_std[id] = torch.tensor(np.nanstd(data["y_obs"].numpy()), dtype=torch.float32)

    def calculate_global_statistics(self, path_save_scaler: Optional[str] = ""):
        """Calculate statistics of data.

        The function calculates the global mean and standard deviation of the dynamic inputs, target variables and
        static attributes, and store the in a dictionary. It will be used later to standardize the data used in the
        LSTM. This function should be called ONLY in the training period.

        Parameters
        ----------
        path_save_scalar : str
            path to save the scaler as a pickle file

        """
        # To avoid running out of memory, we calculate the statistics of the inputs one variable (column) at the time
        x_d_mean = torch.zeros(len(self.unique_dynamic_input), dtype=torch.float32)
        x_d_std = torch.zeros(len(self.unique_dynamic_input), dtype=torch.float32)
        for i, var in enumerate(self.unique_dynamic_input):
            global_x = np.hstack([df.loc[:, var].values for df in self.df_ts.values()])
            x_d_mean[i] = torch.tensor(np.nanmean(global_x, axis=0))
            x_d_std[i] = torch.tensor(np.nanstd(global_x, axis=0))
            del global_x

        # Check if the std is (almost) zero and adjust. The 1e-5 is a threshold to consider a std as zero (due to
        # numerical issues).
        zero_std_indices = (x_d_std <= 1e-5).nonzero(as_tuple=True)[0]
        if len(zero_std_indices) > 0:
            zero_std_vars = [self.unique_dynamic_input[idx] for idx in zero_std_indices.tolist()]
            warnings.warn(
                f"The standard deviation of the following variable(s) is zero: {zero_std_vars}. "
                f"The std of this variable(s) has been forced to 1 to avoid NaN issues during normalization.",
                stacklevel=2,
            )
        x_d_std[zero_std_indices] = 1.0

        # Save scaler of dynamic variables
        self.scaler["x_d_mean"] = x_d_mean
        self.scaler["x_d_std"] = x_d_std

        # Calculate statistics of the target
        global_y = np.vstack([df.loc[:, self.target].values for df in self.df_ts.values()])
        self.scaler["y_mean"] = torch.tensor(np.nanmean(global_y, axis=0), dtype=torch.float32)
        self.scaler["y_std"] = torch.tensor(np.nanstd(global_y, axis=0), dtype=torch.float32)
        del global_y

        if self.static_input:
            # Calculate mean
            self.scaler["x_s_mean"] = torch.tensor(self.df_attributes.mean().values, dtype=torch.float32)
            # Calculate std
            x_s_std = torch.tensor(self.df_attributes.std().values, dtype=torch.float32)
            # Values can be NaN if we only have one basin, or can be 0 if the value for a specific attribute is the
            # same in all catchments. In this case we replace it by 1, so it does not affect the standardization. The
            # 1e-5 is a threshold to consider a std as zero (due to numerical issues).
            nan_or_zero_std_indices = (torch.isnan(x_s_std) | (x_s_std <= 1e-5)).nonzero(as_tuple=True)[0]
            if len(nan_or_zero_std_indices) > 0:
                nan_or_zero_std_vars = [self.df_attributes.columns[idx] for idx in nan_or_zero_std_indices.tolist()]
                warnings.warn(
                    f"The standard deviation of the following attribute(s) is NaN or zero: {nan_or_zero_std_vars}. "
                    f"The std of this attribute(s) has been forced to 1 to avoid NaN issues during normalization.",
                    stacklevel=2,
                )
            x_s_std[nan_or_zero_std_indices] = 1.0
            self.scaler["x_s_std"] = x_s_std

        if path_save_scaler:  # save the results in a pickle file
            with open(path_save_scaler + "/scaler.pickle", "wb") as f:
                pickle.dump(self.scaler, f)

    def standardize_data(self, standardize_output: bool = True):
        """Standardize data

        Parameters
        ----------
        standardize_output : bool
            Boolean to define if the output should be standardize or not.

        """
        for basin in self.sequence_data.values():
            basin["x_d"] = (basin["x_d"] - self.scaler["x_d_mean"]) / self.scaler["x_d_std"]
            if self.static_input:
                basin["x_s"] = (basin["x_s"] - self.scaler["x_s_mean"]) / self.scaler["x_s_std"]
            if standardize_output:
                basin["y_obs"] = (basin["y_obs"] - self.scaler["y_mean"]) / self.scaler["y_std"]

    @staticmethod
    def collate_fn(
        samples: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Specify how to construct the batches based on the information recieved from __getitem()__"""
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature in ("basin", "date"):
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            else:
                # Everything else is a torch.Tensor
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)
        return batch


# @njit()
def validate_samples(
    x: np.ndarray,
    y: np.ndarray,
    ablation_flag: np.ndarray,
    attributes: np.ndarray,
    seq_length: int,
    predict_last_n: int,
    block_id: np.ndarray,
    check_NaN: bool,
) -> np.ndarray:
    """Checks for invalid samples due to NaN or insufficient sequence length.

    This function was taken from Neural Hydrology [#]_ and adapted for our specific case.

    Parameters
    ----------
    x : np.ndarray
        array of dynamic input;
    y : np.ndarray
        array of target values;
    ablation_flag: np.ndarray
        1D-array of 0/1 flags for deliberate exclusion of samples (0 for exclusion)
    attributes : np.ndarray
        array containing the static attributes;
    seq_length : int
        Sequence lengths; one entry per frequency
    predict_last_n: int
        Number of values that want to be used to calculate the loss
    block_id : np.ndarray
        Array with the indexes of the block-samples that will be checked.
    check_NaN : bool
        Boolean to specify if Nan should be checked or not

    Returns
    -------
    flag:np.ndarray
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.

    References
    ----------
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    # Number of samples we will check. It can be less than the total number of samples if we are using
    # unique_prediction_blocks and predict_last_n>1
    n_samples = len(block_id)

    # Initialize vector to store the flag. 1 means valid sample for training
    flag = np.ones(n_samples)

    for i in prange(n_samples):  # iterate through all samples
        # id of the block we are considering
        block_index = block_id[i]

        # too early, not enough information
        if block_index < seq_length - 1:
            flag[i] = 0
            continue

        if check_NaN:
            # any NaN in the dynamic inputs makes the sample invalid
            x_sample = x[block_index - seq_length + 1 : block_index + 1, :]
            if np.any(np.isnan(x_sample)):
                flag[i] = 0
                continue

        if check_NaN:
            # all-NaN in the targets makes the sample invalid
            y_sample = y[block_index - predict_last_n + 1 : block_index + 1]
            if np.all(np.isnan(y_sample)):
                flag[i] = 0
                continue

        # any NaN in the static features makes the sample invalid
        if attributes is not None and check_NaN:
            if np.any(np.isnan(attributes)):
                flag[i] = 0
                continue

        if ablation_flag is not None:
            if ablation_flag[i] == 0 or np.isnan(ablation_flag[i]):
                flag[i] = 0

    return flag
