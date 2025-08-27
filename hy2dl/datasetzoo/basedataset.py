import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
from tqdm import tqdm

from hy2dl.utils.config import Config


class BaseDataset(Dataset):
    """Class to read and process data.

    This class is inherited by other subclasses (e.g. CAMELS_US, CAMELS_GB, ...) to read and process the data. The
    class contains all the common operations that need to be done, independently of which database is being used.

    This class and some of its methods are based on Neural Hydrology [#]_ and adapted for our specific case.

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
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    # Function to initialize the data
    def __init__(
        self,
        cfg: Config,
        time_period: str,
        check_NaN: Optional[bool] = True,
        entities_ids: Optional[str | list[str]] = None,
    ):
        # Store configuration file
        self.cfg = cfg

        # Define time period type
        allowed_periods = {"training", "validation", "testing"}
        if time_period not in allowed_periods:
            raise ValueError(f"`time_period` must be one of: {allowed_periods}, but got '{time_period}'.")

        self.time_period = getattr(self.cfg, f"{time_period}_period")

        # Read entities_ids from variable
        if entities_ids:
            self.entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids
        # Read entities_ids from configuration file
        elif hasattr(self.cfg, f"path_entities_{time_period}"):
            path_entities = getattr(self.cfg, f"path_entities_{time_period}")
            entities_ids = np.loadtxt(path_entities, dtype="str").tolist()
            self.entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids
        else:
            raise ValueError(
                f"No entities_ids found. Provide the `entities_ids` variable directly or define in the configuration"
                f"file either `path_entities` or `path_entities_{time_period}`"
            )

        # Dictionaries to store the information used by the model. The dictionaries are basin-indexed.
        self.x_d = {}  # dynamic input going into the lstm
        self.y_obs = {}  # target variable
        if self.cfg.forecast_input:
            self.x_fc = {}
        if self.cfg.static_input:
            self.x_s = {}  # static input (e.g. catchment attributes)
        if self.cfg.dynamic_input_conceptual_model:
            self.x_d_conceptual = {}  # conceptual input (case of hybrid models)

        # Dictionary to store additional information
        self.df_ts = {}  # processed dataframe for each basin.
        self.scaler = {}  # information to standardize the data
        self.basin_std = {}  # std of the target variable of each basin (can be used later in the loss function)

        # List to store the valid entities (basin, time_index) that will be used for training.
        self.valid_entities = []
        basins_without_samples = []

        # --------------------------------------------------------------------------
        # Process static attributes
        if self.cfg.static_input:
            self.df_attributes = self._read_attributes()

        # Process additional features that can be included as inputs
        if self.cfg.path_additional_features:
            self.additional_features = self._load_additional_features()

        # Retrieve unique dynamic input names
        self.unique_dynamic_input = BaseDataset.unique_values(x=self.cfg.dynamic_input)

        # Model input for hindcast period (historical period)
        self.hindcast_input = list(
            dict.fromkeys(
                self.unique_dynamic_input + BaseDataset.unique_values(x=self.cfg.dynamic_input_conceptual_model)
            )
        )

        # Retrive unique forecast input names
        self.unique_forecast_input = BaseDataset.unique_values(x=self.cfg.forecast_input)

        # Concatenate and extract the unique variables per frequency if applicable. This is useful
        # if we have different groups of variables for each frequency
        if self.cfg.custom_seq_processing is not None and isinstance(self.cfg.dynamic_input, dict):
            self.unique_input_per_freq = {
                k: BaseDataset.unique_values(self.cfg.dynamic_input[k]) for k in self.cfg.custom_seq_processing
            }

        # This loop goes one by one through all the entities. For each entity it creates an entry in the different
        # dictionaries. We define a progress bar if self.entitites_ids contains more than one entity.
        iterator = (
            tqdm(self.entities_ids, desc="Processing entities", unit="entity", ascii=True)
            if len(self.entities_ids) > 1
            else self.entities_ids
        )
        for id in iterator:
            # Load time series for specific catchment id
            df_ts = self._read_data(catch_id=id)

            additional_flag = []
            if self.cfg.path_additional_features:
                # Add additional features (optional)
                df_ts = pd.concat([df_ts, self.additional_features[id]], axis=1)
                # We can add a flag using additional features, that indicate which samples should be excluded from
                # training. For this we need a date-indexed pandas DataFrame with a column named "ablation_flag",
                # containing 0/1 flags (0 for exclusion).
                if "ablation_flag" in df_ts.columns:
                    additional_flag.append("ablation_flag")

            # In case we need to add lagged features
            if isinstance(self.cfg.lagged_features, dict):
                df_ts = self._add_lagged_features(df=df_ts)

            # Defines the start date considering the offset due to sequence length. We want that, if possible, the start
            # date is the first date of prediction.
            freq = pd.infer_freq(df_ts.index)
            start_date = self._parse_datetime(date_str=self.time_period[0], freq=freq)
            end_date = self._parse_datetime(date_str=self.time_period[1], freq=freq)
            warmup_start_date = start_date - (
                self.cfg.seq_length_hindcast + self.cfg.seq_length_forecast - self.cfg.predict_last_n
            ) * pd.tseries.frequencies.to_offset(freq)

            # Filter dataframe for the period and variables of interest
            df_ts = df_ts.loc[
                warmup_start_date:end_date,
                list(
                    dict.fromkeys(self.hindcast_input + self.unique_forecast_input + self.cfg.target + additional_flag)
                ),
            ]

            # Reindex the dataframe to assure continuos data between the start and end date of the time period. Missing
            # data will be filled with NaN, so this will be taken care of later by the valid_samples function.
            full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=freq)
            df_ts = df_ts.reindex(full_range)

            # Checks for invalid samples due to NaN or insufficient sequence length
            flag = self._validate_samples(df_ts=df_ts, df_attributes=self.df_attributes.loc[id], check_NaN=check_NaN)
            # Index of valid samples
            valid_samples = np.where(flag)[0]

            # When working seq-seq, if we want non-overlapping blocks, we calculate their respective starting indices.
            if self.cfg.unique_prediction_blocks:
                block_id = np.arange(len(df_ts) // self.cfg.predict_last_n) * self.cfg.predict_last_n + (
                    self.cfg.predict_last_n - 1
                )
                valid_samples = block_id[flag[block_id]]

            # Create a list that contain the indexes (basin, time_index) of the valid samples
            self.valid_entities.extend([(id, int(f)) for f in valid_samples])

            # Store the processed information of the basin, in basin-indexed dictionaries.
            if valid_samples.size > 0:
                # Store processed dataframes
                self.df_ts[id] = df_ts

                # Store dynamic input as nested dictionary. First indexed by basin and then by variable name.
                self.x_d[id] = {
                    col: torch.tensor(df_ts[col].values, dtype=torch.float32) for col in self.unique_dynamic_input
                }

                # Store target data as dictionary indexed by basin.
                self.y_obs[id] = torch.tensor(df_ts[self.cfg.target].values, dtype=torch.float32)

                # Store forecast input as nested dictionary. First indexed by basin and then by variable name.
                if self.cfg.forecast_input:
                    self.x_fc[id] = {
                        col: torch.tensor(df_ts[col].values, dtype=torch.float32) for col in self.unique_forecast_input
                    }

                # Store static input (e.g. catchment attributes) as dictionary indexed by basin.
                if self.cfg.static_input:
                    self.x_s[id] = torch.tensor(self.df_attributes.loc[id].values, dtype=torch.float32)

                # Store conceptual input as nested dictionary. First indexed by basin and then by variable name
                if self.cfg.dynamic_input_conceptual_model:
                    self.x_d_conceptual[id] = {}
                    for k, v in self.cfg.dynamic_input_conceptual_model.items():
                        col = [v] if isinstance(v, str) else v
                        self.x_d_conceptual[id][k] = torch.tensor(
                            df_ts[col].mean(axis=1, skipna=True).values,
                            dtype=torch.float32,
                        )

            else:  # Basins without valid samples
                basins_without_samples.append(id)

        # Print information of basins without valid samples
        if len(basins_without_samples) > 0:
            cfg.logger.info(f"Basins without valid samples in period of interest: {basins_without_samples}")

    def __len__(self):
        return len(self.valid_entities)

    def __getitem__(self, id) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        """Function used to construct the elements of the batches"""
        basin, i = self.valid_entities[id]
        sample = {}
        # --------------------------
        # Input in hindcast period
        # --------------------------
        # If we do not have custom processing (process the whole sequence length the same way)
        if self.cfg.custom_seq_processing is None:
            # Dynamic input
            sample["x_d"] = {k: v[i - self.cfg.seq_length_hindcast + 1 : i + 1] for k, v in self.x_d[basin].items()}

        # If we have custom processing along the hindcast sequence length (e.g. multiple temporal frequencies)
        else:
            current_index = 0  # index to keep track of the current position in the x_d tensor
            # Iterate through each part of the custom processing
            for subset_name, subset_info in self.cfg.custom_seq_processing.items():
                sample["x_d_" + subset_name] = {}

                # Retrieve the variables of interest for the current frequency
                var_of_interest = (
                    self.unique_input_per_freq[subset_name]
                    if isinstance(self.cfg.dynamic_input, dict)
                    else self.cfg.dynamic_input
                )

                # Iterate through each variable in the dynamic input
                for k in var_of_interest:
                    x_lstm = self.x_d[basin][k][i - self.cfg.seq_length_hindcast + 1 : i + 1]
                    # Select timesteps of interest
                    x_lstm = x_lstm[current_index : current_index + subset_info["n_steps"] * subset_info["freq_factor"]]
                    # Process values using the frequency factor
                    x_lstm = x_lstm.reshape(subset_info["n_steps"], subset_info["freq_factor"]).mean(dim=1)
                    # Store processed sequence
                    sample["x_d_" + subset_name][k] = x_lstm

                # Update start position for next part of the sequence
                current_index += subset_info["n_steps"] * subset_info["freq_factor"]
        # --------------------------
        # Input in forecast period
        # --------------------------
        if self.cfg.forecast_input:
            sample["x_d_fc"] = {k: v[i + 1 : i + 1 + self.cfg.seq_length_forecast] for k, v in self.x_fc[basin].items()}

            # Forecast metadata
            sample["date_issue_fc"] = self.df_ts[basin].index[i].to_numpy()
            sample["persistent_q"] = self.y_obs[basin][i, :]  # last available discharge (for metric calculation)
        # --------------------------
        # Information about the static input
        # --------------------------
        if self.cfg.static_input:
            sample["x_s"] = self.x_s[basin]
        # --------------------------
        # Information about target variable
        # --------------------------
        sample["y_obs"] = self.y_obs[basin][
            i + self.cfg.seq_length_forecast + 1 - self.cfg.predict_last_n : i + self.cfg.seq_length_forecast + 1,
            :,
        ]
        # --------------------------
        # Information about the conceptual (hybrid model)
        # --------------------------
        if self.cfg.dynamic_input_conceptual_model:
            sample["x_d_conceptual"] = {
                k: v[i - self.cfg.seq_length_hindcast + 1 : i + 1] for k, v in self.x_d_conceptual[basin].items()
            }

        # --------------------------
        # Additional data
        # --------------------------
        # use in the basin-averaged NSE loss function)
        if self.basin_std:
            sample["std_basin"] = self.basin_std[basin].repeat(sample["y_obs"].size(0)).unsqueeze(1)
        # Information about the basin and the dates to which predictions will be made. This facilitates evaluating and
        # ploting the results.
        sample["basin"] = np.array(basin, dtype=np.str_)
        sample["date"] = (
            self.df_ts[basin]
            .index[
                i + self.cfg.seq_length_forecast + 1 - self.cfg.predict_last_n : i + self.cfg.seq_length_forecast + 1
            ]
            .to_numpy()
        )

        return sample

    def calculate_basin_std(self):
        """Fill the self.basin_std dictionary with the standard deviation of the target variables for each basin.

        This information is necessary if we use the basin-averaged NSE loss function during training [#]_.

        References
        ----------
        .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
            universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
            Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019

        """
        for k, v in self.y_obs.items():
            self.basin_std[k] = torch.tensor(np.nanstd(v.numpy()), dtype=torch.float32)

    def calculate_global_statistics(self, save_scaler: bool = False):
        """Calculate statistics of data.

        The function calculates the global mean and standard deviation of the dynamic inputs, target variables and
        static attributes, and store them in a dictionary. It will be used later to standardize the data. This function
        should ONLY be called in the training period.

        Parameters
        ----------
        save_scalar : bool
            If True, the scaler will be saved in a pickle file in the path defined by `path_save_results` in the config
            file

        """
        # Dynamic variables in hindcast period
        x_d_mean = {}
        x_d_std = {}
        for k in self.unique_dynamic_input:
            global_x = np.hstack([df[k].values for df in self.df_ts.values()])
            x_d_mean[k] = torch.tensor(np.nanmean(global_x, axis=0), dtype=torch.float32)
            x_d_std[k] = torch.tensor(np.nanstd(global_x, axis=0), dtype=torch.float32)
            del global_x

        self.scaler["x_d_mean"] = x_d_mean
        self.scaler["x_d_std"] = self._check_std(x_d_std)

        # Target variables
        global_y = np.vstack([df[self.cfg.target].values for df in self.df_ts.values()])
        self.scaler["y_mean"] = torch.tensor(np.nanmean(global_y, axis=0), dtype=torch.float32)
        self.scaler["y_std"] = torch.tensor(np.nanstd(global_y, axis=0), dtype=torch.float32)
        del global_y

        # Dynamic variables in forecast period
        if self.cfg.forecast_input:
            x_fc_mean = {}
            x_fc_std = {}
            for k in self.unique_forecast_input:
                global_x = np.hstack([df[k].values for df in self.df_ts.values()])
                x_fc_mean[k] = torch.tensor(np.nanmean(global_x, axis=0), dtype=torch.float32)
                x_fc_std[k] = torch.tensor(np.nanstd(global_x, axis=0), dtype=torch.float32)
                del global_x

            self.scaler["x_fc_mean"] = x_fc_mean
            self.scaler["x_fc_std"] = self._check_std(x_fc_std)

        # Static attributes
        if self.cfg.static_input:
            # Calculate mean
            self.scaler["x_s_mean"] = torch.tensor(self.df_attributes.mean().values, dtype=torch.float32)
            # Calculate std
            x_s_std = dict(
                zip(
                    self.df_attributes.columns,
                    torch.tensor(self.df_attributes.std().values, dtype=torch.float32),
                )
            )

            self.scaler["x_s_std"] = torch.tensor(list(self._check_std(x_s_std).values()), dtype=torch.float32)

        if save_scaler:  # save the results in a pickle file
            with open(self.cfg.path_save_folder / "scaler.pickle", "wb") as f:
                pickle.dump(self.scaler, f)

    def standardize_data(self, standardize_output: bool = True):
        """Standardize data, basin by basin.

        Parameters
        ----------
        standardize_output : bool
            Boolean to define if the output should be standardize or not.

        """
        for basin in self.x_d.keys():
            # Dynamic input
            for k, v in self.x_d[basin].items():
                self.x_d[basin][k] = (v - self.scaler["x_d_mean"][k]) / self.scaler["x_d_std"][k]

            # Forecast input
            if self.cfg.forecast_input:
                for k, v in self.x_fc[basin].items():
                    self.x_fc[basin][k] = (v - self.scaler["x_fc_mean"][k]) / self.scaler["x_fc_std"][k]

            # Static input
            if self.cfg.static_input:
                self.x_s[basin] = (self.x_s[basin] - self.scaler["x_s_mean"]) / self.scaler["x_s_std"]

            # Output
            if standardize_output:
                self.y_obs[basin] = (self.y_obs[basin] - self.scaler["y_mean"]) / self.scaler["y_std"]

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged input features to dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with lagged input features added.

        """
        for feature, shift in self.cfg.lagged_features.items():
            if isinstance(shift, list):  # If we have a list and we want to shift a variable multiple times
                for s in set(shift):  # only consider unique values
                    df[f"{feature}_shift{s}"] = df[feature].shift(periods=s)
            elif isinstance(shift, int):
                df[f"{feature}_shift{shift}"] = df[feature].shift(periods=shift)
            else:
                raise ValueError("The value of the 'lagged_features' arg must be either an int or a list of ints")

        return df

    def _check_std(self, std: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Check if the standard deviation is (almost) zero and adjust.

        The 1e-5 is a threshold to consider a std as zero (due to numerical issues).

        Parameters
        ----------
        std : dict[str, torch.Tensor]
            Dictionary with the standard deviation of the variables

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with the adjusted standard deviation

        """
        # Check if the std is (almost) zero and adjust. The 1e-5 is a threshold to consider a std as zero (due to
        # numerical issues).
        for k, v in std.items():
            if v <= 1e-5 or torch.isnan(v):
                std[k] = torch.tensor(1.0, dtype=torch.float32)
                warnings.warn(
                    f"The standard deviation of {k} is NaN or zero. "
                    f"The std has been forced to 1 to avoid NaN issues during normalization.",
                    stacklevel=2,
                )
        return std

    def _load_additional_features(self) -> dict[str, pd.DataFrame]:
        """Read pickle dictionary containing additional features.

        Returns
        -------
        additional_features : dict[str, pd.DataFrame]
            Dictionary where each key is a basin and each value is a date-time indexed pandas DataFrame with the
            additional features

        """
        with open(self.cfg.path_additional_features, "rb") as file:
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

    def _read_attributes(self) -> pd.DataFrame:
        # This function is specific for each dataset
        raise NotImplementedError

    def _read_data(self) -> pd.DataFrame:
        # This function is specific for each dataset
        raise NotImplementedError

    def _validate_samples(self, df_ts: pd.DataFrame, df_attributes: pd.DataFrame, check_NaN: bool) -> np.ndarray:
        """Checks for invalid samples due to NaN or insufficient sequence length.

        Parameters
        ----------
        df_ts : pd.DataFrame
            DataFrame with time series of inputs and targets;
        df_attributes : pd.DataFrame
            DataFrame of static attributes;
        check_NaN : bool
            Boolean to specify if Nan should be checked or not

        Returns
        -------
        flag:np.ndarray
            Array has a value of 1 for valid samples and a value of 0 for invalid samples.
        """
        # Initialize binary flags
        flag = np.ones(len(df_ts), dtype=bool)

        # Too early (not enough information)
        flag &= np.arange(len(df_ts)) >= (self.cfg.seq_length_hindcast - 1)

        # Too late, not enough information to get a full forecast
        last_forecast = len(df_ts) - self.cfg.seq_length_forecast
        flag &= np.arange(len(df_ts)) < last_forecast

        if check_NaN:
            # -------------------------
            # Check attributes: any NaN in the static features makes all the samples invalid
            # -------------------------
            if df_attributes is not None:
                if df_attributes.isna().values.any():
                    flag[:] = False
                    return flag

            # -------------------------
            # Hindcast NaN check
            # -------------------------
            # Case 1: If we use the same variables along the sequence length, and only one group of variables,
            # any NaN makes the sample invalid.
            # Examples:
            # - We have a single group of variables and we work with single frequency data
            # - We have multi-frequency approaches but all frequencies use the same single group of variables.
            if isinstance(self.cfg.dynamic_input, list):
                # values of interest
                x = df_ts[self.hindcast_input].values
                x_slide_view = sliding_window_view(x, (self.cfg.seq_length_hindcast, x.shape[1]))
                mask = ~np.any(np.isnan(x_slide_view), axis=(2, 3)).flatten()
                flag[self.cfg.seq_length_hindcast - 1 :] &= mask

            # Case 2: If we have multiple groups of variables, but use the same groups along the whole sequence
            # length, all the groups need to have NaN  elements in the same point to make the sample invalid.
            # Example:
            # - We have multiple group of variables but have single frequency data
            elif isinstance(self.cfg.dynamic_input, dict) and self.cfg.custom_seq_processing is None:
                group_masks = []
                # Check for each group
                for v in self.cfg.dynamic_input.values():
                    x = df_ts[v].values
                    x_slide_view = sliding_window_view(x, (self.cfg.seq_length_hindcast, x.shape[1]))
                    group_masks.append(~np.any(np.isnan(x_slide_view), axis=(2, 3)).flatten())
                # I get an invalid sample if all groups are invalid
                mask = np.any(group_masks, axis=0)
                flag[self.cfg.seq_length_hindcast - 1 :] &= mask

            # Case 3: If we use the different variables (or group of variables) along the sequence length.
            # Example:
            # - We have multi-frequency approaches and the variables change along the sequence.
            elif isinstance(self.cfg.dynamic_input, dict) and isinstance(self.cfg.custom_seq_processing, dict):
                aux_index = 0  # start of sequence subset
                for k, v in self.cfg.custom_seq_processing.items():
                    # If we have single group of variables for each frequency
                    if isinstance(self.cfg.dynamic_input[k], list):
                        x = df_ts[self.cfg.dynamic_input[k]].values
                        x_slide_view = sliding_window_view(x, (self.cfg.seq_length_hindcast, x.shape[1]))
                        x_freq_subset = x_slide_view[:, :, aux_index : aux_index + (v["n_steps"] * v["freq_factor"])]
                        mask = ~np.any(np.isnan(x_freq_subset), axis=(2, 3)).flatten()

                    # If we have multiple groups of variables for each frequency
                    elif isinstance(self.cfg.dynamic_input[k], dict):
                        group_masks = []
                        for group in self.cfg.dynamic_input[k].values():
                            x = df_ts[group].values
                            x_slide_view = sliding_window_view(x, (self.cfg.seq_length_hindcast, x.shape[1]))
                            x_freq_subset = x_slide_view[
                                :,
                                :,
                                aux_index : aux_index + (v["n_steps"] * v["freq_factor"]),
                            ]
                            group_masks.append(~np.any(np.isnan(x_freq_subset), axis=(2, 3)).flatten())
                        mask = np.any(group_masks, axis=0)

                    flag[self.cfg.seq_length_hindcast - 1 :] &= mask
                    aux_index += v["n_steps"] * v["freq_factor"]

            # -------------------------
            # Target NaN check: all-NaN in the targets makes the sample invalid
            # -------------------------
            i_start = self.cfg.seq_length_hindcast + self.cfg.seq_length_forecast - self.cfg.predict_last_n
            y = df_ts[self.cfg.target].values[i_start:]
            y_slide_view = sliding_window_view(y, (self.cfg.predict_last_n, y.shape[1]))
            mask = ~np.all(np.isnan(y_slide_view), axis=(2, 3)).flatten()
            flag[self.cfg.seq_length_hindcast - 1 : last_forecast] &= mask

            # -------------------------
            # Forecast NaN check: any-NaN in the x makes the sample invalid
            # -------------------------
            if self.cfg.forecast_input:
                # Case 1: If have only one group of variables, any NaN makes the sample invalid.
                if isinstance(self.cfg.forecast_input, list):
                    x = df_ts[self.cfg.forecast_input].values[self.cfg.seq_length_hindcast :]
                    x_slide_view = sliding_window_view(x, (self.cfg.seq_length_forecast, x.shape[1]))
                    mask = ~np.any(np.isnan(x_slide_view), axis=(2, 3)).flatten()

                # Case 2: If have multiple groups of variables, all the groups need to have NaN elements in
                # the same point to make the sample invalid.
                elif isinstance(self.cfg.forecast_input, dict):
                    group_masks = []
                    for v in self.cfg.forecast_input.values():
                        x = df_ts[v].values[self.cfg.seq_length_hindcast :]
                        x_slide_view = sliding_window_view(x, (self.cfg.seq_length_forecast, x.shape[1]))
                        group_masks.append(~np.any(np.isnan(x_slide_view), axis=(2, 3)).flatten())
                    mask = np.any(group_masks, axis=0)

                flag[self.cfg.seq_length_hindcast - 1 : last_forecast] &= mask

            # -------------------------
            # Ablation_flag check: If I want to exclude certain points
            # -------------------------
            if "ablation_flag" in df_ts.columns:
                ablation_flag = df_ts["ablation_flag"].values
                flag &= (ablation_flag != 0) & (~np.isnan(ablation_flag))

        return flag

    @staticmethod
    def collate_fn(
        samples: list[dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        """Collate a list of samples into a single batch.

        This function is used by the DataLoader to combine a list of individual samples
        (as returned by `__getitem__`) into a batch. Each sample is a dictionary containing
        tensors, NumPy arrays, or nested dictionaries of tensors.

        The function stacks or concatenates corresponding fields across the list of samples
        so that each resulting field in the batch has a shape starting with (batch_size, ...).

        This function was taken from Neural Hydrology [#]_ and adapted for our specific case.

        Parameters
        ----------
            samples (List[dict]): A list of sample dictionaries, each returned by `__getitem__`.
                Each dictionary may contain:
                    - tensors
                    - numpy arrays
                    - Nested dictionaries of tensors

        Returns:
        ---------
            dict: A single dictionary with the same structure as the input samples,
                where values have been batched along the first dimension.

        References
        ----------
        .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
            research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022
        """
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith(("x_d")):
                # Dynamic variables are stored as dictionaries with feature names as keys.
                batch[feature] = {
                    k: torch.stack([sample[feature][k] for sample in samples], dim=0)
                    for k in samples[0][feature].keys()
                }
            elif feature.startswith(("basin", "date")):
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            else:
                # Everything else is a torch.Tensor
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)
        return batch

    @staticmethod
    def flatten_dict_values(d: dict) -> list:
        """Flatten the values of a (nested) dictionary into a list."""
        flatten_v = []
        for v in d.values():
            if isinstance(v, dict):
                flatten_v.extend(BaseDataset.flatten_dict_values(v))
            elif isinstance(v, list):
                flatten_v.extend(v)
            else:
                flatten_v.append(v)
        return flatten_v

    @staticmethod
    def unique_values(x: list | dict[str, list | dict[str, list]] | None) -> list[str]:
        """Retrieve unique values

        Parameters
        ----------
        x : list | dict[str, list | dict[str, list]] | None
            Data to retrieve unique variables from.

        Returns
        -------
        List[str]
            List of unique values

        """
        if isinstance(x, list):
            return list(dict.fromkeys(x))
        elif isinstance(x, dict):
            return list(dict.fromkeys(BaseDataset.flatten_dict_values(x)))
        elif x is None:
            return []
