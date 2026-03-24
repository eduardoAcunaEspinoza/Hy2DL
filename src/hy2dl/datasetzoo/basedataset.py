import datetime
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import dask
import numpy as np
import pandas as pd
import psutil
import torch
import xarray as xr
import xarray_tensorstore
import yaml
import zarr
from dask.distributed import Client, LocalCluster, as_completed
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.dask import TqdmCallback

from hy2dl.utils.config import Config


class BaseDataset(Dataset):
    """Class to read and process data.

    This class is inherited by other subclasses (e.g. CAMELS_US, CAMELS_GB, ...) to read and process the data. The
    class contains all the common operations that need to be done, independently of which database is being used.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    time_period : {'training', 'validation', 'testing'}
        Defines the period for which the data will be loaded.
    gauge_id : Optional[str | list[str]], default=None
        Id of gauge(s) to be loaded.

    """

    def __init__(
        self,
        cfg: Config,
        time_period: str,
        gauge_id: Optional[str | list[str]] = None,
    ):
        # Store configuration file
        self.cfg = cfg

        # Define time period type
        allowed_periods = {"training", "validation", "testing"}
        if time_period not in allowed_periods:
            raise ValueError(f"`time_period` must be one of: {allowed_periods}, but got '{time_period}'.")
        self.period = time_period
        self.time_period = getattr(self.cfg, f"{time_period}_period")

        ## ------------------------------------------------------
        ## Read gauge_id
        ## ------------------------------------------------------
        if gauge_id:  # from variable
            self.gauge_id = [gauge_id] if isinstance(gauge_id, str) else gauge_id
        elif hasattr(self.cfg, f"path_entities_{time_period}"):  # from configuration file
            path_entities = getattr(self.cfg, f"path_entities_{time_period}")
            gauge_id = np.loadtxt(path_entities, dtype="str").tolist()
            self.gauge_id = [gauge_id] if isinstance(gauge_id, str) else gauge_id
        else:
            raise ValueError(
                f"No gauge_id found. Provide the `gauge_id` variable directly or define in the configuration"
                f"file either `path_entities` or `path_entities_{time_period}`"
            )

        ## ------------------------------------------------------
        ## Extract variables of interest for the different groups
        ## ------------------------------------------------------
        self.dynamic_input = BaseDataset.unique_values(x=self.cfg.dynamic_input)  # dynamic input

        # If applicable, extract variables per frequency. Useful if we have different groups per frequency
        if self.cfg.custom_seq_processing is not None and isinstance(self.cfg.dynamic_input, dict):
            self.input_per_freq = {
                k: BaseDataset.unique_values(self.cfg.dynamic_input[k]) for k in self.cfg.custom_seq_processing
            }

        # input for hindcast period
        self.hindcast_input = list(
            dict.fromkeys(self.dynamic_input + BaseDataset.unique_values(x=self.cfg.dynamic_input_conceptual_model))
        )

        # input for pseudo-forecast and/or forecast period
        self.pseudo_forecast_input = BaseDataset.unique_values(x=self.cfg.pseudo_forecast_input)
        self.forecast_input = BaseDataset.unique_values(x=self.cfg.forecast_input)

        ### -----------------------------------------------
        ### Read sample dataset to get metadata of interest
        ### -----------------------------------------------
        df_sample = self._read_data(gauge_id=gauge_id[0])

        # Extract time metadata
        self.data_freq = pd.infer_freq(df_sample.index)
        self.data_step = pd.tseries.frequencies.to_offset(self.data_freq)

        self.start_date = self._parse_datetime(date_str=self.time_period[0], freq=self.data_freq)
        self.end_date = self._parse_datetime(date_str=self.time_period[1], freq=self.data_freq)
        self.warmup_start_date = (
            self.start_date
            - (self.cfg.seq_length_hindcast + self.cfg.seq_length_forecast - self.cfg.predict_last_n) * self.data_step
        )

        # Check if we have ablation_flag as additional features, and if so, we add it to the variables of interest
        additional_flag = []
        if self.cfg.path_additional_features:
            ds_af = xarray_tensorstore.open_zarr(self.cfg.path_additional_features)
            if "ablation_flag" in ds_af.feature:
                additional_flag.append("ablation_flag")

        # Collect all variables of interest in one list
        self.variables_of_interest = list(
            dict.fromkeys(self.hindcast_input + self.pseudo_forecast_input + self.cfg.target + additional_flag)
        )

        # Initialized variables
        self.dataset_in_ram = False
        self.fc_in_ram = False
        self.basin_std = None

    def __len__(self):
        return self.valid_samples.shape[0]

    def __getitem__(self, idx: int | list[int]):
        """Extract a sample from dataset.

        This function is NOT called by Dataloader to contruct the batches, instead we use __getitems__.

        Parameters
        ----------
        idx: int or list[int]
            Index(es) of the data-sample to extract. The index refers to the position in the `self.valid_samples` table,
            which contains the (gauge_id, date, source) combination(s), that can be used for training/evaluation.

        """
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, list):
            indices = idx
        else:
            raise ValueError(f"Invalid index type: {type(idx)}. Expected int or list of ints.")
        return self.__getitems__(indices)

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """Construct a data-sample that will be send to the model

        The function used vectorized logic to extract the required information to construct a data-sample. It supports
        the case where the data is being loaded directly from RAM, of lazily from disk.

        Parameters
        ----------
        idx: list[int]
            Indexes of the data-sample to extract. The indexes refers to the positions in the `self.valid_samples`
            table, which contains the (gauge_id, date, source) combinations, that can be used for training/evaluation.

        Returns
        -------
        sample: dict[str, Any]
            A dictionary containing the information that will be used to train the model. Elements of the dictionary
            include:
            - x_d: dict[str, torch.Tensor]
                Dynamic inputs in hindcast period. Keys are the variable names and the values are tensors of shape
                (B, L).In case of multiple frequencies, one dictionary per frequency is created.
            - y_obs: torch.Tensor, shape (B, N, T)
                Target variables.
            - x_s: Optional[torch.Tensor], shape (B, NS)
                Static inputs.
            - x_d_fc: Optional[dict[str, torch.Tensor]]
                Dynamic inputs in forecast period. Keys are the variable names and the values are tensors of shape
                (B, LF).
            - source_fc: Optional[np.ndarray], shape (B,)
                String array to differentiate the source of forecast signals: pseudoforecast ("obs") or forecast ("fc").
            - init_time_fc: Optional[np.ndarray], shape (B,)
                Time where the forecast is emitted.
            - persistent_q: Optional[torch.Tensor], shape (B, 1, T)
                Target at the time the forecast is emitted.
            - gauge_id: np.ndarray, shape (B,)
                ID of the gauge for each element of the batch.
            - date: np.ndarray, shape (B, L)
                Dates associated with the target.
            - std_basin: Optional[torch.Tensor], shape (B, T)
                Standard deviation of target variables for the training period. Necessary if one uses basin-averaged NSE
                as a loss function.

        Notes
        -----
        Shape abbreviations used:
        - B: batch size
        - L: sequence length hindcast period.
        - N: length of the target sequence, based on `predict_last_n` cofiguration argument
        - LF: length of forecast period
        - T: number of target variables
        - NS: number of static inputs

        """
        # If we are loading data lazily from disk, we do the lazy loading directly pytorch´s workers.
        if not self.dataset_in_ram and getattr(self, "ds_ts", None) is None:
            self._load_dataset(engine="tensorstore")

        if self.cfg.path_forecast_dataset is not None and not self.fc_in_ram and getattr(self, "ds_fc", None) is None:
            self._load_forecast_dataset(engine="tensorstore")

        sample = {}
        # Valid samples
        valid_sample = self.valid_samples[indices]
        id, date, source = valid_sample["gauge_id"], valid_sample["date"], valid_sample["source"]
        unique_sources = np.unique(source)

        # Indexes of the valid samples
        valid_idx_hc = self.valid_idx_hc[indices]
        id_idx_hc, date_idx_hc = valid_idx_hc["gauge_idx"], valid_idx_hc["date_idx"]

        valid_idx_fc = self.valid_idx_fc[indices]
        id_idx_fc, date_idx_fc = valid_idx_fc["gauge_idx"], valid_idx_fc["date_idx"]

        def _extract_xd_sequence(
            var_list: list[str],
            id_idx: np.ndarray,
            start_t_indices: np.ndarray,
            length: int,
            freq_factor: Optional[int] = None,
            as_dict: bool = True,
        ) -> dict[str, torch.Tensor] | torch.Tensor:
            """Helper function to extract a block (Batch, Sequence, Features) from dataset.

            Parameters
            ----------
            var_list: list[str]
                List of variable names to extract.
            id_idx: np.ndarray
                Indexes of the gauges to extract
            start: np.ndarray
                Starting index of each batch element
            length: int
                Length of the sequence.
            freq_factor: Optional[int], default=None
                If provided, the sequence wil averaged over the frequency factor
            as_dict: bool, default=True
                Whether to return the result as a dictionary (with variable names as keys) or as a tensor

            Returns
            -------
            dict[str, torch.Tensor] or torch.Tensor
                The extracted block, either as a dictionary or as a tensor
                - dict: features names as keys, and values being the corresponding tensors of shape (B, L)
                - tensor: a single tensor of shape (B, L, F)

            """
            var_indices = [self.feature_to_idx["obs"][var] for var in var_list]

            if self.dataset_in_ram:
                batch_gauge_idx = id_idx[:, None]
                batch_time_idx = start_t_indices[:, None] + np.arange(length)
                x_tensor = self.ds_ts[batch_gauge_idx, batch_time_idx][:, :, var_indices]
            else:
                batch_gauge_idx = xr.DataArray(id_idx, dims=["batch"])
                batch_time_idx = xr.DataArray(start_t_indices[:, None] + np.arange(length), dims=["batch", "time"])
                data_slice = (
                    self.ds_ts["data"]
                    .isel(gauge_id=batch_gauge_idx, date=batch_time_idx, feature=var_indices)
                    .compute()
                )
                # standardize data
                data_slice = (data_slice - self.scaler.sel(statistic="mean", feature=var_list).data) / self.scaler.sel(
                    statistic="std", feature=var_list
                ).data

                x_tensor = torch.from_numpy(data_slice.data).float()  # convert to tensor

            # Apply frequency reduction to the whole batch at once
            if freq_factor is not None:
                b, t, f = x_tensor.shape
                x_tensor = x_tensor.reshape(b, t // freq_factor, freq_factor, f).mean(dim=2)

            # Return as dictionary or tensor
            return dict(zip(var_list, x_tensor.unbind(dim=2), strict=True)) if as_dict else x_tensor

        def _extract_xfc_sequence(
            var_list: list[str],
            id_idx: np.ndarray,
            start_t_indices: np.ndarray,
        ) -> dict[str, torch.Tensor] | torch.Tensor:
            """Helper function to extract a block (Batch, Sequence, Features) from forecast dataset.

            Parameters
            ----------
            var_list: list[str]
                List of variable names to extract.
            id_idx: np.ndarray
                Indexes of the gauges to extract
            start: np.ndarray
                Indexes (dates) of the forecasts to extract

            Returns
            -------
            torch.Tensor
                The extracted block (B, L, F)

            """
            var_indices = [self.feature_to_idx["fc"][var] for var in var_list]

            if self.fc_in_ram:
                x_tensor = self.ds_fc[id_idx, start_t_indices][:, :, var_indices]
            else:
                batch_gauge_idx = xr.DataArray(id_idx, dims=["batch"])
                batch_time_idx = xr.DataArray(start_t_indices, dims=["batch"])

                data_slice = (
                    self.ds_fc["data"]
                    .isel(
                        gauge_id=batch_gauge_idx,
                        date=batch_time_idx,
                        feature=var_indices,
                    )
                    .compute()
                )

                data_slice = (
                    data_slice - self.scaler_fc.sel(statistic="mean", feature=var_list).data
                ) / self.scaler_fc.sel(statistic="std", feature=var_list).data

                x_tensor = torch.from_numpy(data_slice.data).float()  # convert to tensor

            return x_tensor

        # --------------------------
        # Hindcast period
        # --------------------------
        start_hindcasts = date_idx_hc - self.cfg.seq_length_hindcast + 1
        if self.cfg.custom_seq_processing is None:  # single frequency
            sample["x_d"] = _extract_xd_sequence(
                var_list=self.dynamic_input,
                id_idx=id_idx_hc,
                start_t_indices=start_hindcasts,
                length=self.cfg.seq_length_hindcast,
            )
        else:  # multi-frequency
            current_index = 0
            for subset_name, subset_info in self.cfg.custom_seq_processing.items():
                subset_length = subset_info["n_steps"] * subset_info["freq_factor"]
                subset_variables = (
                    self.input_per_freq[subset_name]
                    if isinstance(self.cfg.dynamic_input, dict)
                    else self.cfg.dynamic_input
                )
                sample["x_d_" + subset_name] = _extract_xd_sequence(
                    var_list=subset_variables,
                    id_idx=id_idx_hc,
                    start_t_indices=start_hindcasts + current_index,
                    length=subset_length,
                    freq_factor=subset_info["freq_factor"],
                )
                current_index += subset_length

        # --------------------------
        # Static input
        # --------------------------
        if self.cfg.static_input:
            sample["x_s"] = self.ds_attributes[id_idx_hc]

        # --------------------------
        # Target
        # --------------------------
        sample["y_obs"] = _extract_xd_sequence(
            var_list=self.cfg.target,
            id_idx=id_idx_hc,
            start_t_indices=date_idx_hc + 1 + self.cfg.seq_length_forecast - self.cfg.predict_last_n,
            length=self.cfg.predict_last_n,
            as_dict=False,
        )

        # --------------------------
        # Forecast period
        # --------------------------
        if self.cfg.pseudo_forecast_input or self.cfg.forecast_input:
            if self.cfg.merge_forecast_signal:  # If I need to combine the pseudo-forecast and forecast
                shared_vars = self.pseudo_forecast_input if self.pseudo_forecast_input else self.forecast_input
                x_tensor = torch.full(
                    size=(len(id_idx_fc), self.cfg.seq_length_forecast, len(shared_vars)),
                    fill_value=np.nan,
                    dtype=torch.float32,
                )
                if "obs" in unique_sources:
                    source_mask = source == "obs"
                    x_tensor[source_mask, :, :] = _extract_xd_sequence(
                        var_list=shared_vars,
                        id_idx=id_idx_fc[source_mask],
                        start_t_indices=date_idx_fc[source_mask] + 1,
                        length=self.cfg.seq_length_forecast,
                        as_dict=False,
                    )

                if "fc" in unique_sources:
                    source_mask = source == "fc"
                    x_tensor[source_mask, :, :] = _extract_xfc_sequence(
                        var_list=shared_vars, id_idx=id_idx_fc[source_mask], start_t_indices=date_idx_fc[source_mask]
                    )

                sample["x_d_fc"] = dict(zip(shared_vars, x_tensor.unbind(dim=2), strict=True))

            else:  # If I need to keep the pseudo-forecast and forecast as separate variables
                sample["x_d_fc"] = {}
                if "obs" in unique_sources:
                    x_tensor = torch.full(
                        size=(len(id_idx_fc), self.cfg.seq_length_forecast, len(self.pseudo_forecast_input)),
                        fill_value=np.nan,
                        dtype=torch.float32,
                    )
                    source_mask = source == "obs"
                    x_tensor[source_mask, :, :] = _extract_xd_sequence(
                        var_list=self.pseudo_forecast_input,
                        id_idx=id_idx_fc[source_mask],
                        start_t_indices=date_idx_fc[source_mask] + 1,
                        length=self.cfg.seq_length_forecast,
                        as_dict=False,
                    )
                    sample["x_d_fc"].update(dict(zip(self.pseudo_forecast_input, x_tensor.unbind(dim=2), strict=True)))

                if "fc" in unique_sources:
                    x_tensor = torch.full(
                        size=(len(id_idx_fc), self.cfg.seq_length_forecast, len(self.forecast_input)),
                        fill_value=np.nan,
                        dtype=torch.float32,
                    )
                    source_mask = source == "fc"
                    x_tensor[source_mask, :, :] = _extract_xfc_sequence(
                        var_list=self.forecast_input,
                        id_idx=id_idx_fc[source_mask],
                        start_t_indices=date_idx_fc[source_mask],
                    )
                    sample["x_d_fc"].update(dict(zip(self.forecast_input, x_tensor.unbind(dim=2), strict=True)))

            sample["source_fc"] = source
            sample["init_time_fc"] = date
            sample["persistent_q"] = _extract_xd_sequence(
                var_list=self.cfg.target, id_idx=id_idx_hc, start_t_indices=date_idx_hc, length=1, as_dict=False
            )

        # --------------------------
        # Additional metadata
        # --------------------------
        sample["gauge_id"] = id
        end_dates = date + np.timedelta64(self.cfg.seq_length_forecast, self.data_freq)
        offsets = np.arange(-self.cfg.predict_last_n + 1, 1) * np.timedelta64(1, self.data_freq)
        sample["date"] = end_dates[:, None] + offsets
        if torch.is_tensor(self.basin_std):
            sample["std_basin"] = self.basin_std[id_idx_hc]

        return sample

    def setup_dataset(
        self,
        check_nan=True,
        path_scaler: Optional[Path | str] = None,
        path_validate_samples: Optional[Path | str] = None,
    ):
        """Get data ready for training or evaluation.

        This is the function you should call to load and process the dataset, and get it ready for use. It process,
        validates, maps, standardizes, and optimizes the dataset that will be sent to the model.

        The setup follows these step:
        1. Load Data: Either processes the dataset from scratch or loads an existing pre-processed one. Processing the
        dataset is done in `_process_df` and includes
            - reading the raw data
            - selecting the time periods and variables of interest
            - adding additional and lagged features (if specified)
            - reindexing the data to have a continuous time index.
        2. Validate Samples: Look for valid samples or load a pre-computed list. Criteria for valid samples is defined
        in `_valid_samples_mask`.
        3. Map indexes: Map the valid samples to the corresponding indexes in the dataset. This is necessary for
        efficient data loading during training.
        3. Calculate statistics: Calculate data statistics that are used for standardization.
        4. Finalize: runs `_finalize_setup()` to optimize memory and data access speed.

        Parameters
        ----------
        check_nan : bool, default=True
            Check for nan values during validate_samples-
        path_scaler : Path or str, optional
            Path to saved `scaler.yml` file.
        path_validate_samples : Path or str, optional
            Path to a saved CSV file that lists all the valid samples.

        """
        processing_time = time.time()
        # Check if path to existing dataset was specified in config
        _load_existing_dataset = getattr(self.cfg, f"path_dataset_{self.period}") is not None
        # Check if we are in training mode
        _is_training = self.period == "training"
        if not _is_training and not path_scaler:
            raise ValueError(
                "When dataset for validation or testing is being created, argument `path_scaler` should be provided"
            )
        # --------------------------
        # Create or load dataset of observed data
        # --------------------------
        # Create dataset in RAM
        if not _load_existing_dataset and self.cfg.dataset_in_ram:
            self._create_dataset()
            self.dataset_in_ram = True
        else:
            # Create zarr dataset in disk and load it lazily
            if not _load_existing_dataset:
                self._create_zarr_dataset()
                self._load_dataset(path_dataset=self.path_dataset)
            else:  # Load existing zarr dataset lazily
                self._load_dataset()
            self.dataset_in_ram = False

        # static attributes
        self._process_attributes()

        # --------------------------
        # Load forecast dataset
        # --------------------------
        if self.cfg.path_forecast_dataset is not None:
            self._load_forecast_dataset()
            self.fc_in_ram = False

        # --------------------------
        # Validate samples
        # --------------------------
        if path_validate_samples is not None:
            self._load_valid_samples(path_scaler=self.path_scaler)
        else:
            self._validate_samples(check_nan=check_nan)

        # --------------------------
        # Scalers
        # --------------------------
        if path_scaler is not None:
            self._load_scaler(path_scaler=path_scaler)
        else:
            self._calculate_scaler()

        # --------------------------
        # Additional variables
        # --------------------------
        if self.period == "training" and self.cfg.loss == "nse_basin_averaged":
            self._calculate_basin_std()

        # --------------------------
        # Finalize dataset setup
        # --------------------------
        self._finalize_setup()

        self.cfg.logger.info(
            f"Time required to process the dataset: {datetime.timedelta(seconds=int(time.time() - processing_time))}"
        )

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

    def _calculate_basin_std(self):
        """Calculate standard deviation of the target variables for each basin.

        This information is necessary if we use the basin-averaged NSE loss function during training [#]_.

        References
        ----------
        .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
            universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
            Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            basin_std = self.ds_ts["data"].sel(feature=self.cfg.target).std(dim=["date"], skipna=True)
            self.basin_std = torch.tensor(basin_std.values, dtype=torch.float32)
            self.basin_std.share_memory_()

    def _calculate_scaler(self):
        """Calculate data statistics"""

        ### ------------------
        ### Dynamic variables
        ### ------------------
        # Calculate mean of variables using all the gauges that will be used for training (self.valid_gauges). In case
        # we get a NaN value for the mean, we replace it with 0.0, to avoid affecting the standardization.
        xd_mean = self.ds_ts.sel(gauge_id=self.valid_gauges).mean(dim=["gauge_id", "date"], skipna=True).fillna(0.0)
        # Calculate standard deviation of variables using all the gauges that will be used for training
        # (self.valid_gauges). In case we get a NaN or really small values, we replace it with 1.0, to avoid affecting
        # the standardization.
        xd_std = self.ds_ts.sel(gauge_id=self.valid_gauges).std(dim=["gauge_id", "date"], skipna=True)
        xd_std = xd_std.where((xd_std.notnull()) & (xd_std >= 1e-5), 1.0)
        # Combine into one and computes the values
        self.scaler = xr.concat([xd_mean, xd_std], dim="statistic")
        self.scaler = self.scaler.assign_coords(statistic=["mean", "std"])
        self.scaler = self.scaler.compute()

        ### ------------------
        ### Forecast variables
        ### ------------------
        if self.cfg.forecast_input:
            xfc_mean = (
                self.ds_fc.sel(gauge_id=self.valid_gauges)
                .mean(dim=["gauge_id", "date", "lead_time"], skipna=True)
                .fillna(0.0)
            )
            xfc_std = self.ds_fc.sel(gauge_id=self.valid_gauges).std(dim=["gauge_id", "date", "lead_time"], skipna=True)
            xfc_std = xfc_std.where((xfc_std.notnull()) & (xfc_std >= 1e-5), 1.0)
            # Combine into one and computes the values
            self.scaler_fc = xr.concat([xfc_mean, xfc_std], dim="statistic")
            self.scaler_fc = self.scaler_fc.assign_coords(statistic=["mean", "std"])
            self.scaler_fc = self.scaler_fc.compute()

        ### ------------------
        ### Static attributes
        ### ------------------
        if self.cfg.static_input:
            xs_mean = self.ds_attributes.sel(gauge_id=self.valid_gauges).mean(dim=["gauge_id"], skipna=True).fillna(0.0)
            xs_std = self.ds_attributes.sel(gauge_id=self.valid_gauges).std(dim=["gauge_id"], skipna=True)
            xs_std = xs_std.where(xs_std.notnull() & (xs_std >= 1e-5), 1.0)
            self.scaler_attributes = xr.concat([xs_mean, xs_std], dim="statistic")
            self.scaler_attributes = self.scaler_attributes.assign_coords(statistic=["mean", "std"])

        # Save scaler
        scaler = {
            "xd_mean": self.scaler.sel(statistic="mean").to_pandas()["data"].to_dict(),
            "xd_std": self.scaler.sel(statistic="std").to_pandas()["data"].to_dict(),
        }
        if self.cfg.forecast_input:
            scaler.update(
                {
                    "xfc_mean": self.scaler_fc.sel(statistic="mean").to_pandas()["data"].to_dict(),
                    "xfc_std": self.scaler_fc.sel(statistic="std").to_pandas()["data"].to_dict(),
                }
            )
        if self.cfg.static_input:  # Update scaler dictionary with static attributes if applicable
            scaler.update(
                {
                    "xs_mean": self.scaler_attributes.sel(statistic="mean").to_pandas()["data"].to_dict(),
                    "xs_std": self.scaler_attributes.sel(statistic="std").to_pandas()["data"].to_dict(),
                }
            )

        with open(self.cfg.path_save_folder / "scaler.yml", "w") as file:
            yaml.dump(scaler, file, default_flow_style=False, sort_keys=False)

    def _check_dataset(self, ds: xr.Dataset):
        """Check if the dataset has the information requested in the configuration file.

        Parameters
        ----------
        ds : xr.Dataset
            Lazy view of the dataset

        Raises
        ------
        ValueError
            If the dataset is missing information

        """
        # Check if all gauge_id are present in the dataset
        if not np.isin(self.gauge_id, ds["gauge_id"].values).all():
            missing_ids = np.setdiff1d(self.gauge_id, ds["gauge_id"].values)
            raise ValueError(
                f"The following gauge_id are not present in the dataset: {', '.join(missing_ids)}\n"
                f"Please remove them from gauge_id or include the information in the dataset."
            )

        # Check if all variables of interest are present in the dataset
        if not np.isin(self.variables_of_interest, ds["feature"].values).all():
            missing_ids = np.setdiff1d(self.variables_of_interest, ds["feature"].values)
            raise ValueError(
                f"The following variables are not present in the dataset: {', '.join(missing_ids)}\n"
                f"Please remove them from variables_of_interest or include the information in the dataset."
            )

        # Check if all dates exist in ds_date
        date_range = pd.date_range(start=self.warmup_start_date, end=self.end_date, freq=self.data_freq)
        if not np.isin(date_range, ds["date"].values).all():
            missing = np.setdiff1d(date_range, ds["date"].values)
            raise ValueError(
                f"Dataset is missing {len(missing)} required dates.\n"
                f"Required dates span from {date_range[0]} to {date_range[-1]} with frequency {self.data_freq}.\n"
                f"Dataset dates span from {ds['date'].values[0]} to {ds['date'].values[-1]}"
            )

    def _check_ram(self, required_ram: float) -> bool:
        """Check if there is enough RAM available to load the dataset into memory.

        Parameters
        ----------
        required_ram : float
            Estimated size of the dataset in bytes.

        Returns
        -------
        bool
            True if there is enough RAM available, False otherwise.

        """
        available_ram = psutil.virtual_memory().available
        if available_ram >= required_ram * self.cfg.ram_safety_factor:
            return True
        else:
            self.cfg.logger.warning(
                f"Not enough RAM available to load the dataset into memory. Available RAM: {available_ram / 1e9:.2f}"
                f"GB, Required RAM: {(required_ram * self.cfg.ram_safety_factor) / 1e9:.2f} GB (considering safety"
                f"factor of {self.cfg.ram_safety_factor})."
            )
            return False

    def _create_dataset(self):
        """Creates the dataset and keeps it in memory.

        The function reads and process the data, for each entity (e.g., catchment), and store the results as an
        in-memory `xarray.Dataset` (`self.ds_ts`) with the following structure:

        - Dimensions / Coordinates:
            - gauge_id: Unique identifiers for the gauges (e.g., catchments).
            - date: Time dimension.
            - feature: Variables of interest
        - Data Variables:
            - data (gauge_id, date, feature): The primary data values stored as a float array.

        Note: Because this function builds the entire dataset in RAM, you must have enough memory to hold the processed
        data.

        """
        self.cfg.logger.info(f"Creating {self.period} dataset in memory...")

        # Get metadata (dates and features) from a sample to pre-allocate memory
        dates = pd.date_range(start=self.warmup_start_date, end=self.end_date, freq=self.data_freq)
        features = self.variables_of_interest

        # Pre-allocate a 3D numpy array: (ids, dates, features)
        data_array = np.zeros((len(self.gauge_id), len(dates), len(features)), dtype="float32")

        # Process each entity using dask
        with (
            LocalCluster(n_workers=max(self.cfg.num_workers, 1), threads_per_worker=1) as cluster,
            Client(cluster) as client,
        ):
            futures = client.map(self._process_df, self.gauge_id, extract_values=True)
            # Create a mapping to place arrays in correct order
            future_to_idx = {future: idx for idx, future in enumerate(futures)}
            # Monitor progress
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing gauges", unit="entity", ascii=True
            ):
                idx = future_to_idx[future]
                data_array[idx, :, :] = future.result()

        # Construct xarray Dataset
        self.ds_ts = xr.Dataset(
            coords={
                "gauge_id": ("gauge_id", np.array(self.gauge_id, dtype=object)),
                "date": dates,
                "feature": features,
            },
            data_vars={"data": (("gauge_id", "date", "feature"), data_array)},
        )

        self.ds_ts = self.ds_ts.chunk({"gauge_id": 1, "date": -1, "feature": -1})  # chunking helps speed up operations.
        self.cfg.logger.info("Dataset created successfully.")

    def _create_zarr_dataset(self):
        """Creates the dataset and writes it directly to a zarr file on disk.

        This function reads and processes the data for each entity (e.g., catchment) and stores the results,
        sequentially, in a zarr structure with the following format:

        - Dimensions / Coordinates:
            - gauge_id: Unique identifiers for the gauges (e.g., catchments).
            - date: Time dimension.
            - feature: Variables of interest
        - Data Variables:
            - data (gauge_id, date, feature): The primary data values stored as a float32 array.

        Note: This function is designed to handle datasets larger than available RAM by writing the processed data
        sequentially to disk.

        """
        self.cfg.logger.info("Creating zarr dataset...")
        # Initialize zarr structure to store the processed results
        self.path_dataset = self.cfg.path_data / f"dataset_{self.period}.zarr"
        self._initialize_zarr()

        # Create array of entities' index (used later to write the data in the right position in the zarr)
        zarr_ids = zarr.open(self.path_dataset, mode="r")["gauge_id"][:]
        gauge_idx = [np.where(zarr_ids == id)[0][0] for id in self.gauge_id]

        # Process each entity using dask
        with (
            LocalCluster(n_workers=max(self.cfg.num_workers, 1), threads_per_worker=1) as cluster,
            Client(cluster) as client,
        ):
            futures = client.map(self._write_df_to_zarr, self.gauge_id, gauge_idx)  # Run function
            # Monitor progress
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing gauges", unit="entity", ascii=True
            ):
                future.result()

        zarr.consolidate_metadata(self.path_dataset)  # consolidate metadata to optimize read performance
        self.cfg.logger.info(f"Dataset created successfully. Zarr file can be found at {self.path_dataset}")

    def _finalize_setup(self):
        """Prepares the dataset for PyTorch training by optimizing memory and data access.

        This method is the final step in the setup process. It takes the prepared data and formats it specifically for
        fast loading during model training.

        1. RAM Loading: If the user requested to load the dataset in RAM and we have enough RAM, we load the dataset in
        memory, standardize it, and transform it into share_memory() tensors to avoid memory crashes when PyTorch uses
        multiple workers.
        2. Lazy Loading: If the datasets will not be loaded into RAM, we remove their reference (set it to None),
        because we will re-open their reference directly in the PyTorch workers using tensorstore engine, which speeds
        up the reading of the information and avoid fork/spawn issues.

        """
        # If we still do not have the dataset in RAM is because we did a lazy loading of an existing dataset. If the
        # user requested to load the dataset in RAM, we check if we have enough RAM to do it and if so, we load it.
        # Otherwise, we remove the reference, to re-open it directly in the workers using tensorstore.
        if not self.dataset_in_ram:
            if self.cfg.dataset_in_ram and self._check_ram(required_ram=self.ds_ts.nbytes):
                self.ds_ts = self.ds_ts.compute()
                self.dataset_in_ram = True
            else:
                self.ds_ts = None

        # If we have a forecast dataset, we do the same for it.
        if self.cfg.path_forecast_dataset is not None:
            if self.cfg.forecast_dataset_in_ram and self._check_ram(required_ram=self.ds_fc.nbytes):
                self.ds_fc = self.ds_fc.compute()
                self.fc_in_ram = True
            else:
                self.ds_fc = None

        # Standardize data
        self._standardize_data()

        # Converts xarray to shared memory tensors if the dataset is in RAM
        if self.dataset_in_ram:
            self.ds_ts = torch.from_numpy(self.ds_ts["data"].values).float()
            self.ds_ts.share_memory_()

        if self.fc_in_ram:
            self.ds_fc = torch.from_numpy(self.ds_fc["data"].values).float()
            self.ds_fc.share_memory_()

        if self.cfg.static_input:
            self.ds_attributes = torch.from_numpy(self.ds_attributes["data"].values).float()
            self.ds_attributes.share_memory_()

    def _initialize_zarr(self):
        """Creates the zarr structure to store the data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: Unique identifiers for the gauges (e.g., catchments).
            - date: Time dimension.
            - feature: Variables of interest
        - Data Variables:
            - data (gauge_id, date, feature): The primary data values stored as a float array.

        """
        sample_df = self._process_df(gauge_id=self.gauge_id[0])
        features = sample_df.columns.tolist()

        # Create template to store the processed data
        ds_template = xr.Dataset(
            coords={
                "gauge_id": ("gauge_id", np.array(self.gauge_id, dtype=object)),
                "date": sample_df.index,
                "feature": features,
            },
            data_vars={
                "data": (
                    ("gauge_id", "date", "feature"),
                    np.zeros((len(self.gauge_id), len(sample_df), len(features)), dtype="float32"),
                )
            },
        )

        # Chunk the dataset by id to optimize read/write operations
        ds_template = ds_template.chunk({"gauge_id": 1, "date": -1, "feature": -1})

        # Save zarr template to disk (at this point the zarr file is empty, but with the right structure to store the
        # processed data).
        ds_template.to_zarr(self.path_dataset, compute=False, mode="w", consolidated=False)

    def _load_dataset(self, path_dataset: Optional[Path] = None, engine: str = "xarray"):
        """Loads an existing zarr dataset from disk.

        This function opens a zarr dataset and filters it to include only the gauges, time periods, and variables of
        interest specified in the configuration file. It also performs sanity checks to ensure all requested data is
        present.

        Expected format of zarr file:
        - Dimensions / Coordinates:
            - gauge_id: Unique identifiers for the gauges (e.g., catchments).
            - date: Time dimension.
            - feature: Variables of interest
        - Data Variables:
            - data (gauge_id, date, feature): The primary data values stored as a float32 array.

        Parameters
        ----------
        path_dataset: Optional[Path]
            If provided, the function will load the dataset from the specified path instead of the default path in the
            configuration.
        engine: str, default="xarray"
            The engine to use for loading the zarr dataset.

        """
        if path_dataset is not None:
            self.path_dataset = path_dataset
        else:
            self.path_dataset = self.cfg.path_data / f"dataset_{self.period}.zarr"

        if engine == "tensorstore":  # used for batch construction
            ds = xarray_tensorstore.open_zarr(self.path_dataset)
        elif engine == "xarray":  # used for data processing and validation
            self.cfg.logger.info("Loading dataset from zarr...")
            ds = xr.open_zarr(self.path_dataset, consolidated=True)
            self._check_dataset(ds=ds)
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'xarray' or 'tensorstore'")

        # Filter dataset based on the information requested in the configuration file (in lazy mode)
        self.ds_ts = ds.sel(
            gauge_id=self.gauge_id,
            date=slice(self.warmup_start_date, self.end_date),
            feature=self.variables_of_interest,
        )

    def _load_forecast_dataset(self, engine: str = "xarray"):
        """Load forecast dataset from zarr file.

        This function opens a zarr dataset and filters it to include only the gauges, time periods, lead_times and
        variables of interest, specified in the configuration file. I

        Expected format of zarr file:
        - Dimensions / Coordinates:
            - gauge_id: Unique identifiers for the gauges (e.g., catchments).
            - date: Time dimension.
            - lead_time: Lead time dimension (e.g., forecast horizon).
            - feature: Variables of interest
        - Data Variables:
            - data (gauge_id, date, lead_time, feature): The primary data values stored as a float32 array.

        Parameters
        ----------
        engine: str, default="xarray"
            The engine to use for loading the zarr dataset.

        """
        if engine == "tensorstore":  # used for batch construction
            ds_fc = xarray_tensorstore.open_zarr(self.cfg.path_forecast_dataset)
        elif engine == "xarray":  # used for data processing and validation
            self.cfg.logger.info("Loading forecast dataset from zarr...")
            ds_fc = xr.open_zarr(self.cfg.path_forecast_dataset, consolidated=True)
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'xarray' or 'tensorstore'")

        self.ds_fc = ds_fc.sel(
            gauge_id=self.gauge_id,
            date=slice(self.warmup_start_date, self.end_date),
            lead_time=slice(0, self.cfg.seq_length_forecast),
            feature=self.forecast_input,
        )

    def _load_scaler(self, path_scaler: Path | str):
        """Load precomputed scaler.

        Parameters
        ----------
        path_scaler: Path | str
            Path to the .yml file containing the scaler

        """
        # Load scaler
        with open(path_scaler, "r") as file:
            scaler = yaml.safe_load(file)

        # ----------------------------------
        # Re-build scaler for dynamic variables
        # ----------------------------------
        dyn_keys = list(scaler["xd_mean"].keys())
        dyn_means = [scaler["xd_mean"][k] for k in dyn_keys]
        dyn_stds = [scaler["xd_std"][k] for k in dyn_keys]

        da = xr.DataArray(
            np.stack([dyn_means, dyn_stds], axis=0),
            coords={"statistic": ["mean", "std"], "feature": dyn_keys},
            dims=("statistic", "feature"),
        )
        self.scaler = da.to_dataset(name="data")

        # ----------------------------------
        # Re-build scaler for forecast
        # ----------------------------------
        if self.cfg.forecast_input:
            fc_keys = list(scaler["xfc_mean"].keys())
            fc_means = [scaler["xfc_mean"][k] for k in fc_keys]
            fc_stds = [scaler["xfc_std"][k] for k in fc_keys]

            da = xr.DataArray(
                np.stack([fc_means, fc_stds], axis=0),
                coords={"statistic": ["mean", "std"], "feature": fc_keys},
                dims=("statistic", "feature"),
            )
            self.scaler_fc = da.to_dataset(name="data")

        # ----------------------------------
        # Re-build scaler for static attributes
        # ----------------------------------
        if self.cfg.static_input:
            xs_keys = list(scaler["xs_mean"].keys())
            xs_means = [scaler["xs_mean"][k] for k in xs_keys]
            xs_stds = [scaler["xs_std"][k] for k in xs_keys]

            da = xr.DataArray(
                np.stack([xs_means, xs_stds], axis=0),
                coords={"statistic": ["mean", "std"], "feature": xs_keys},
                dims=("statistic", "feature"),
            )
            self.scaler_attributes = da.to_dataset(name="data")

    def _load_valid_samples(self, path_valid_samples: Path | str):
        """Load precomputed valid-samples.

        Parameters
        ----------
        path_valid_samples: Path | str
            Path to the csv file containing the valid samples

        """
        self.valid_samples = np.loadtxt(
            path_valid_samples,
            delimiter=",",
            dtype=[("gauge_id", "O"), ("date", "datetime64[ns]"), ("source", "O")],
            skiprows=1,  # header
            encoding="utf-8",
        )

        # Sort valid samples by gauge_id and date
        self.valid_samples = np.sort(self.valid_samples, order=["gauge_id", "date"])

        # Map valid samples to their corresponding indexes in the zarr structure for faster access in getitem
        self._map_indexes()

    def _map_indexes(self):
        """Map gauge_id, date and feature to their corresponding indexes

        This function creates the mapping between gauge_id, date and features to their corresponding indexes in the
        dataset. This is done to speed up the __getitems__ function. Depending on the source of the valid samples
        (obs or forecast), the index-mapping is done to a different dataset.

        """
        self.cfg.logger.info("Mapping ids, dates and features to their corresponding indexes")

        # Create structured array to store the valid indexes in the hindcast and the forecast period.
        self.valid_idx_hc = np.zeros(len(self.valid_samples), dtype=[("gauge_idx", "i4"), ("date_idx", "i4")])
        self.valid_idx_fc = np.zeros(len(self.valid_samples), dtype=[("gauge_idx", "i4"), ("date_idx", "i4")])
        self.feature_to_idx = {}

        # -------------------------
        # Valid indices hindcast (always mapped to ds_ts)
        # -------------------------
        # Check all dates in valid_samples are contained in ds_ts. Important to avoid errors when mapping indexexs
        if not np.isin(self.valid_samples["date"], self.ds_ts["date"].values).all():
            raise ValueError("Some dates in valid_samples are missing from ds_ts.")
        self.valid_idx_hc["date_idx"] = np.searchsorted(self.ds_ts["date"].values, self.valid_samples["date"])
        id_to_idx = dict(zip(self.ds_ts["gauge_id"].values, range(self.ds_ts["gauge_id"].size), strict=True))
        self.valid_idx_hc["gauge_idx"] = [id_to_idx[sample_id] for sample_id in self.valid_samples["gauge_id"]]

        # -------------------------
        # Valid indices forecat: Depending on the source (obs or forecast) the mapping is done to a different dataset.
        # -------------------------
        for source in np.unique(self.valid_samples["source"]):
            # Extract the valid dates and gauge_ids for current source
            source_mask = self.valid_samples["source"] == source
            valid_date = self.valid_samples["date"][source_mask]
            valid_gauge = self.valid_samples["gauge_id"][source_mask]

            # Define target date and gauge_id based on the source
            if source == "obs":
                target_date = self.ds_ts["date"].values
                target_gauge = self.ds_ts["gauge_id"].values
                target_feature = self.ds_ts["feature"].values
            elif source == "fc":
                target_date = self.ds_fc["date"].values
                target_gauge = self.ds_fc["gauge_id"].values
                target_feature = self.ds_fc["feature"].values
            else:
                raise ValueError(f"Invalid source: {source}. Expected 'obs' or 'fc'.")

            if not np.isin(valid_date, target_date).all():
                raise ValueError(f"Some dates in valid_samples are missing from {source} dataset.")
            self.valid_idx_fc["date_idx"][source_mask] = np.searchsorted(target_date, valid_date)
            id_to_idx = dict(zip(target_gauge, range(len(target_gauge)), strict=True))
            self.valid_idx_fc["gauge_idx"][source_mask] = [id_to_idx[sample_id] for sample_id in valid_gauge]

            # Map features
            self.feature_to_idx[source] = dict(zip(target_feature, range(len(target_feature)), strict=True))

    def _parse_datetime(self, date_str: str, freq: str) -> pd.Timestamp:
        """Convert frequency string into pandas Timestamp object.

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

    def _process_attributes(self):
        """Read and process static attributes."""
        if self.cfg.static_input:
            ds_attributes = self._read_attributes().to_xarray()
            self.ds_attributes = (
                ds_attributes.reindex(gauge_id=self.ds_ts.gauge_id)
                .to_array(dim="feature")
                .transpose("gauge_id", "feature")
                .to_dataset(name="data")
            )

    def _process_df(self, gauge_id: str, extract_values: bool = False) -> pd.DataFrame | np.ndarray:
        """Read and process the data for a specific gauge_id (e.g., catchment).

        Parameters
        ----------
        gauge_id : str
            Id of the gauge to be processed.
        extract_values: bool, default=False
            Whether to return the processed data as a pandas DataFrame (False) or as a numpy array with only the values

        Returns
        -------
        pd.DataFrame or np.ndarray
            Processed data for the specific entity, either as a pandas DataFrame or as a numpy array.

        """
        # Load time series for specific catchment id
        df_ts = self._read_data(gauge_id=gauge_id)

        # Load additional features, if applicable, and concatenate them to the main dataframe
        if self.cfg.path_additional_features:
            ds_additional_features = xarray_tensorstore.open_zarr(self.cfg.path_additional_features)
            df_additional_features = ds_additional_features.sel(gauge_id=gauge_id)["data"].to_pandas()
            df_ts = pd.concat([df_ts, df_additional_features], axis=1)

        # Add lagged features, if aplicable.
        if isinstance(self.cfg.lagged_features, dict):
            df_ts = self._add_lagged_features(df=df_ts)

        # Filter dataframe for the period and variables of interest
        df_ts = df_ts.loc[self.warmup_start_date : self.end_date, self.variables_of_interest]

        # Reindex the dataframe to assure continuos data between the start and end date of the time period. Missing
        # data will be filled with NaN, so this will be taken care of later.
        full_range = pd.date_range(start=self.warmup_start_date, end=self.end_date, freq=self.data_freq)
        df_ts = df_ts.reindex(full_range)

        return df_ts if not extract_values else df_ts.values.astype("float32")

    def _read_attributes(self) -> pd.DataFrame:
        # This function is specific for each dataset
        raise NotImplementedError

    def _read_data(self) -> pd.DataFrame:
        # This function is specific for each dataset
        raise NotImplementedError

    def _standardize_data(self):
        """Standardize the data using the previously-computed scaler."""

        # If observed dataset is in RAM, standardize it
        if self.dataset_in_ram:
            self.ds_ts = (self.ds_ts - self.scaler.sel(statistic="mean")) / self.scaler.sel(statistic="std")
            self.cfg.logger.info("Dataset was successfully standardized.")
        else:
            self.cfg.logger.info(
                "To keep lazy loading, the data will be standardized on the fly while constructing the batches."
            )

        # If forecast dataset exist and is in RAM, standardize it
        if self.forecast_input:
            if self.fc_in_ram:
                self.ds_fc = (self.ds_fc - self.scaler_fc.sel(statistic="mean")) / self.scaler_fc.sel(statistic="std")
                self.cfg.logger.info("Forecast dataset was successfully standardized.")
            else:
                self.cfg.logger.info(
                    "To keep lazy loading, the forecast data will be standardized on the fly while constructing the "
                    "batches."
                )

        # If we have static attributes, standardize them
        if self.cfg.static_input:
            self.ds_attributes = (
                self.ds_attributes - self.scaler_attributes.sel(statistic="mean")
            ) / self.scaler_attributes.sel(statistic="std")
            self.cfg.logger.info("Static attributes were successfully standardized.")

    def _validate_samples(self, check_nan: bool = True):
        """Function to construct the valid samples table.

        Parameters
        ----------
        check_nan : Optional[bool], default=True
            Whether to check for NaN values while processing the data. This should typically be True during training,
            and can be set to False during evaluation (validation/testing).

        """
        self.cfg.logger.info("Validating samples...")
        # Construct the validity mask as a lazy Dask array. This will not execute the computations yet, but will build
        # the graph of operations to compute the validity mask.
        logic_valid_mask = self._valid_samples_mask(check_nan=check_nan)

        # Compute validation mask
        with TqdmCallback(desc="Validating samples", unit="tasks", tqdm_class=tqdm, ascii=True):
            valid_mask = logic_valid_mask.compute()

        # If the user requested to have unique prediction blocks (non-overlapping samples)
        if self.cfg.unique_prediction_blocks:
            # non-overlapping block indices
            block_id = np.arange(valid_mask.sizes["date"] // self.cfg.predict_last_n) * self.cfg.predict_last_n + (
                self.cfg.predict_last_n - 1
            )
            # non-overlapping mask
            date_filter_np = np.zeros(valid_mask.sizes["date"], dtype=bool)
            date_filter_np[block_id] = True
            xr_date_filter = xr.DataArray(date_filter_np, dims=["date"], coords={"date": valid_mask.date})
            # Apply mask
            valid_mask = valid_mask & xr_date_filter

        # Check for gauge_id with valid samples
        self.valid_gauges = valid_mask.gauge_id[valid_mask.any(dim=["date", "source"])].values
        self.cfg.logger.info(f"Number of gauges with valid samples: {len(self.valid_gauges)}")
        invalid_gauges = valid_mask.gauge_id[~valid_mask.any(dim=["date", "source"])].values
        if len(invalid_gauges) > 0:
            self.cfg.logger.warning(f"Gauges without valid samples in period of interest: {', '.join(invalid_gauges)}")

        # Extract valid samples (id, date, source) from the mask
        gauge_indices, date_indices, source_indices = np.where(valid_mask.values)

        # store valid samples in a structured array
        self.valid_samples = np.empty(
            len(gauge_indices), dtype=[("gauge_id", "O"), ("date", "datetime64[ns]"), ("source", "O")]
        )
        self.valid_samples["gauge_id"] = valid_mask.gauge_id.values[gauge_indices]
        self.valid_samples["date"] = valid_mask.date.values[date_indices]
        self.valid_samples["source"] = valid_mask.source.values[source_indices]
        self.valid_samples = np.sort(self.valid_samples, order=["gauge_id", "date"])  # sort by gauge_id and date

        self.cfg.logger.info(f"Number of valid samples: {self.valid_samples.size:_}".replace("_", " "))

        # Map valid samples to their corresponding indexes in the zarr structure for faster access in getitem
        self._map_indexes()

    def _valid_samples_mask(self, check_nan: bool = True) -> xr.DataArray:
        """Logic to check for valid samples.

        Parameters
        ----------
        check_nan : Optional[bool], default=True
            Whether to check for NaN values while processing the data. This should be True during training,
            and can be set to False during evaluation (validation/testing).

        Returns
        -------
        is_valid: xr.DataArray
            A boolean mask (gauge_id, date, source) where True indicates valid sample.
            - gauge_id: id of gauge
            - date: date
            - source: "obs" for hindcast and pseudoforecast, "fc" for forecast (if applicable)

        """

        def _check_vars(
            var_list: list[str], window_size: int, shift_val: int = 0, any_nan: bool = True
        ) -> xr.DataArray:
            """Helper function to check a list of variables for NaNs over a window of a given size.

            This function checks if there are any NaN values in the specified variables, over a rolling window of a
            given size.

            Parameters
            ----------
            var_list: list[str]
                List of variable names to check.
            window_size: int
                Size of the rolling window.
            shift_val: int, default=0
                Number of steps to shift the resulting mask.
            any_nan: bool, default=True
                True: the sample is invalid if ANY variable is NaN
                False: the sample is invalid if ALL variables are NaN

            Returns
            -------
            valid_mask: xr.DataArray
                A boolean mask (gauge_id, date) where True indicates valid sample

            """
            # Select variables and convert to single DataArray
            ds_subset = self.ds_ts["data"].sel(feature=var_list)

            # Check NaNs across variables
            is_invalid = ds_subset.isnull().any(dim="feature") if any_nan else ds_subset.isnull().all(dim="feature")

            # Rolling count of NaNs
            invalid_targets_in_window = is_invalid.rolling(
                date=window_size, center=False, min_periods=window_size
            ).sum()

            valid_mask = invalid_targets_in_window == 0 if any_nan else invalid_targets_in_window < window_size

            # Shift if checking a sub-segment that ends displaced from the valid sample index (e.g., when we have
            # multiple frequencies or pseudo-forecast input)
            if shift_val != 0:
                valid_mask = valid_mask.shift(date=shift_val, fill_value=False)

            return valid_mask

        def _check_forecast_vars(var_list: list[str]) -> xr.DataArray:
            """Helper function to check a list of variables for NaNs over lead_time dimension"""
            ds_subset = self.ds_fc["data"].sel(feature=var_list).isel(lead_time=slice(0, self.cfg.seq_length_forecast))
            valid_mask = ds_subset.notnull().all(dim=["feature", "lead_time"])
            return valid_mask

        def _check_groups(
            group_vars: dict[str, list[str]],
            window_size: int = 1,
            shift_val: int = 0,
            source: str = "obs",
            any_nan: bool = True,
        ) -> xr.DataArray:
            """Helper function to check group of variables.

            The sample is invalid if:
            - all the groups have NaN elements in the same point
            - a "mandatory group" have NaN elements. A mandatory group is a group that according to the
            'nan_probability' configuration argument have a nan_seq = 0

            Parameters
            ----------
            group_vars: dict[str, list[str]]
                Dictionary where keys are group names and values are lists of variable names to check.
            window_size: int, default=1
                Size of the rolling window (e.g., sequence length).
            shift_val: int, default=0
                Number of steps to shift the resulting mask. This is useful when checking sub-segments that end before
                the full sequence end.
            source: str, default="obs"
                Whether we are checking the groups for the hindcast/pseudoforecast ("obs") or for the forecast ("fc").
            any_nan: bool, default=True
                True: the sample is invalid if ANY variable are NaN
                False: the sample is invalid if ALL variables are NaN
            Returns
            -------
            mask: xr.DataArray
                A boolean mask (gauge_id, date), for the whole group, where True indicates valid samples.

            """
            mask_groups = None
            mask_mandatory_groups = None

            for g_name, g_vars in group_vars.items():
                if source == "obs":
                    g_mask = _check_vars(var_list=g_vars, window_size=window_size, shift_val=shift_val, any_nan=any_nan)
                elif source == "fc":
                    g_mask = _check_forecast_vars(var_list=g_vars)
                # The mask is True if there is at least one valid group (logical OR -> |)
                mask_groups = g_mask if mask_groups is None else (mask_groups | g_mask)
                # The mask is True if all mandatory groups are True (logical AND -> &)
                if self.cfg.nan_probability and self.cfg.nan_probability[g_name]["nan_seq"] == 0:
                    mask_mandatory_groups = (
                        g_mask if mask_mandatory_groups is None else (mask_mandatory_groups & g_mask)
                    )

            # Mask considering all groups: True if at least one group valid AND all mandatory groups valid
            valid_mask = mask_groups if mask_mandatory_groups is None else (mask_groups & mask_mandatory_groups)

            return valid_mask

        # -------------------------
        # Initialize mask
        # -------------------------
        is_valid = xr.full_like(self.ds_ts["data"].isel(feature=0), True, dtype=bool).drop_vars("feature")

        # Too early (not enough history to form the sequence)
        too_early = is_valid.shift(date=(self.cfg.seq_length_hindcast - 1), fill_value=False)
        is_valid = is_valid & too_early

        # Too late (not enough future to form a pseudo-forecast sequence)
        if self.pseudo_forecast_input:
            too_late_mask = is_valid.shift(date=-self.cfg.seq_length_forecast, fill_value=False)
            is_valid = is_valid & too_late_mask

        # Ablation flag check
        if "ablation_flag" in self.ds_ts.coords["feature"].values:
            da_ablation = self.ds_ts["data"].sel(feature="ablation_flag")
            ablation_mask = (da_ablation != 0) & da_ablation.notnull()
            is_valid = is_valid & ablation_mask

        # -------------------------
        # # Early exit if we don't want to check NaNs.
        # -------------------------
        if not check_nan:
            if not self.pseudo_forecast_input and not self.forecast_input:
                return is_valid.expand_dims(source=["obs"]).transpose(..., "source")
            elif self.pseudo_forecast_input and not self.forecast_input:
                return is_valid.expand_dims(source=["obs"]).transpose(..., "source")
            elif not self.pseudo_forecast_input and self.forecast_input:
                # Even if we are not checking for NaNs, we can only consider as valid the samples the dates that have
                # forecast information.
                mask = xr.DataArray(
                    True, coords={"gauge_id": self.ds_fc.gauge_id, "date": self.ds_fc.date}, dims=["gauge_id", "date"]
                )
                mask = mask.reindex(date=is_valid.date, fill_value=False)
                is_valid = is_valid & mask
                return is_valid.expand_dims(source=["fc"]).transpose(..., "source")
            else:
                raise ValueError(
                    "Having pseudo-forecast and forecast inputs is only supported for training,"
                    "where check_nan should be set to True."
                )

        # -------------------------
        # Check attributes: any NaN in the static features makes all the samples of that id invalid
        # -------------------------
        if self.cfg.static_input:
            # Check if any attribute is NaN for each id, if attr is invalid, the whole time series for that id is False
            attr_mask = self.ds_attributes["data"].notnull().all(dim="feature")
            is_valid = is_valid & attr_mask

        # -------------------------
        # Hindcast NaN check
        # -------------------------
        # Case 1: If we use the same variables along the sequence length, and only one group of variables, any NaN makes
        # the sample invalid.
        # Examples:
        #   - Single group of variables and and single frequency
        #   - Multi-frequency approaches but all frequencies use the same single group of variables.
        if isinstance(self.cfg.dynamic_input, list):
            mask = _check_vars(var_list=self.cfg.dynamic_input, window_size=self.cfg.seq_length_hindcast)
            is_valid = is_valid & mask

        # Case 2: If we have multiple groups of variables, and use the same groups along the whole sequence.
        # Examples:
        #   - We have multiple group of variables but have single frequency data.
        elif isinstance(self.cfg.dynamic_input, dict) and self.cfg.custom_seq_processing is None:
            mask = _check_groups(group_vars=self.cfg.dynamic_input, window_size=self.cfg.seq_length_hindcast)
            is_valid = is_valid & mask

        # Case 3: If we use the different variables (or group of variables) along the sequence length.
        # Examples:
        #   - We have multi-frequency approaches and the variables change along the sequence.
        elif isinstance(self.cfg.dynamic_input, dict) and isinstance(self.cfg.custom_seq_processing, dict):
            aux_index = 0  # start of sequence subset
            seq_valid = None  # Initialize a mask that we will accumulate into

            # Goes through each sub-segment defined in custom_seq_processing
            for k, v in self.cfg.custom_seq_processing.items():
                sub_seq_len = v["n_steps"] * v["freq_factor"]  # Length of sub-segment

                # Calculate shift: The validity of the sub-segment should be assigned at the end of the full sequence.
                shift_fwd = self.cfg.seq_length_hindcast - (aux_index + sub_seq_len)

                # If we have single group of variables for each frequency
                if isinstance(self.cfg.dynamic_input[k], list):
                    vars_to_check = self.cfg.dynamic_input[k]
                    seg_mask = _check_vars(var_list=vars_to_check, window_size=sub_seq_len, shift_val=shift_fwd)

                # Sub-groups within the frequency block
                elif isinstance(self.cfg.dynamic_input[k], dict):
                    seg_mask = _check_groups(
                        group_vars=self.cfg.dynamic_input[k], window_size=sub_seq_len, shift_val=shift_fwd
                    )

                # Accumulate: All segments must be valid (AND logic across time segments)
                seq_valid = seg_mask if seq_valid is None else (seq_valid & seg_mask)

                # Advance index
                aux_index += sub_seq_len

            is_valid = is_valid & seq_valid

        # -------------------------
        # Target NaN check: all-NaN in the targets makes the sample invalid
        # -------------------------
        target_mask = _check_vars(
            var_list=self.cfg.target,
            window_size=self.cfg.predict_last_n,
            shift_val=-self.cfg.seq_length_forecast,
            any_nan=False,
        )
        is_valid = is_valid & target_mask

        # -------------------------
        # Pseudo-forecat and forecast
        # We need to check two potential cases, with different information source: pseudo-forecast and forecast. Both
        # are treated independently, meaning both can generate valid_samples. However, we need to distinguish the
        # sources so we can later look for the respective sample.
        # -------------------------
        source_masks = {}
        if self.cfg.pseudo_forecast_input:  # pseudo-forecast
            pfc_mask = is_valid.copy()
            if isinstance(self.cfg.pseudo_forecast_input, list):
                mask = _check_vars(
                    var_list=self.cfg.pseudo_forecast_input,
                    window_size=self.cfg.seq_length_forecast,
                    shift_val=-self.cfg.seq_length_forecast,
                )
                pfc_mask = pfc_mask & mask

            elif isinstance(self.cfg.pseudo_forecast_input, dict):
                mask = _check_groups(
                    group_vars=self.cfg.pseudo_forecast_input,
                    window_size=self.cfg.seq_length_forecast,
                    shift_val=-self.cfg.seq_length_forecast,
                )
                pfc_mask = pfc_mask & mask

            source_masks["obs"] = pfc_mask.expand_dims(source=["obs"]).transpose(..., "source")

        if self.cfg.forecast_input:  # forecast
            fc_mask = is_valid.copy()
            if isinstance(self.cfg.forecast_input, list):
                mask = _check_forecast_vars(var_list=self.cfg.forecast_input)
            elif isinstance(self.cfg.forecast_input, dict):
                mask = _check_groups(group_vars=self.cfg.forecast_input, source="fc")

            mask = mask.reindex(date=fc_mask.date, fill_value=False)
            fc_mask = fc_mask & mask
            source_masks["fc"] = fc_mask.expand_dims(source=["fc"]).transpose(..., "source")

        # Combine masks if neccesary, otherwise returns single mask with default source "obs"
        return (
            xr.concat(list(source_masks.values()), dim="source")
            if len(source_masks) > 0
            else is_valid.expand_dims(source=["obs"]).transpose(..., "source")
        )

    def _write_df_to_zarr(self, gauge_id: str, gauge_idx: int):
        """Process the data for a specific entity (e.g., catchment) and write it to the zarr file.

        Parameters
        ----------
        gauge_id : str
            Id of the gauge to be processed and stored in the zarr structure.
        gauge_idx: int
            Index of the gauge in the zarr structure (we use it to write the data in the right position in the zarr).

        """
        # Write the processed dataframe to the specific region of the existing zarr file
        zarr_file = zarr.open(self.path_dataset, mode="r+")
        zarr_file["data"][gauge_idx, :, :] = self._process_df(gauge_id=gauge_id, extract_values=True)

    @staticmethod
    def dask_worker_init_fn(worker_id):
        """Initialization function for Dask workers.

        Note: This function is called inside each PyTorch worker

        """
        dask.config.set(scheduler="synchronous")

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to construct batches

        Because we are using getitems instead of getitem, we are already constructing the batch inside the getitems
        function. Therefore, the collate function does not need any further processing.

        """
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
