import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm

from hy2dl.datasetzoo import BaseDataset
from hy2dl.utils.config import Config
from hy2dl.utils.sampler import GaugeBatchSampler
from hy2dl.utils.utils import upload_to_device


class BaseEvaluator:
    """Class to produce and store the evaluation results.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    evaluation_dataset : BaseDataset
        Dataset used for evaluation.

    """

    def __init__(self, cfg: Config, evaluation_dataset: BaseDataset):
        # Intialize variables
        self.cfg = cfg

        # Extract gauges and dates from evaluation dataset
        self.unique_gauges = np.unique(evaluation_dataset.valid_samples["gauge_id"])
        self.gauge_idx = dict(zip(self.unique_gauges, range(len(self.unique_gauges)), strict=True))
        self.date_range = pd.date_range(
            start=evaluation_dataset.valid_samples["date"].min(),
            end=evaluation_dataset.valid_samples["date"].max(),
            freq=evaluation_dataset.data_freq,
        )

        # We use a custom sampler that construct the batches by sampling elements of the same gauge_id.
        evaluation_sampler = GaugeBatchSampler(
            valid_samples=evaluation_dataset.valid_samples, batch_size=cfg.batch_size_evaluation
        )
        # Create evaluation dataloader.
        self.evaluation_loader = DataLoader(
            dataset=evaluation_dataset,
            batch_sampler=evaluation_sampler,
            num_workers=self.cfg.num_workers,
            collate_fn=evaluation_dataset.collate_fn,
            worker_init_fn=evaluation_dataset.dask_worker_init_fn,
        )

        # Calculate target mean and std to de-normalize the results
        self.target_mean = (
            torch.from_numpy(
                evaluation_dataset.scaler["data"].sel(statistic="mean", feature=evaluation_dataset.cfg.target).values
            )
            .float()
            .to(self.cfg.device)
        )

        self.target_std = (
            torch.from_numpy(
                evaluation_dataset.scaler["data"].sel(statistic="std", feature=evaluation_dataset.cfg.target).values
            )
            .float()
            .to(self.cfg.device)
        )

        self.gauge_data = {"date": [], "y_sim": [], "y_obs": []}
        self.path_zarr = self.cfg.path_save_folder / f"{evaluation_dataset.period}_results.zarr"
        self.forecast_mode = (
            False if len(self.cfg.pseudo_forecast_input) == 0 and len(self.cfg.forecast_input) == 0 else True
        )

        # Given how the forecast zarr are constructed, seq_length_forecast and predict_last_n must be the same.
        if self.forecast_mode and self.cfg.seq_length_forecast != self.cfg.predict_last_n:
            raise ValueError(
                f"seq_length_forecast ({self.cfg.seq_length_forecast}) and predict_last_n ({self.cfg.predict_last_n})"
                f" must be the same."
            )

    def evaluate_model(self, model: torch.nn.Module) -> None:
        model.eval()
        self._initialize_zarr()
        current_gauge = None

        with torch.no_grad():
            iterator = tqdm(
                self.evaluation_loader,
                desc="Evaluation",
                unit="batches",
                ascii=True,
                leave=False,
            )

            for sample in iterator:
                sample = upload_to_device(sample, self.cfg.device)
                batch_gauge_id = sample["gauge_id"][0]

                # If I am switching to a new gauge, I write the results of the previous one to disk and clear memory
                if current_gauge is not None and batch_gauge_id != current_gauge:
                    if not self.forecast_mode:
                        self._write_simulation_to_zarr(current_gauge)
                    else:
                        self._write_forecast_to_zarr(current_gauge)

                    self.gauge_data = {"date": [], "y_sim": [], "y_obs": []}  # clear memory for the new gauge

                # update current gauge
                current_gauge = batch_gauge_id

                # run model
                pred = model(sample)
                y_sim = pred["y_hat"] * self.target_std + self.target_mean
                y_obs = sample["y_obs"] * self.target_std + self.target_mean

                # process results from current batche
                if not self.forecast_mode:
                    self._simulation_results(sample, y_sim, y_obs)
                else:
                    y_obs = (sample["persistent_q"] * self.target_std + self.target_mean).squeeze(1)
                    self._forecast_results(sample, y_sim, y_obs)

                del sample, pred, y_sim, y_obs

            # Write last gauge to disk
            if current_gauge is not None:
                if not self.forecast_mode:
                    self._write_simulation_to_zarr(current_gauge)
                else:
                    self._write_forecast_to_zarr(current_gauge)

        zarr.consolidate_metadata(self.path_zarr)  # consolidate metadata to optimize read performance

    def _initialize_zarr(self):
        if not self.forecast_mode:
            self._initialize_simulation_zarr()
        else:
            self._initialize_forecast_zarr()

    def _initialize_forecast_zarr(self):
        """Creates the zarr structure to store forecast data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: id of gauge (basin)
            - date: date
            - lead_time: forecast lead time
            - feature: target variables
            -
        - Data Variables:
            - y_sim ("gauge_id", "date", "lead_time", "feature"): model predictions.
            - y_obs ("gauge_id", "date", "feature"): observed targets.

        """

        ds_template = xr.Dataset(
            data_vars={
                "y_sim": (
                    ("gauge_id", "date", "lead_time", "feature"),
                    np.full(
                        (
                            len(self.unique_gauges),
                            len(self.date_range),
                            self.cfg.seq_length_forecast,
                            len(self.cfg.target),
                        ),
                        np.nan,
                        dtype="float32",
                    ),
                ),
                "y_obs": (
                    ("gauge_id", "date", "feature"),
                    np.full(
                        (len(self.unique_gauges), len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32"
                    ),
                ),
            },
            coords={
                "gauge_id": self.unique_gauges,
                "date": self.date_range,
                "lead_time": np.arange(1, self.cfg.seq_length_forecast + 1),
                "feature": self.cfg.target,
            },
        )

        # Chunk the dataset. Note we include lead_time in the chunking.
        ds_template = ds_template.chunk({"gauge_id": 1, "date": -1, "lead_time": -1, "feature": -1})
        ds_template.to_zarr(self.path_zarr, compute=False, mode="w", consolidated=False)

    def _initialize_simulation_zarr(self):
        """Creates the zarr structure to store the data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: id of gauge (basin)
            - date: date
            - feature: target variables
        - Data Variables:
            - y_sim: model predictions.
            - y_obs: observed targets.

        """
        # Create template to store the processed data
        ds_template = xr.Dataset(
            data_vars={
                "y_sim": (
                    ("gauge_id", "date", "feature"),
                    np.full(
                        (len(self.unique_gauges), len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32"
                    ),
                ),
                "y_obs": (
                    ("gauge_id", "date", "feature"),
                    np.full(
                        (len(self.unique_gauges), len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32"
                    ),
                ),
            },
            coords={
                "gauge_id": self.unique_gauges,
                "date": self.date_range,
                "feature": self.cfg.target,
            },
        )

        # Chunk the dataset by id to optimize read/write operations
        ds_template = ds_template.chunk({"gauge_id": 1, "date": -1, "feature": -1})

        # Save zarr template to disk (empty, but with the right structure)
        ds_template.to_zarr(self.path_zarr, compute=False, mode="w", consolidated=False)

    def _forecast_results(self, sample, y_sim, y_obs):
        self.gauge_data["date"].append(sample["init_time_fc"])
        self.gauge_data["y_sim"].append(y_sim.cpu().numpy())
        self.gauge_data["y_obs"].append(y_obs.cpu().numpy())

    def _simulation_results(self, sample, y_sim, y_obs):
        # Resolve the sequence dimension (when predict_last_n>1) to build a continuous timeline.
        # - True (Non-overlapping blocks): Flatten batch and sequence dimensions together.
        # - False (Overlapping blocks): Keep only the final timestep to prevent duplicate dates.
        # Note: If predict_last_n == 1, both options yield the same result.
        if self.cfg.unique_prediction_blocks:
            self.gauge_data["date"].append(sample["date"].flatten())
            self.gauge_data["y_obs"].append(y_obs.cpu().flatten(start_dim=0, end_dim=1).numpy())
            self.gauge_data["y_sim"].append(y_sim.cpu().flatten(start_dim=0, end_dim=1).numpy())
        else:
            self.gauge_data["date"].append(sample["date"][:, -1])
            self.gauge_data["y_obs"].append(y_obs[:, -1, :].cpu().numpy())
            self.gauge_data["y_sim"].append(y_sim[:, -1, :].cpu().numpy())

    def _write_forecast_to_zarr(self, gauge_id):
        idx = self.gauge_idx[gauge_id]

        dates = np.concatenate(self.gauge_data["date"], axis=0)
        y_obs = np.concatenate(self.gauge_data["y_obs"], axis=0)
        y_sim = np.concatenate(self.gauge_data["y_sim"], axis=0)

        # avoid issues with missing dates by creating a full timeline and filling in the available data
        y_sim_filled = np.full(
            (len(self.date_range), self.cfg.seq_length_forecast, len(self.cfg.target)), np.nan, dtype="float32"
        )
        y_obs_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")

        init_date_indices = self.date_range.get_indexer(dates)
        valid_ = init_date_indices >= 0
        y_sim_filled[init_date_indices[valid_], :, :] = y_sim[valid_]
        y_obs_filled[init_date_indices[valid_], :] = y_obs[valid_]

        # write to zarr
        zarr_file = zarr.open(self.path_zarr, mode="r+")
        zarr_file["y_sim"][idx, :, :, :] = y_sim_filled
        zarr_file["y_obs"][idx, :, :] = y_obs_filled

    def _write_simulation_to_zarr(self, gauge_id):
        idx = self.gauge_idx[gauge_id]

        # concatenate gauge data
        dates = np.concatenate(self.gauge_data["date"], axis=0)
        y_sim = np.concatenate(self.gauge_data["y_sim"], axis=0)
        y_obs = np.concatenate(self.gauge_data["y_obs"], axis=0)

        # avoid issues with missing dates by creating a full timeline and filling in the available data
        y_sim_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")
        y_obs_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")
        date_indices = self.date_range.get_indexer(dates)
        valid_ = date_indices >= 0
        y_sim_filled[date_indices[valid_], :] = y_sim[valid_]
        y_obs_filled[date_indices[valid_], :] = y_obs[valid_]

        # write to zarr
        zarr_file = zarr.open(self.path_zarr, mode="r+")
        zarr_file["y_sim"][idx, :, :] = y_sim_filled
        zarr_file["y_obs"][idx, :, :] = y_obs_filled
