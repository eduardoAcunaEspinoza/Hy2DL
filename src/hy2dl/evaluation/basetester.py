import datetime
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm

from hy2dl.datasetzoo import BaseDataset
from hy2dl.evaluation.evaluator import calculate_metrics
from hy2dl.utils.config import Config
from hy2dl.utils.sampler import GaugeBatchSampler
from hy2dl.utils.utils import upload_to_device


class BaseTester(object):
    """Class to process and store evaluation results.

    This class is inherited by other evaluator subclasses (e.g. simulation_evaluator, forecast_evaluator) to produce
    and store the evaluation results.

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
        self.date_range = self._create_date_range(evaluation_dataset)

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
        self.target_scaler = {
            "mean": torch.from_numpy(
                evaluation_dataset.scaler["data"].sel(statistic="mean", feature=evaluation_dataset.cfg.target).values
            )
            .float()
            .to(self.cfg.device),
            "std": torch.from_numpy(
                evaluation_dataset.scaler["data"].sel(statistic="std", feature=evaluation_dataset.cfg.target).values
            )
            .float()
            .to(self.cfg.device),
        }
        # Set path to save results
        if self.cfg.path_save_zarr is None:
            self.path_zarr = self.cfg.path_save_folder / f"{evaluation_dataset.period}_results.zarr"
        else:
            self.path_zarr = self.cfg.path_save_zarr
        # Initialize validation report
        self.validation_report = ""

    def evaluate_model(self, model: torch.nn.Module):
        """Evaluate the model and store the results in a zarr file.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.
        """
        model.eval()
        self._initialize_zarr()
        self.gauge_data = {k: [] for k in self.gauge_data}
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
                    self._write_to_zarr(current_gauge)
                    self.gauge_data = {k: [] for k in self.gauge_data}  # clear memory for the new gauge

                # update current gauge
                current_gauge = batch_gauge_id

                # run model
                pred = model(sample)
                # process results from current batch
                self._process_results(sample, pred)
                # free memory
                del sample, pred

            # Write last gauge to disk
            if current_gauge is not None:
                self._write_to_zarr(current_gauge)

        zarr.consolidate_metadata(self.path_zarr)  # consolidate metadata to optimize read performance

    def validate_model(
        self,
        model: torch.nn.Module,
        epoch: int,
        filter_mask: xr.DataArray = None,
    ):
        """Validate the model every cfg.validate_every epochs and calculate the validation metric.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.
        epoch : int
            Current epoch number.
        forecast_mode : bool
            True if the dataset is from a forecast model (with lead_time dimension), False if from a simulation model.
        filter_mask : xr.DataArray, optional
            Boolean DataArray to filter values during evaluation. Expected dimensions (gauge_id, date).

        """
        if epoch % self.cfg.validate_every == 0:
            validation_time = time.time()
            # Model evaluation and results storage
            self.evaluate_model(model=model)
            # Calculate validation metrics
            validation_loss = calculate_metrics(
                ds_results=self.path_zarr,
                metric_name=self.cfg.validation_metric,
                forecast_mode=False if self.cfg.forecast_signals == [] else True,
                distribution=self.cfg.distribution,
                filter_mask=filter_mask,
            )
            if "lead_time" in validation_loss.dims:  # If applicable, calculate mean over lead_time
                validation_loss = validation_loss.mean(dim="lead_time")
            # Calculate median over remaining dimensions
            validation_loss = validation_loss.median(dim=[d for d in validation_loss.dims if d != "metric"])
            validation_loss = validation_loss.to_series().to_dict()

            metrics_str = "".join([f"{validation_loss[m]:^10.3f}|" for m in self.cfg.validation_metric])
            time_str = f"{str(datetime.timedelta(seconds=int(time.time() - validation_time))):^10}|"

            self.validation_report = metrics_str + time_str
        else:
            # Generate the correct number of empty columns for skipped epochs
            empty_metrics = "".join([f"{'':^10}|" for _ in range(len(self.cfg.validation_metric))])
            empty_time = f"{'':^10}|"
            self.validation_report = empty_metrics + empty_time

    def _create_date_range(self, evaluation_dataset):
        """Creates a date range that encompasses all possible observation dates in the evaluation dataset.

        Parameters
        ----------
        evaluation_dataset : BaseDataset
            The dataset used for evaluation, which contains the valid samples with their corresponding dates.

        Returns
        -------
        pd.DatetimeIndex
            A date range that covers all possible observation dates in the evaluation dataset, taking into account the
            emission dates and the forecast length.

        """
        freq = evaluation_dataset.data_freq

        # Get the min and max emission dates (init_time_fc)
        min_init = pd.Timestamp(evaluation_dataset.valid_samples["date"].min())
        max_init = pd.Timestamp(evaluation_dataset.valid_samples["date"].max())

        # Calculate offsets based on your predict_last_n and forecast logic
        # Max obs date is the last emission + the full forecast length
        max_obs_date = max_init + pd.to_timedelta(self.cfg.seq_length_forecast, unit=freq)
        # Min obs date might step backwards if predict_last_n > seq_length_forecast
        min_obs_date = min_init + pd.to_timedelta(self.cfg.seq_length_forecast - self.cfg.predict_last_n + 1, unit=freq)

        # Full range must encompass all emissions AND all observations
        return pd.date_range(start=min(min_init, min_obs_date), end=max(max_init, max_obs_date), freq=freq)

    def _initialize_zarr(self):
        raise NotImplementedError

    def _extract_y_obs(self, sample):
        return sample["y_obs"] * self.target_scaler["std"] + self.target_scaler["mean"]

    def _extract_y_sim(self, pred):
        return pred["y_hat"] * self.target_scaler["std"] + self.target_scaler["mean"]

    def _process_results(self, sample, y_sim, y_obs):
        raise NotImplementedError

    def _write_to_zarr(self, gauge_id):
        raise NotImplementedError
