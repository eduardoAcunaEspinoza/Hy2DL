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
from hy2dl.evaluation.metrics import calculate_metrics
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
        self.path_zarr = self.cfg.path_save_folder / f"{evaluation_dataset.period}_results.zarr"
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
                # extract variables of interest
                y_sim = self._extract_y_sim(pred)
                y_obs = self._extract_y_obs(sample)
                # process results from current batch
                self._process_results(sample, y_sim, y_obs)
                # free memory
                del sample, pred, y_sim, y_obs

            # Write last gauge to disk
            if current_gauge is not None:
                self._write_to_zarr(current_gauge)

        zarr.consolidate_metadata(self.path_zarr)  # consolidate metadata to optimize read performance

    def _validate_model(
        self, model: torch.nn.Module, epoch: int, forecast_mode: bool, filter_mask: xr.DataArray = None
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
            self.evaluate_model(model=model)
            validation_loss = calculate_metrics(
                ds_results=self.path_zarr,
                metrics=self.cfg.validation_metric,
                forecast_mode=forecast_mode,
                filter_mask=filter_mask,
                collapse=True,
            )
            self.validation_report = (
                f"{validation_loss:^10.3f}|{str(datetime.timedelta(seconds=int(time.time() - validation_time))):^10}|"
            )
        else:
            self.validation_report = f"{'':^10}|{'':^10}|"

    def _initialize_zarr(self):
        raise NotImplementedError

    def _extract_y_obs(self, sample):
        raise NotImplementedError

    def _extract_y_sim(self, pred):
        raise NotImplementedError

    def _process_results(self, sample, y_sim, y_obs):
        raise NotImplementedError

    def _write_to_zarr(self, gauge_id):
        raise NotImplementedError
