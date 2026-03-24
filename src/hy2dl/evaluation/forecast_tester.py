import numpy as np
import torch
import xarray as xr
import zarr

from hy2dl.datasetzoo import BaseDataset
from hy2dl.evaluation.basetester import BaseTester
from hy2dl.utils.config import Config


class ForecastTester(BaseTester):
    """Class to process and store the results of forecast models.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    evaluation_dataset : BaseDataset
        Dataset used for evaluation.

    """

    def __init__(self, cfg: Config, evaluation_dataset: BaseDataset):
        super(ForecastTester, self).__init__(cfg=cfg, evaluation_dataset=evaluation_dataset)
        self.gauge_data = {"init_time_fc": [], "y_sim": [], "date": [], "y_obs": []}

        if self.cfg.seq_length_forecast != self.cfg.predict_last_n:
            raise ValueError(
                "Given how we store the forecast evaluation data, `seq_length_forecast` and `predict_last_n`"
                "must be the same length."
            )

    def validate_model(self, model: torch.nn.Module, epoch: int, filter_mask: xr.DataArray = None):
        self._validate_model(model=model, epoch=epoch, forecast_mode=True, filter_mask=filter_mask)

    def _initialize_zarr(self):
        """Creates the zarr structure to store the data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: id of gauge (basin)
            - date: date
            - lead_time: forecast lead time
            - feature: target variables
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

    def _extract_y_obs(self, sample):
        return sample["y_obs"] * self.target_std + self.target_mean

    def _extract_y_sim(self, pred):
        return pred["y_hat"] * self.target_std + self.target_mean

    def _process_results(self, sample, y_sim, y_obs):
        """ "Processes the evaluation results for a batch of samples and stores them in the gauge_data dictionary."""
        self.gauge_data["init_time_fc"].append(sample["init_time_fc"])
        self.gauge_data["y_sim"].append(y_sim.cpu().numpy())
        self.gauge_data["date"].append(sample["date"].flatten())
        self.gauge_data["y_obs"].append(y_obs.cpu().flatten(start_dim=0, end_dim=1).numpy())

    def _write_to_zarr(self, gauge_id):
        """Writes the results of a gauge to the zarr file."""
        idx = self.gauge_idx[gauge_id]

        # concatenate gauge data
        init_time_fc = np.concatenate(self.gauge_data["init_time_fc"], axis=0)
        y_sim = np.concatenate(self.gauge_data["y_sim"], axis=0)

        dates = np.concatenate(self.gauge_data["date"], axis=0)
        y_obs = np.concatenate(self.gauge_data["y_obs"], axis=0)

        # avoid issues with missing dates by creating a full timeline and filling in the available data
        y_sim_filled = np.full(
            (len(self.date_range), self.cfg.seq_length_forecast, len(self.cfg.target)), np.nan, dtype="float32"
        )
        y_obs_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")

        # Map simulation values to their date-index
        date_indices = self.date_range.get_indexer(init_time_fc)
        valid_ = date_indices >= 0
        y_sim_filled[date_indices[valid_], :, :] = y_sim[valid_]

        # Map observed values to their date-index. Overlapping forecast windows will overwrite themselves
        date_indices = self.date_range.get_indexer(dates)
        valid_ = date_indices >= 0
        y_obs_filled[date_indices[valid_], :] = y_obs[valid_]

        # write to zarr
        zarr_file = zarr.open(self.path_zarr, mode="r+")
        zarr_file["y_sim"][idx, :, :, :] = y_sim_filled
        zarr_file["y_obs"][idx, :, :] = y_obs_filled
