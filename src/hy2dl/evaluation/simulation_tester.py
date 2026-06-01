import numpy as np
import xarray as xr
import zarr

from hy2dl.datasetzoo import BaseDataset
from hy2dl.evaluation.basetester import BaseTester
from hy2dl.utils.config import Config


class SimulationTester(BaseTester):
    """Class to process and store the results of simulation models.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    evaluation_dataset : BaseDataset
        Dataset used for evaluation.

    """

    def __init__(self, cfg: Config, evaluation_dataset: BaseDataset):
        super(SimulationTester, self).__init__(cfg=cfg, evaluation_dataset=evaluation_dataset)
        self.gauge_data = {"date": [], "y_obs": [], "y_sim": []}

    def _initialize_zarr(self):
        """Creates the zarr structure to store the data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: id of gauge (basin)
            - date: date
            - feature: target variables
        - Data Variables:
            - y_obs: observed targets.
            - y_sim: model predictions.

        """
        ds_template = xr.Dataset(
            data_vars={
                "y_obs": (
                    ("gauge_id", "date", "feature"),
                    np.full(
                        (len(self.unique_gauges), len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32"
                    ),
                ),
                "y_sim": (
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

    def _process_results(self, sample, pred):
        """ "Processes the evaluation results for a batch of samples and stores them in the gauge_data dictionary.

        The function resolve the sequence dimension (when predict_last_n>1) to build a continuous timeline.
            - self.cfg.unique_prediction_blocks = True (Non-overlapping blocks): Flatten batch and sequence dimensions
            together.
            - self.cfg.unique_prediction_blocks = False (Overlapping blocks): Keep only the final timestep to prevent
            duplicate dates.

        Note: If predict_last_n == 1, both options yield the same result.

        """
        # extract variables of interest
        y_sim = self._extract_y_sim(pred)
        y_obs = self._extract_y_obs(sample)

        if self.cfg.unique_prediction_blocks:
            self.gauge_data["date"].append(sample["date"].flatten())
            self.gauge_data["y_obs"].append(y_obs.cpu().flatten(start_dim=0, end_dim=1).numpy())
            self.gauge_data["y_sim"].append(y_sim.cpu().flatten(start_dim=0, end_dim=1).numpy())
        else:
            self.gauge_data["date"].append(sample["date"][:, -1])
            self.gauge_data["y_obs"].append(y_obs[:, -1, :].cpu().numpy())
            self.gauge_data["y_sim"].append(y_sim[:, -1, :].cpu().numpy())

    def _write_to_zarr(self, gauge_id):
        """Writes the results of a gauge to the zarr file."""
        idx = self.gauge_idx[gauge_id]

        # concatenate gauge data
        dates = np.concatenate(self.gauge_data["date"], axis=0)
        y_obs = np.concatenate(self.gauge_data["y_obs"], axis=0)
        y_sim = np.concatenate(self.gauge_data["y_sim"], axis=0)

        # avoid issues with missing dates by creating a full timeline and filling in the available data
        y_obs_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")
        y_sim_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")
        date_indices = self.date_range.get_indexer(dates)
        valid_ = date_indices >= 0
        y_obs_filled[date_indices[valid_], :] = y_obs[valid_]
        y_sim_filled[date_indices[valid_], :] = y_sim[valid_]

        # write to zarr
        zarr_file = zarr.open(self.path_zarr, mode="r+")
        zarr_file["y_obs"][idx, :, :] = y_obs_filled
        zarr_file["y_sim"][idx, :, :] = y_sim_filled
