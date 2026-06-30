import numpy as np
import xarray as xr
import zarr

from hy2dl.datasetzoo import BaseDataset
from hy2dl.evaluation.basetester import BaseTester
from hy2dl.modelzoo import get_model
from hy2dl.utils.config import Config


class HybridModelTester(BaseTester):
    """Class to process and store the results of hybrid models.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    evaluation_dataset : hy2dl.datasetzoo.basedataset.BaseDataset
        Dataset used for evaluation.

    """

    def __init__(self, cfg: Config, evaluation_dataset: BaseDataset):
        super(HybridModelTester, self).__init__(cfg=cfg, evaluation_dataset=evaluation_dataset)

        self.model = get_model(cfg)
        self.gauge_data = {"date": [], "y_obs": [], "y_sim": []}
        # Iterate through model parameters
        for param in self.model.conceptual_model.parameter_ranges:
            self.gauge_data[param] = []
        # Iterate through model states
        for state in self.model.conceptual_model._initial_states:
            self.gauge_data[state] = []

    def _initialize_zarr(self):
        """Creates the zarr structure to store the data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: id of gauge (basin)
            - date: date
            - feature: target variables
            - mixture_component: mixture density network components
        - Data Variables:
            - y_obs ("gauge_id", "date", "feature"): observed targets.
            - y_sim ("gauge_id", "date", "feature"): model predictions.
            - *parameters ("gauge_id", "date", "num_conceptual_models"): dynamic parameterization of conceptual model.
            - *states ("gauge_id", "date", "num_conceptual_models"): dynamic states of conceptual model.

        """
        base_shape = (len(self.unique_gauges), len(self.date_range), len(self.cfg.target))
        param_shape = (len(self.unique_gauges), len(self.date_range), self.cfg.num_conceptual_models)

        # Define data variables
        data_vars = {
            "y_obs": (
                ("gauge_id", "date", "feature"),
                np.full(base_shape, np.nan, dtype="float32"),
            ),
            "y_sim": (
                ("gauge_id", "date", "feature"),
                np.full(base_shape, np.nan, dtype="float32"),
            ),
        }

        for param in self.model.conceptual_model.parameter_ranges:
            data_vars[param] = (
                ("gauge_id", "date", "num_conceptual_models"),
                np.full(param_shape, np.nan, dtype="float32"),
            )

        for state in self.model.conceptual_model._initial_states:
            data_vars[state] = (
                ("gauge_id", "date", "num_conceptual_models"),
                np.full(param_shape, np.nan, dtype="float32"),
            )

        ds_template = xr.Dataset(
            data_vars=data_vars,
            coords={
                "gauge_id": self.unique_gauges,
                "date": self.date_range,
                "num_conceptual_models": np.arange(self.cfg.num_conceptual_models),
                "feature": self.cfg.target,
            },
        )

        # Chunk the dataset by id to optimize read/write operations
        ds_template = ds_template.chunk({"gauge_id": 1, "date": -1, "feature": -1, "num_conceptual_models": -1})

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
        self.gauge_data["date"].append(sample["date"].flatten())
        self.gauge_data["y_obs"].append(sample["y_obs"].cpu().flatten(start_dim=0, end_dim=1).numpy())
        self.gauge_data["y_sim"].append(pred["y_hat"].cpu().flatten(start_dim=0, end_dim=1).numpy())

        # Store parameters and states for each conceptual model
        for param in self.model.conceptual_model.parameter_ranges:
            self.gauge_data[param].append(pred["parameters"][param].cpu().flatten(start_dim=0, end_dim=1).numpy())

        for state in self.model.conceptual_model._initial_states:
            self.gauge_data[state].append(pred["internal_states"][state].cpu().flatten(start_dim=0, end_dim=1).numpy())

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

        # write observed, simulated values to zarr
        zarr_file = zarr.open(self.path_zarr, mode="r+")
        zarr_file["y_obs"][idx, :, :] = y_obs_filled
        zarr_file["y_sim"][idx, :, :] = y_sim_filled

        # Write parameters to zarr
        for param in self.model.conceptual_model.parameter_ranges:
            param_values = np.concatenate(self.gauge_data[param], axis=0)
            param_filled = np.full((len(self.date_range), self.cfg.num_conceptual_models), np.nan, dtype="float32")
            param_filled[date_indices[valid_], :] = param_values[valid_]
            zarr_file[param][idx, :, :] = param_filled

        # Write states to zarr
        for state in self.model.conceptual_model._initial_states:
            state_values = np.concatenate(self.gauge_data[state], axis=0)
            state_filled = np.full((len(self.date_range), self.cfg.num_conceptual_models), np.nan, dtype="float32")
            state_filled[date_indices[valid_], :] = state_values[valid_]
            zarr_file[state][idx, :, :] = state_filled
