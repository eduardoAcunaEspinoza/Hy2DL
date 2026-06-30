import numpy as np
import xarray as xr
import zarr

from hy2dl.datasetzoo import BaseDataset
from hy2dl.evaluation.basetester import BaseTester
from hy2dl.utils import get_distribution
from hy2dl.utils.config import Config


class ForecastTesterMDN(BaseTester):
    """Class to process and store the results of Mixture Density Network forecast models.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    evaluation_dataset : hy2dl.datasetzoo.basedataset.BaseDataset
        Dataset used for evaluation.

    """

    def __init__(self, cfg: Config, evaluation_dataset: BaseDataset):
        super(ForecastTesterMDN, self).__init__(cfg=cfg, evaluation_dataset=evaluation_dataset)
        self.distribution = get_distribution(distribution=cfg.distribution)
        self.num_mixture_components = cfg.num_mixture_components
        self.gauge_data = {"date": [], "y_obs": [], "y_sim": [], "init_time_fc": [], "mdn_weight": []}
        for param in self.distribution.parameters:
            self.gauge_data[param] = []

    def _initialize_zarr(self):
        """Creates the zarr structure to store the data.

        Zarr structure:
        - Dimensions / Coordinates:
            - gauge_id: id of gauge (basin)
            - date: date
            - lead_time: forecast lead time
            - mixture_component: mixture density network components
            - feature: target variables
        - Data Variables:
            - y_obs ("gauge_id", "date", "feature"): observed targets.
            - y_sim ("gauge_id", "date", "lead_time", "feature"): model predictions..
            - mixture_component ("gauge_id", "date", "lead_time", "num_mixture_components", "feature"): mixture weights
            for each component of the mixture density network.
            - *parameters "gauge_id", "date", "lead_time", "num_mixture_components", "feature"): distribution parameter
            of each component of the mixture density network.

        """
        # Calculate the lead times based on seq_length_forecast and predict_last_n
        start_lead_time = self.cfg.seq_length_forecast - self.cfg.predict_last_n + 1
        lead_times = np.arange(start_lead_time, self.cfg.seq_length_forecast + 1)

        obs_shape = (len(self.unique_gauges), len(self.date_range), len(self.cfg.target))
        sim_shape = (len(self.unique_gauges), len(self.date_range), self.cfg.predict_last_n, len(self.cfg.target))
        mdn_shape = (
            len(self.unique_gauges),
            len(self.date_range),
            self.cfg.predict_last_n,
            self.num_mixture_components,
            len(self.cfg.target),
        )

        # Define data variables
        data_vars = {
            "y_obs": (
                ("gauge_id", "date", "feature"),
                np.full(obs_shape, np.nan, dtype="float32"),
            ),
            "y_sim": (
                ("gauge_id", "date", "lead_time", "feature"),
                np.full(sim_shape, np.nan, dtype="float32"),
            ),
            "mdn_weight": (
                ("gauge_id", "date", "lead_time", "mixture_component", "feature"),
                np.full(mdn_shape, np.nan, dtype="float32"),
            ),
        }
        # Add distribution-specific parameters to data variables
        for param in self.distribution.parameters:
            data_vars[param] = (
                ("gauge_id", "date", "lead_time", "mixture_component", "feature"),
                np.full(mdn_shape, np.nan, dtype="float32"),
            )

        ds_template = xr.Dataset(
            data_vars=data_vars,
            coords={
                "gauge_id": self.unique_gauges,
                "date": self.date_range,
                "lead_time": lead_times,
                "mixture_component": np.arange(self.num_mixture_components),
                "feature": self.cfg.target,
            },
        )

        # Chunk the dataset. Note we include lead_time in the chunking.
        ds_template = ds_template.chunk(
            {"gauge_id": 1, "date": -1, "lead_time": -1, "mixture_component": -1, "feature": -1}
        )
        ds_template.to_zarr(self.path_zarr, compute=False, mode="w", consolidated=False)

    def _process_results(self, sample, pred):
        """Processes the evaluation results for a batch of samples and stores them in the gauge_data dictionary"""
        # extract variables of interest
        y_sim = self._extract_y_sim(pred)
        y_obs = self._extract_y_obs(sample)
        mixture_weights = pred["weights"]
        parameters = self.distribution.backtransform_parameters(params=pred["params"], target_scaler=self.target_scaler)

        self.gauge_data["date"].append(sample["date"].flatten())
        self.gauge_data["y_obs"].append(y_obs.cpu().flatten(start_dim=0, end_dim=1).numpy())
        self.gauge_data["y_sim"].append(y_sim.cpu().numpy())
        self.gauge_data["init_time_fc"].append(sample["init_time_fc"])
        self.gauge_data["mdn_weight"].append(mixture_weights.cpu().numpy())
        for param in self.distribution.parameters:
            self.gauge_data[param].append(parameters[param].cpu().numpy())

    def _write_to_zarr(self, gauge_id):
        """Writes the results of a gauge to the zarr file."""
        idx = self.gauge_idx[gauge_id]

        # concatenate gauge data
        dates = np.concatenate(self.gauge_data["date"], axis=0)
        y_obs = np.concatenate(self.gauge_data["y_obs"], axis=0)
        y_sim = np.concatenate(self.gauge_data["y_sim"], axis=0)
        init_time_fc = np.concatenate(self.gauge_data["init_time_fc"], axis=0)
        mdn_weights = np.concatenate(self.gauge_data["mdn_weight"], axis=0)

        # avoid issues with missing dates by creating a full timeline and filling in the available data
        y_obs_filled = np.full((len(self.date_range), len(self.cfg.target)), np.nan, dtype="float32")
        y_sim_filled = np.full(
            (len(self.date_range), self.cfg.predict_last_n, len(self.cfg.target)), np.nan, dtype="float32"
        )
        mdn_weights_filled = np.full(
            (len(self.date_range), self.cfg.predict_last_n, self.num_mixture_components, len(self.cfg.target)),
            np.nan,
            dtype="float32",
        )

        # Map observed values to their date-index. Overlapping forecast windows will overwrite themselves
        obs_indices = self.date_range.get_indexer(dates)
        valid_obs = obs_indices >= 0
        y_obs_filled[obs_indices[valid_obs], :] = y_obs[valid_obs]

        # Map simulation values to their date-index
        sim_indices = self.date_range.get_indexer(init_time_fc)
        valid_sim = sim_indices >= 0
        y_sim_filled[sim_indices[valid_sim], :, :] = y_sim[valid_sim]
        mdn_weights_filled[sim_indices[valid_sim], :, :, :] = mdn_weights[valid_sim]

        # write observed, simulated and mdn weights to zarr
        zarr_file = zarr.open(self.path_zarr, mode="r+")
        zarr_file["y_obs"][idx, :, :] = y_obs_filled
        zarr_file["y_sim"][idx, :, :, :] = y_sim_filled
        zarr_file["mdn_weight"][idx, :, :, :, :] = mdn_weights_filled

        # Write distribution parameters to zarr
        for param in self.distribution.parameters:
            param_values = np.concatenate(self.gauge_data[param], axis=0)
            param_filled = np.full(
                (len(self.date_range), self.cfg.predict_last_n, self.num_mixture_components, len(self.cfg.target)),
                np.nan,
                dtype="float32",
            )
            # Safely explicitly using the simulation indices
            param_filled[sim_indices[valid_sim], :, :, :] = param_values[valid_sim]
            zarr_file[param][idx, :, :, :, :] = param_filled
