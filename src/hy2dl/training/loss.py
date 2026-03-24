from typing import Any

import torch
import torch.nn as nn

from hy2dl.utils import get_distribution
from hy2dl.utils.config import Config


class BaseLoss(nn.Module):
    """Abstract base class to ensure all losses use the same format in the forward pass"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg  # Store it in case a subclass needs it later

    def forward(self, pred: dict[str, torch.Tensor], sample: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class NLL(BaseLoss):
    """Negative log-likelihood.

    Calculate negative log-likelihood i.e. the log probability of `y_obs` given a mixture distribution, applying an
    optional weight to each target variable.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.distribution = get_distribution(cfg)

        if cfg.target_weights is not None:
            self.target_weights = torch.tensor(cfg.target_weights, dtype=torch.float32, device=cfg.device)
        else:
            self.target_weights = torch.ones(len(cfg.target), dtype=torch.float32, device=cfg.device)

    def forward(self, pred: dict[str, torch.Tensor], sample: dict[str, Any]) -> torch.Tensor:
        log_p = self.distribution.calc_logpdf(params=pred["params"], weights=pred["weights"], x=sample["y_obs"])
        nll = -log_p.mean(dim=(0, 1))
        return torch.dot(nll, self.target_weights)


class NSEBasinAveraged(BaseLoss):
    """Basin-averaged Nash--Sutcliffe Efficiency.

    Loss function where the squared errors are weighed by the standard deviation of each basin. A description of this
    function is available at [#]_.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
    universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
    Hydrology and Earth System Sciences, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019

    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def forward(self, pred: dict[str, torch.Tensor], sample: dict[str, Any]) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : dict[str, torch.Tensor]
            Model predictions
        sample: dict[str, Any]
            Dictionary containing observed targets and other sample information

        Returns
        -------
        torch.Tensor
            Value of the basin-averaged NSE

        """
        # Extract variables of interest
        y_sim = pred["y_hat"]
        y_obs = sample["y_obs"]
        basin_std = sample["std_basin"].unsqueeze(1).expand(-1, y_obs.shape[1], -1)  # broadcast std (B, S, N)

        # calculate mask to avoid nan in observation to affect the loss
        mask = ~torch.isnan(y_obs)

        # Filter nans
        y_sim_masked = y_sim[mask]
        y_obs_masked = y_obs[mask]
        basin_std_masked = basin_std[mask]

        # Calculate loss
        squared_error = (y_sim_masked - y_obs_masked) ** 2
        weights = 1 / (basin_std_masked + 0.1) ** 2  # 0.1 is a small constant for numerical stability
        loss = weights * squared_error

        return torch.mean(loss)


class WeightedMSE(BaseLoss):
    """Weighted Mean Squared Error.

    Calculates the MSE between simulated and observed targets, applying an optional weight to each target variable.

    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        if cfg.target_weights is not None:
            self.target_weights = torch.tensor(cfg.target_weights, dtype=torch.float32, device=cfg.device)
        else:
            self.target_weights = torch.ones(len(cfg.target), dtype=torch.float32, device=cfg.device)

    def forward(self, pred: dict[str, torch.Tensor], sample: dict[str, Any]) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : dict[str, torch.Tensor]
            Model predictions
        sample: dict[str, Any]
            Dictionary containing observed targets and other sample information

        Returns
        -------
        torch.Tensor
            Value of the weighted MSE.

        """
        # Extract variables of interest
        y_sim = pred["y_hat"]
        y_obs = sample["y_obs"]
        target_weights = self.target_weights.unsqueeze(0).unsqueeze(0).expand_as(y_obs)

        # calculate mask to avoid nan in observation to affect the loss
        mask = ~torch.isnan(y_obs)

        # Filter nans
        y_sim_masked = y_sim[mask]
        y_obs_masked = y_obs[mask]
        target_weights_masked = target_weights[mask]

        # Calculate loss
        squared_error = (y_sim_masked - y_obs_masked) ** 2
        weighted_squared_error = target_weights_masked * squared_error

        return torch.mean(weighted_squared_error)
