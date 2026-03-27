import math

import torch
import torch.nn as nn

from hy2dl.modelzoo.cudalstm import CudaLSTM
from hy2dl.utils import get_distribution
from hy2dl.utils.config import Config

PI = torch.tensor(math.pi)


class LSTMMDN(nn.Module):
    """LSTM with Mixture Density Network (MDN) head layer.

    This class implements an LSTM layer followed by a MDN head, which maps the hidden states produced by the LSTM into
    the parameters of a mixture distribution.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.

    Notation
    ----------
    - B: batch_size
    - L: seq_length
    - N: predict_last_n
    - K: num_mixture_components
    - T: num_targets
    - S: num_samples
    - Q: num_quantiles

    """

    def __init__(self, cfg: Config):
        super().__init__()

        # LSTM layer
        self.cudaLSTM = CudaLSTM(cfg)

        # Mix-density network head
        self.distribution = get_distribution(distribution=cfg.distribution)

        self.num_mixture_components = cfg.num_mixture_components
        self.output_features = cfg.output_features
        self.fc_params = nn.Linear(
            cfg.hidden_size, len(self.distribution.parameters) * cfg.num_mixture_components * cfg.output_features
        )
        self.fc_weights = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.num_mixture_components * cfg.output_features),
            nn.Unflatten(-1, (cfg.num_mixture_components, cfg.output_features)),
            nn.Softmax(dim=-2),
        )

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization.

        Initializes the mixture weights from -0.5 to 0.5 to avoid symmetry issues at the beginning of training.

        Parameters
        ----------
        cfg : Config
            Configuration object containing model hyperparameters and settings.
        """

        # mixture weights
        bias_init = torch.linspace(0.5, -0.5, cfg.num_mixture_components).repeat(cfg.output_features)
        self.fc_weights[0].bias.data = bias_init

    def forward(self, sample):
        """Forward pass of LSTM-MDN

        Processes hindcast features, and optionally concatenates forecast features along the sequence dimension, before
        passing them through the LSTM. The LSTM returns thethe parameters and weights of the mixture distribution of
        predictions.

        Parameters
        ----------
        sample: dict[str, Any]
            Dictionary with the different variables that will be used in the forward pass.
            See `hy2dl.datasetzoo.basedataset.Basedataset.__getitems__()` for details.

        Returns
        -------
        dict
            Dictionary containing:
            - 'y_hat': expected value of the mixture distribution, shape [B, N, T]
            - 'params': dict of distribution parameters [B, N, K, T]
            - 'weights': mixture weights of shape [B, N, K, T]

        """
        out = self.cudaLSTM(sample)

        # Probabilistic head layer
        params = self.distribution.map_parameters(
            raw_params=self.fc_params(out["hs"]),
            num_mixture_components=self.num_mixture_components,
            num_targets=self.output_features,
        )
        weights = self.fc_weights(out["hs"])
        # expected value of the distribution -> deterministic prediction
        y_hat = self.distribution.mean(params=params, weights=weights)

        return {"y_hat": y_hat, "params": params, "weights": weights}
