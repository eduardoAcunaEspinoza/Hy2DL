import math

import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
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

        # Embedding network hindcast period
        self.embedding_hindcast = InputLayer(cfg)

        # Embedding network forecast period, if neccesary
        self.forecast_mode = False
        if cfg.forecast_signals:
            self.forecast_mode = True
            self.embedding_forecast = InputLayer(cfg, embedding_type="forecast")

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_hindcast.output_size, hidden_size=cfg.hidden_size, batch_first=True
        )
        self.dropout = torch.nn.Dropout(p=cfg.dropout_rate)

        # Mix-density network head
        self.distribution = get_distribution(cfg)

        self.fc_params = nn.Linear(
            cfg.hidden_size, self.distribution.num_params * cfg.num_mixture_components * cfg.output_features
        )
        self.fc_weights = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.num_mixture_components * cfg.output_features),
            nn.Unflatten(-1, (cfg.num_mixture_components, cfg.output_features)),
            nn.Softmax(dim=-2),
        )

        self.predict_last_n = cfg.predict_last_n
        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization.

        Sets the initial forget gate bias to a specified value (if any). Initializes the mixture weights from
        -0.5 to 0.5 to avoid symmetry issues at the beginning of training.

        Parameters
        ----------
        cfg : Config
            Configuration object containing model hyperparameters and settings.
        """
        # forget gate lstm
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias
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
            - 'params': dict of distribution parameters [B, N, K, T]
            - 'weights': mixture weights of shape [B, N, K, T]

        """
        # Preprocess data for hindcast period
        x_lstm = self.embedding_hindcast(sample)

        if self.forecast_mode:
            x_fc = self.embedding_forecast(sample)
            x_lstm = torch.cat((x_lstm, x_fc), dim=1)

        # Forward pass through the LSTM
        hs, _ = self.lstm(x_lstm)
        # Extract sequence of interest
        hs = hs[:, -self.predict_last_n :, :]
        out = self.dropout(hs)

        # Probabilistic head layer
        params = self.distribution.map_parameters(raw_params=self.fc_params(out))
        weights = self.fc_weights(out)

        return {"params": params, "weights": weights}
