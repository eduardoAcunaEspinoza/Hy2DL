from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class CudaLSTM(nn.Module):
    """LSTM model supporting both hindcast and forecast modes.

    This class implements an LSTM layer followed by a linear head, which maps the hidden states produced by the LSTM
    into predictions. Depending on the configuration, it operates in either a standard mode (hindcast only) or a
    forecast mode. In forecast mode, the LSTM cell rolls out continuously through both the hindcast and forecast periods
    using specific embedding layers for each case.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
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
        self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.output_features)

        self.predict_last_n = cfg.predict_last_n
        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of certain model weights."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass of the LSTM network.

        Processes hindcast features, and optionally concatenates forecast features along the sequence dimension, before
        passing them through the LSTM and linear head.

        Parameters
        ----------
        sample: dict[str, Any]
            Dictionary with the different variables that will be used in the forward pass.
            See `hy2dl.datasetzoo.basedataset.Basedataset.__getitems__()` for details.

        Returns
        -------
        dict[str, torch.Tensor]
            y_hat: model predictions, shape (B, N, T)
            hs: hidden states of LSTM cell, shape (B, N, cfg.hidden_size)

        Notes
        -----
        Shape abbreviations used:
        - B: batch size
        - N: length of the target sequence, based on `predict_last_n` cofiguration argument
        - T: number of target variables

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
        # Transform the output to the desired shape using a linear layer
        out = self.linear(out)

        return {"y_hat": out, "hs": hs}
