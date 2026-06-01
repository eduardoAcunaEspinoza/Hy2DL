from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class CudaLSTM(nn.Module):
    """LSTM model.

    This class implements an LSTM layer. If config.model == "cudalstm", the LSTM layer is followed by a linear head,
    which maps the hidden states produced by the LSTM into predictions. Otherwise, it is assumed the model is being used
    as part of a larger architecture, and only the hidden states are returned.

    The LSTM layer can operate either in a standard mode (hindcast only) or forecast mode. In forecast mode, the LSTM
    cell rolls out continuously through both the hindcast and forecast periods using specific embedding layers for each
    case.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        # Embedding network hindcast period
        self.emb_hc = InputLayer(cfg)

        # Configurations from forecast period, if neccesary
        self.forecast_mode = False
        self.forecast_counter = 0
        if cfg.forecast_signals:
            self.forecast_mode = True
            self.emb_fc = InputLayer(cfg, embedding_type="forecast")

            if cfg.forecast_counter:
                self.forecast_counter = 1
                fc_counter = torch.zeros(
                    self.emb_hc.input_seq_length + self.emb_fc.input_seq_length, dtype=torch.float32
                )
                fc_counter[self.emb_hc.input_seq_length :] = torch.arange(
                    1, self.emb_fc.input_seq_length + 1, dtype=torch.float32
                )
                self.register_buffer("fc_counter", fc_counter)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.emb_hc.output_size + self.forecast_counter, hidden_size=cfg.hidden_size, batch_first=True
        )

        self.dropout = torch.nn.Dropout(p=cfg.dropout_rate)

        # Add linear head if the LSTM layer is used as a standalone model.
        if cfg.model.lower() == "cudalstm":
            self.cudalstm = True
            self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.output_features)
        else:
            self.cudalstm = False

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
        # Data for hindcast period
        x_lstm = self.emb_hc(sample)

        # Data for forecast period, if specified
        if self.forecast_mode:
            x_fc = self.emb_fc(sample)
            x_lstm = torch.cat((x_lstm, x_fc), dim=1)

            if self.forecast_counter == 1:
                x_lstm = torch.cat((x_lstm, self.fc_counter.view(1, -1, 1).expand(x_lstm.shape[0], -1, -1)), dim=2)

        # Forward pass through the LSTM
        hs, _ = self.lstm(x_lstm)
        # Extract sequence of interest
        hs = hs[:, -self.predict_last_n :, :]
        hs = self.dropout(hs)

        return {"y_hat": self.linear(hs), "hs": hs} if self.cudalstm else {"hs": hs}
