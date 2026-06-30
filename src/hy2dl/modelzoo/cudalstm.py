from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo import get_head
from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class CudaLSTM(nn.Module):
    """LSTM model.

    This class implements Pytorch's cuda-optimized LSTM model (nn.LSTM).

    The LSTM layer can operate either in a standard mode (hindcast only) or forecast mode. In forecast mode, the model
    implementes a sequential-forecast framework [1]_ rolling continuously through both the hindcast and forecast periods,
    using specific embedding layers for each case .

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.

    References
    ----------
    .. [1] Cohen, D., Amira, R., Aschner, R., Carny, Y., Feinstein, B., Fester, H., Fronman, S., Gauch, M., Gilon, O.,
        Green, R., Hassidim, A., Klotz, D., Kratzert, F., Korenfeld, D., Loike, G., Markel, A., Matias, Y., Mayo, R.,
        Metzger, A., . . . Nearing, G. (2026). Extending medium-range global flood forecasts: The Google Global Flood
        Forecasting Model Version 2. EGUsphere, 2026, 1–31. https://doi.org/10.5194/egusphere-2026-2283
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

        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.predict_last_n = cfg.predict_last_n

        # Head layer
        self.head = get_head(cfg=cfg)

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of certain model weights."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass of the LSTM network.

        Parameters
        ----------
        sample: dict[str, Any]
            Dictionary with the different variables that will be used in the forward pass.
            See `hy2dl.datasetzoo.basedataset.Basedataset.__getitems__()` for details.

        Returns
        -------
        dict[str, torch.Tensor]
            Specific output of the model, depending on the head layer used. See `hy2dl.modelzoo.head` for details.

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

        return self.head(hs)
