import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class ForecastLSTM(nn.Module):
    """Forecast model that uses single LSTM layer.

    This class implements a LSTM based forecast model, in which a single LSTM cell rolls out
    through the hindcast and forecast period. Different embedding layers are used in each period.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        # Embedding networks
        self.embedding_hindcast = InputLayer(cfg)
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

    def forward(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Forward pass of lstm network

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        Dict[str, torch.Tensor]
            y_hat: Prediction for the `predict_last_n` time steps.

        """
        # Preprocess data for hindcast period
        x_lstm = self.embedding_hindcast(sample)

        # Preprocess data for forecast period
        x_fc = self.embedding_forecast(sample)

        # Concatenate both periods along the sequence dimension
        x_lstm = torch.cat((x_lstm, x_fc), dim=1)

        # Forward pass through the LSTM
        out, _ = self.lstm(x_lstm)
        # Extract sequence of interest
        out = out[:, -self.predict_last_n :, :]
        out = self.dropout(out)
        # Transform the output to the desired shape using a linear layer
        out = self.linear(out)

        return {"y_hat": out}
