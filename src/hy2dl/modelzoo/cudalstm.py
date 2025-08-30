import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class CudaLSTM(nn.Module):
    """LSTM model.

    This class implements an LSTM layer followed by a linear head, which maps the
    hidden states produced by the LSTM into predictions for the specified time steps.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        # Embedding network
        self.embedding_hindcast = InputLayer(cfg)

        # Pytorch implementation of LSTM
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

        # Forward pass through the LSTM
        hs, _ = self.lstm(x_lstm)
        # Extract sequence of interest
        hs = hs[:, -self.predict_last_n :, :]
        out = self.dropout(hs)
        # Transform the output to the desired shape using a linear layer
        out = self.linear(out)

        return {"y_hat": out, "hs": hs}
