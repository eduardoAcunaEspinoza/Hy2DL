import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class CudaLSTM(nn.Module):
    """LSTM model class, which relies on PyTorch's CUDA LSTM class.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size, batch_first=True)

        self.dropout = torch.nn.Dropout(p=cfg.dropout_rate)

        self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.output_features)

        self.predict_last_n = cfg.predict_last_n

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of certain model weights."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]])-> dict[str, torch.Tensor]:
        """Forward pass of lstm network

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        pred: Dict[str, torch.Tensor]

        """
        # Preprocess data to be sent to the LSTM
        processed_sample = self.embedding_net(sample)
        x_lstm = self.embedding_net.assemble_sample(processed_sample)
        
        # Forward pass through the LSTM
        out, _ = self.lstm(x_lstm)
        # Extract sequence of interest
        out = out[:, -self.predict_last_n :, :]
        out = self.dropout(out)
        # Transform the output to the desired shape using a linear layer
        out = self.linear(out)

        return {"y_hat": out}