from typing import Dict, Union

import torch
import torch.nn as nn


class CudaLSTM(nn.Module):
    """LSTM network.

    Parameters
    ----------
    model_configuration : Dict[str, Union[int, float, str, dict]]
        Configuration of the model

    """

    def __init__(self, model_configuration: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size_lstm = model_configuration["input_size_lstm"]
        self.hidden_size = model_configuration["hidden_size"]
        self.num_layers = model_configuration["no_of_layers"]
        self.predict_last_n = model_configuration["predict_last_n"]

        self.lstm = nn.LSTM(
            input_size=self.input_size_lstm, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers
        )

        self.dropout = torch.nn.Dropout(model_configuration["dropout_rate"])
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=model_configuration.get("out_features", 1))

    def forward(self, sample: Dict[str, torch.Tensor]):
        """Forward pass of lstm network

        Parameters
        ----------
        sample: Dict[str, torch.Tensor]
            Dictionary with the different tensors that will be used for the forward pass.

        Returns
        -------
        pred: Dict[str, torch.Tensor]

        """
        # Dynamic input
        x_lstm = sample["x_d"]

        # Concatenate static
        if sample.get("x_s") is not None:
            x_lstm = torch.cat((x_lstm, sample["x_s"].unsqueeze(1).repeat(1, x_lstm.shape[1], 1)), dim=2)

        h0 = torch.zeros(
            self.num_layers,
            x_lstm.shape[0],
            self.hidden_size,
            requires_grad=True,
            dtype=torch.float32,
            device=x_lstm.device,
        )
        c0 = torch.zeros(
            self.num_layers,
            x_lstm.shape[0],
            self.hidden_size,
            requires_grad=True,
            dtype=torch.float32,
            device=x_lstm.device,
        )

        out, (hn_1, cn_1) = self.lstm(x_lstm, (h0, c0))
        out = out[:, -self.predict_last_n :, :]
        out = self.dropout(out)
        out = self.linear(out)

        return {"y_hat": out}
