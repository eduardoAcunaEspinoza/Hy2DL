from typing import Dict, Union

import torch
import torch.nn as nn


class MFLSTM(nn.Module):
    """Multi-frequency LSTM network.

    Parameters
    ----------
    model_configuration : Dict[str, Union[int, float, str, dict]]
        Configuration of the model

    References
    ----------

    """

    def __init__(self, model_configuration: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size_lstm = model_configuration["input_size_lstm"]
        self.hidden_size = model_configuration["hidden_size"]
        self.num_layers = model_configuration["no_of_layers"]
        self.predict_last_n = model_configuration["predict_last_n"]
        self.custom_freq_processing = model_configuration["custom_freq_processing"]

        # In case we use different embeddings for the different frequencies
        self.embedding_net = self._get_embedding_net(model_configuration)

        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=self.input_size_lstm, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers
        )

        # Dropout and linear layer
        self.dropout = torch.nn.Dropout(model_configuration["dropout_rate"])
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=model_configuration.get("out_features", 1))

    @staticmethod
    def _get_embedding_net(model_configuration: Dict[str, Union[int, float, str, dict]]) -> Union[None, nn.ModuleDict]:
        # In case we what to use dynamic_embeddings for the different frequencies
        if model_configuration.get("dynamic_embeddings"):
            embedding_net = nn.ModuleDict()
            for freq in model_configuration["custom_freq_processing"].keys():
                if isinstance(model_configuration["dynamic_input_size"], int):
                    input_size = model_configuration["dynamic_input_size"]
                else:
                    input_size = model_configuration["dynamic_input_size"][freq]

                # Linear layer to map from an input size to a predefined number of channels
                embedding_net[freq] = nn.Linear(
                    in_features=input_size, out_features=model_configuration["n_dynamic_channels_lstm"]
                )
        else:
            embedding_net = None

        return embedding_net

    def forward(self, sample: Dict[str, torch.Tensor]):
        """Forward pass

        Parameters
        ----------
        sample: Dict[str, torch.Tensor]
            Dictionary with the different tensors that will be used for the forward pass.

        Returns
        -------
        pred: Dict[str, torch.Tensor]

        """
        process_tensor = []
        # Process the different frequencies
        for freq in self.custom_freq_processing.keys():
            x = sample["x_d_" + freq]
            if self.embedding_net:  # In case we use embedding for the different frequencies
                x = self.embedding_net[freq](x)
            process_tensor.append(x)

        x_lstm = torch.cat(process_tensor, dim=1)

        # Concatenate static attributes
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
