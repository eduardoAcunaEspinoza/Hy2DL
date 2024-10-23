from typing import Dict, Union

import torch
import torch.nn as nn


class Hybrid(nn.Module):
    """Wrapper to combine a deep learning model with a conceptual hydrological models [#]_.

    Hybrid model in which a conceptual hydrological model is parameterized using a LSTM network.

    Parameters
    ----------
    model_configuration : Dict[str, Union[int, float, str, dict]]
        Configuration of the model

    References
    ----------
    .. [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket? 
        Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization, 
        Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    
    """

    def __init__(self, model_configuration: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        # General information for the model
        self.input_size_lstm = model_configuration["input_size_lstm"]
        self.hidden_size = model_configuration["hidden_size"]
        self.num_layers = model_configuration["no_of_layers"]
        self.seq_length = model_configuration["seq_length"]
        self.predict_last_n = model_configuration["predict_last_n"]

        # Warmup period is defined as the difference between seq_length and predict_last_n
        self.warmup_period = self.seq_length - self.predict_last_n

        # lstm
        self.lstm = nn.LSTM(
            input_size=self.input_size_lstm, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers
        )

        #  Conceptual model
        self.n_conceptual_models = model_configuration["n_conceptual_models"]
        self.conceptual_dynamic_parameterization = model_configuration["conceptual_dynamic_parameterization"]
        self.conceptual_model = model_configuration["conceptual_model"](
            n_models=self.n_conceptual_models, parameter_type=self.conceptual_dynamic_parameterization
        )
        self.n_conceptual_model_params = len(self.conceptual_model.parameter_ranges) * self.n_conceptual_models

        # Routing model
        if model_configuration.get("routing_model") is not None:
            self.routing_model = model_configuration["routing_model"]()
            self.n_routing_params = len(self.routing_model.parameter_ranges)
        else:
            self.n_routing_params = 0

        # Linear layer
        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_conceptual_model_params + self.n_routing_params
        )

    def forward(self, sample: Dict[str, torch.Tensor]):
        """Forward pass on hybrid model.

        In the forward pass, each element of the batch is associated with a basin. Therefore, the conceptual model is
        done to run multiple basins in parallel, and also multiple entities of the model at the same time.

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

        # Initialize hidden state with zeros
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

        # run LSTM
        lstm_out, _ = self.lstm(x_lstm, (h0, c0))
        lstm_out = self.linear(lstm_out)
        # map lstm output to parameters of conceptual model
        parameters_warmup, parameters_simulation = self.conceptual_model.map_parameters(
            lstm_out=lstm_out[:, :, : self.n_conceptual_model_params], warmup_period=self.warmup_period
        )
        # run conceptual model: warmup
        with torch.no_grad():
            pred = self.conceptual_model(
                x_conceptual=sample["x_conceptual"][:, : self.warmup_period, :], parameters=parameters_warmup
            )
        # run conceptual model: simulation
        pred = self.conceptual_model(
            x_conceptual=sample["x_conceptual"][:, self.warmup_period :, :],
            parameters=parameters_simulation,
            initial_states=pred["final_states"],
        )
        # Conceptual routing
        if self.n_routing_params > 0:
            _, parameters_simulation = self.routing_model.map_parameters(
                lstm_out=lstm_out[:, :, self.n_conceptual_model_params :], warmup_period=self.warmup_period
            )
            # apply routing routine
            pred["y_hat"] = self.routing_model(discharge=pred["y_hat"], parameters=parameters_simulation)

        return pred
