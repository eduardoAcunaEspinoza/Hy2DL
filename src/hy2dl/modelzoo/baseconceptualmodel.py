from typing import Optional

import torch
import torch.nn as nn

from hy2dl.utils.config import Config


class BaseConceptualModel(nn.Module):
    """Abstract base model class

    The purpose is to have some common operations that all conceptual models will need. Don't use this class for model
    training!

    """

    def __init__(self):
        super(BaseConceptualModel, self).__init__()

    def forward(
        self,
        x_conceptual: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        raise NotImplementedError

    def map_parameters(
        self, lstm_out: torch.Tensor, warmup_period: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Map output of data-driven part to predefined ranges of the conceptual model parameters.

        The result are two dictionaries, one contains the parameters for the warmup period of the conceptual model and
        the other contains the parameters for the simulation period. Moreover, the parameterization can be static or
        dynamic. In the static parameterization the last value is repeated over the whole timeseries, while in the
        dynamic parameterization we have one parameter set for each time step.

        Note:
            The dynamic parameterization only occurs in the simulation phase, not the warmup! The warmup always uses
            static parameterization. Therefore, in case we specified dynamic parameterization, for the warmup period,
            we take the last value of this period and repeat it throughout the warmup phase.

        Parameters
        ----------
        lstm_out : torch.Tensor
            Tensor of size [batch_size, time_steps, n_param] that will be mapped to the predefined ranges of the
            conceptual model parameters to act as the dynamic parameterization.
        warmup_period : int
            Number of timesteps (e.g. days) to warmup the internal states of the conceptual model

        Returns
        -------
        Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            - parameters_warmup : dict[str, torch.Tensor]
                Parameters for the warmup period (always static!)
            - parameters_simulation : dict[str, torch.Tensor]
                Parameterization of the conceptual model in the training/testing period. Can be static or dynamic

        """
        # Reshape tensor to consider multiple conceptual models running in parallel.
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], -1, self.n_conceptual_models)

        parameters_warmup = {}
        parameters_simulation = {}
        for index, (parameter_name, parameter_range) in enumerate(self.parameter_ranges.items()):
            range_t = torch.tensor(parameter_range, dtype=torch.float32, device=lstm_out.device)
            range_t = range_t.unsqueeze(dim=1).unsqueeze(dim=2)

            if self.parameter_type[parameter_name] == "static":
                # If parameter is static, take the last value predicted by the lstm and copy it for all the timesteps.
                warmup_lstm_out = lstm_out[:, -1:, index, :].expand(-1, warmup_period, -1)
                simulation_lstm_out = lstm_out[:, -1:, index, :].expand(-1, lstm_out.shape[1] - warmup_period, -1)
            elif self.parameter_type[parameter_name] == "dynamic":
                warmup_lstm_out = lstm_out[:, warmup_period - 1 : warmup_period, index, :].expand(-1, warmup_period, -1)
                simulation_lstm_out = lstm_out[:, warmup_period:, index, :]
            else:
                raise ValueError(f"Unsupported parameter type {self.parameter_type[parameter_name]}")

            parameters_warmup[parameter_name] = range_t[:1, :, :] + torch.sigmoid(warmup_lstm_out) * (
                range_t[1:, :, :] - range_t[:1, :, :]
            )

            parameters_simulation[parameter_name] = range_t[:1, :, :] + torch.sigmoid(simulation_lstm_out) * (
                range_t[1:, :, :] - range_t[:1, :, :]
            )

        return parameters_warmup, parameters_simulation

    def _map_parameter_type(self, cfg: Config) -> dict[str, str]:
        """Define parameter type, static or dynamic.

        The model parameters can be static or dynamic. This function creates a dictionary that associate the parameter
        name with a type specified by the user. In case the user did not specify a type, the parameter_type is
        automatically specified as static.

        Parameters
        ----------
        cfg : Config
            Configuration file.

        Returns
        -------
        map_parameter_type: dict[str, str]
            dictionary

        """
        map_parameter_type = {}
        for key in self.parameter_ranges:
            if cfg.dynamic_parameterization_conceptual_model and key in cfg.dynamic_parameterization_conceptual_model:
                map_parameter_type[key] = "dynamic"
            else:  # default initialization
                map_parameter_type[key] = "static"

        return map_parameter_type

    def _initialize_information(
        self, conceptual_inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Initialize structures to store the time evolution of the internal states and the outflow

        Parameters
        ----------
        conceptual_inputs: dict[str, torch.Tensor]
            Dictionary with the different inputs as tensors of size [batch_size, time_steps].

        Returns
        -------
        Tuple[dict[str, torch.Tensor], torch.Tensor]
            - states: dict[str, torch.Tensor]
                dictionary to store the time evolution of the internal states (buckets) of the conceptual model
            - q_out: torch.Tensor
                Tensor to store the outputs of the conceptual model

        """
        model_input = next(iter(conceptual_inputs.values()))  # select a variable from dictionary (all have same shape)
        batch_size, seq_length = model_input.shape
        states = {}
        # initialize dictionary to store the evolution of the states
        for name in self._initial_states:
            states[name] = torch.zeros(
                (batch_size, seq_length, self.n_conceptual_models),
                dtype=torch.float32,
                device=model_input.device,
            )

        # initialize vectors to store the evolution of the outputs
        out = torch.zeros((batch_size, seq_length, 1), dtype=torch.float32, device=model_input.device)

        return states, out

    def _get_final_states(self, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Recovers final states of the conceptual model.

        Parameters
        ----------
        states : dict[str, torch.Tensor]
            dictionary with the time evolution of the internal states (buckets) of the conceptual model

        Returns
        -------
        dict[str, torch.Tensor]
            dictionary with the internal states (buckets) of the conceptual model, on the last timestep

        """
        return {name: state[:, -1, :] for name, state in states.items()}

    @property
    def _initial_states(self) -> dict[str, float]:
        raise NotImplementedError

    @property
    def parameter_ranges(self) -> dict[str, list[float]]:
        raise NotImplementedError
