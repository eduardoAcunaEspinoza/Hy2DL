from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseConceptualModel(nn.Module):
    """Abstract base model class

    Don't use this class for model training!The purpose is to have some common operations that all conceptual models
    will need.

    """

    def __init__(
        self,
    ):
        super(BaseConceptualModel, self).__init__()

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        raise NotImplementedError

    def map_parameters(
        self, lstm_out: torch.Tensor, warmup_period: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            - parameters_warmup : Dict[str, torch.Tensor]
                Parameters for the warmup period (always static!)
            - parameters_simulation : Dict[str, torch.Tensor]
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

    def _map_parameter_type(self, parameter_type: List[str] = None):
        """Define parameter type, static or dynamic.

        The model parameters can be static or dynamic. This function creates a dictionary that associate the parameter
        name with a type specified by the user. In case the user did not specify a type, the parameter_type is
        automatically specified as static.

        Parameters
        ----------
        parameter_type : List[str]
            List to specify which parameters of the conceptual model will be dynamic.

        Returns
        -------
        map_parameter_type: Dict[str, str]
            Dictionary

        """
        map_parameter_type = {}
        for key, _ in self.parameter_ranges.items():
            if parameter_type is not None and key in parameter_type:  # if user specified the type
                map_parameter_type[key] = "dynamic"
            else:  # default initialization
                map_parameter_type[key] = "static"

        return map_parameter_type

    def _initialize_information(self, conceptual_inputs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Initialize structures to store the time evolution of the internal states and the outflow

        Parameters
        ----------
        conceptual_inputs: torch.Tensor
            Inputs of the conceptual model

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            - states: Dict[str, torch.Tensor]
                Dictionary to store the time evolution of the internal states (buckets) of the conceptual model
            - q_out: torch.Tensor
                Tensor to store the outputs of the conceptual model

        """
        states = {}
        # initialize dictionary to store the evolution of the states
        for name, _ in self._initial_states.items():
            states[name] = torch.zeros(
                (conceptual_inputs.shape[0], conceptual_inputs.shape[1], self.n_conceptual_models),
                dtype=torch.float32,
                device=conceptual_inputs.device,
            )

        # initialize vectors to store the evolution of the outputs
        out = torch.zeros(
            (conceptual_inputs.shape[0], conceptual_inputs.shape[1], self.output_size),
            dtype=torch.float32,
            device=conceptual_inputs.device,
        )

        return states, out

    def _get_final_states(self, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recovers final states of the conceptual model.

        Parameters
        ----------
        states : Dict[str, torch.Tensor]
            Dictionary with the time evolution of the internal states (buckets) of the conceptual model

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the internal states (buckets) of the conceptual model, on the last timestep

        """
        return {name: state[:, -1, :] for name, state in states.items()}

    @property
    def _initial_states(self) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        raise NotImplementedError
