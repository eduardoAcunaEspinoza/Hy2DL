from typing import Optional

import torch

from hy2dl.modelzoo.baseconceptualmodel import BaseConceptualModel
from hy2dl.utils.config import Config


class custom_reservoir(BaseConceptualModel):
    """Custom reservoir model.

    The model can be used as a differentiable entity to create hybrid models. One can run it in parallel for multiple
    basins, and also multiple entities of the model at the same time.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """

    def __init__(self, cfg: Config):
        super(custom_reservoir, self).__init__()
        self.cfg = cfg
        self.n_conceptual_models = cfg.num_conceptual_models
        self.parameter_type = self._map_parameter_type(cfg=cfg)

    def forward(
        self,
        x_conceptual: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Forward pass on the linear reservoir model

        Parameters
        ----------
        x_conceptual: dict[str, torch.Tensor]
            dictionary with the different inputs as tensors of size [batch_size, time_steps].
        parameters: dict[str, torch.Tensor]
            dictionary with parameterization of conceptual model
        initial_states: Optional[dict[str, torch.Tensor]]
            Optional parameter! In case one wants to specify the initial state of the internal states of the conceptual
            model.

        Returns
        -------
        dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]
            y_hat: torch.Tensor
                Simulated outflow
            parameters: dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            internal_states: dict[str, torch.Tensor]
                Time-evolution of the internal states of the conceptual model
            last_states: dict[str, torch.Tensor]
                Internal states of the conceptual model in the last timestep

        """
        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        batch_size, seq_length = x_conceptual["precipitation"].shape
        device = x_conceptual["precipitation"].device

        if initial_states is None:  # if we did not specify initial states it takes the default values
            si = torch.full(
                (batch_size, self.n_conceptual_models),
                self._initial_states["si"],
                dtype=torch.float32,
                device=device,
            )

        else:  # we specify the initial states
            si = initial_states["si"]

        # run hydrological model for each time step
        for j in range(seq_length):
            # Broadcast tensor to consider multiple conceptual models running in parallel
            p = torch.tile(
                x_conceptual["precipitation"][:, j].unsqueeze(1),
                (1, self.n_conceptual_models),
            )
            et = torch.tile(x_conceptual["pet"][:, j].unsqueeze(1), (1, self.n_conceptual_models))

            # 1 bucket reservoir ------------------
            si = si + p  # [mm]
            ret = et * parameters["aux_ET"][:, j, :]  # [mm]
            si = torch.maximum(torch.tensor(0.0, requires_grad=True, dtype=torch.float32), si - ret)  # [mm]
            qi_out = si * parameters["ki"][:, j, :]  # [mm]
            si = si - qi_out  # [mm]

            # states
            states["si"][:, j, :] = si

            # discharge
            out[:, j, 0] = torch.mean(qi_out, dim=1)  # [mm]

        # last states
        final_states = self._get_final_states(states=states)

        return {
            "y_hat": out,
            "parameters": parameters,
            "internal_states": states,
            "final_states": final_states,
        }

    @property
    def _initial_states(self) -> dict[str, float]:
        return getattr(self, '_initial_states_values', {"si": 0.001})
        
    @_initial_states.setter
    def _initial_states(self, value: dict[str, float]):
        self._initial_states_values = value

    @property
    def parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return getattr(self, "parameter_ranges_values", {"ki": (0.002, 1.0), "aux_ET": (0.0, 1.5)})
        
    @parameter_ranges.setter
    def parameter_ranges(self, value: dict[str, tuple[float, float]]):
        self.parameter_ranges_values = value
        
    def reset(self):
        self.n_conceptual_models = self.cfg.num_conceptual_models
        self.parameter_type = self._map_parameter_type(cfg=self.cfg)
        
