from typing import Dict, List, Optional, Tuple, Union

import torch

from hy2dl.modelzoo.baseconceptualmodel import BaseConceptualModel


class linear_reservoir(BaseConceptualModel):
    """Linear reservoir model [#]_.

    The model can be used as a differentiable entity to create hybrid models. One can run it in parallel for multiple
    basins, and also multiple entities of the model at the same time.

    Parameters
    ----------
    n_models : int
        Number of model entities that will be run at the same time
    parameter_type : List[str]
        List to specify which parameters of the conceptual model will be dynamic.

    References
    ----------
    .. [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket? 
        Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization, 
        Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    
    """

    def __init__(self, n_models: int = 1, parameter_type: List[str] = None):
        super(linear_reservoir, self).__init__()
        self.n_conceptual_models = n_models
        self.parameter_type = self._map_parameter_type(parameter_type=parameter_type)
        self.output_size = 1

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass on the linear reservoir model

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)
        parameters: Dict[str, torch.Tensor]
            Dictionary with parameterization of conceptual model
        initial_states: Optional[Dict[str, torch.Tensor]]
            Optional parameter! In case one wants to specify the initial state of the internal states of the conceptual
            model.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            y_hat: torch.Tensor
                Simulated outflow
            parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model
            last_states: Dict[str, torch.Tensor]]
                Internal states of the conceptual model in the last timestep

        """
        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        if initial_states is None:  # if we did not specify initial states it takes the default values
            si = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["si"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )

        else:  # we specify the initial states
            si = initial_states["si"]

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # Broadcast tensor to consider multiple conceptual models running in parallel
            p = torch.tile(x_conceptual[:, j, 0].unsqueeze(1), (1, self.n_conceptual_models))
            et = torch.tile(x_conceptual[:, j, 1].unsqueeze(1), (1, self.n_conceptual_models))

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

        return {"y_hat": out, "parameters": parameters, "internal_states": states, "final_states": final_states}

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {
            "si": 0.001,
        }

    @property
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {"ki": (0.002, 1.0), "aux_ET": (0.0, 1.5)}
