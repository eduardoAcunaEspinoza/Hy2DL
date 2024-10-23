from typing import Dict, List, Optional, Tuple, Union

import torch

from hy2dl.modelzoo.baseconceptualmodel import BaseConceptualModel


class NonSense(BaseConceptualModel):
    """Nonsense model [#]_.

    Hydrological model with physically non-sensical constraints: water enters the model through the
    snow reservoir, then moves through the baseflow, interflow and finally unsaturated zone reservoirs,
    in that order, before exiting the model.

    Parameters
    ----------
    n_models : int
        Number of model entities that will be run at the same time (in parallel)
    parameter_type : List[str]
        List to specify which parameters of the conceptual model will be dynamic.

    References
    ----------
    .. [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket? 
        Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization, 
        Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    
    """

    def __init__(self, n_models: int = 1, parameter_type: List[str] = None):
        super(NonSense, self).__init__()
        self.n_conceptual_models = n_models
        self.parameter_type = self._map_parameter_type(parameter_type=parameter_type)
        self.output_size = 1

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of the Nonsense model (conceptual model).

        In the forward pass, each element of the batch is associated with a basin, therefore, the conceptual model is
        run for multiple basins in parallel, and also for multiple entities of the model (n_models) at the same time.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that the conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcing used to run the conceptual model
            (e.g. precipitation, temperature, ...)
        parameters: Dict[str, torch.Tensor]
            Dict with parametrization of the conceptual model.
        initial_states: Optional[Dict[str, torch.Tensor]]
            Optional parameter! In case one wants to specify the initial state of the internal states of the conceptual
            model.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            y_hat: torch.Tensor
                Simulated outflow
            parameters: Dict[str, torch.Tensor]
                Dynamic parameters of the conceptual model
            internal_states: Dict[str, torch.Tensor]
                Time evolving internal states of the conceptual model
            last_states: Dict[str, torch.Tensor]
                Internal states of the conceptual model for the last time-step

        """
        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.00, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.00, dtype=torch.float32, device=x_conceptual.device)
        klu = torch.tensor(0.90, dtype=torch.float32, device=x_conceptual.device)  # land use correction factor [-]

        # Broadcast input tensor to consider multiple conceptual models in parallel
        precipitation = torch.tile(x_conceptual[:, :, 0].unsqueeze(2), (1, 1, self.n_conceptual_models))
        et = torch.tile(x_conceptual[:, :, 1].unsqueeze(2), (1, 1, self.n_conceptual_models))
        temperature = torch.tile(x_conceptual[:, :, 2].unsqueeze(2), (1, 1, self.n_conceptual_models))

        # Division between solid and liquid precipitation can be done outside of the loop as temperature is given
        temp_mask = temperature < 0
        snow_melt = temperature * parameters["dd"]
        snow_melt[temp_mask] = zero
        # Liquid precipitation
        liquid_p = precipitation.clone()
        liquid_p[temp_mask] = zero
        # Solid precipitation (Snow)
        snow = precipitation.clone()
        snow[~temp_mask] = zero
        # Permanent wilting point (pwp) used in ET
        pwp = torch.tensor(0.8, dtype=torch.float32, device=x_conceptual.device) * parameters["sumax"]

        if initial_states is None:  # if not specified, take the default values
            ss = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["ss"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            sb = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["sb"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            si = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["si"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            su = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["su"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
        else:  # specified initial states
            ss = initial_states["ss"]
            sb = initial_states["sb"]
            si = initial_states["si"]
            su = initial_states["su"]

        # Run hydrologycal model for every time step
        for j in range(x_conceptual.shape[1]):
            # Snow module --------------------------
            qs_out = torch.minimum(ss, snow_melt[:, j, :])
            ss = ss - qs_out + snow[:, j, :]
            qsp_out = qs_out + liquid_p[:, j, :]

            # Baseflow reservoir -------------------
            sb = sb + qsp_out  # [mm]
            qb_out = sb / parameters["kb"][:, j, :]  # [mm]
            sb = sb - qb_out  # [mm]

            # Interflow
            si = si + qb_out  # [mm]
            qi_out = si / parameters["ki"][:, j, :]  # [mm]
            si = si - qi_out  # [mm]

            # Unsaturated zone --------------------
            psi = (su / parameters["sumax"][:, j, :]) ** parameters["beta"][:, j, :]  # [-]
            su_temp = su + qi_out * (1 - psi)
            su = torch.minimum(su_temp, parameters["sumax"][:, j, :])
            qu_out = qi_out * psi + torch.maximum(zero, su_temp - parameters["sumax"][:, j, :])  # [mm]

            # Evapotranspiration -----------------
            ktetha = su / parameters["sumax"][:, j, :]
            et_mask = su <= pwp[:, j, :]
            ktetha[~et_mask] = one
            ret = et[:, j, :] * klu * ktetha  # [mm]
            su = torch.maximum(zero, su - ret)  # [mm]

            # Save internal states
            states["ss"][:, j, :] = ss
            states["sb"][:, j, :] = sb
            states["si"][:, j, :] = si
            states["su"][:, j, :] = su

            # Outflow
            out[:, j, :] = qu_out  # [mm]

        # Save last states
        final_states = self._get_final_states(states=states)

        return {"y_hat": out, "parameters": parameters, "internal_states": states, "final_states": final_states}

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {"ss": 0.0, "su": 5.0, "si": 10.0, "sb": 15.0}

    @property
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {"dd": (0.0, 10.0), "sumax": (20.0, 700.0), "beta": (1.0, 6.0), "ki": (1.0, 100.0), "kb": (10.0, 1000.0)}
