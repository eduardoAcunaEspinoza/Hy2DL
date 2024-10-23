from typing import Dict, List, Tuple

import torch

from hy2dl.modelzoo.baseconceptualmodel import BaseConceptualModel


class UH_routing(BaseConceptualModel):
    """Unit hydrograph routing based on gamma function.

    Implementation based on Feng et al. [1]_ and Croley [2]_.

    References
    ----------
    .. [1] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water
        Resources Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    .. [2] Croley II, T. E. (1980). Gamma synthetic hydrographs. Journal of Hydrology, 47(1-2), 41-52.
        https://doi.org/10.1016/0022-1694(80)90046-3

    """

    def __init__(self, n_models: int = 1, parameter_type: List[str] = None):
        super(UH_routing, self).__init__()
        self.n_conceptual_models = 1
        self.parameter_type = self._map_parameter_type()
        self.output_size = 1

    def forward(self, discharge: torch.Tensor, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass on the routing model

        Parameters
        ----------
        discharge : torch.Tensor
            Discharge series.
        parameters : Dict[str, torch.Tensor]
            Dictionary with parameterization of routing model.

        Returns
        -------
        y_routed : torch.Tensor
            Discharge series after applying the rouing module

        """
        UH = self._gamma_routing(alpha=parameters["alpha"][:, 0, 0], beta=parameters["beta"][:, 0, 0], uh_len=15)
        y_routed = self._uh_conv(discharge, UH)
        return y_routed

    def _gamma_routing(self, alpha: torch.Tensor, beta: torch.Tensor, uh_len: int):
        """Unit hydrograph based on gamma function.

        Parameters
        ----------
        alpha: torch.Tensor
            Shape parameter of the Gamma distribution.
        beta: torch.Tensor
            Scale parameter of the Gamma distribution.
        uh_len: int
            Number of timesteps the unitary hydrograph will have.

        Returns
        -------
        uh : torch.Tensor
            Unit hydrograph

        """
        # Steps where the probability density function (pdf) will be computed
        x = (
            torch.arange(0.5, 0.5 + uh_len, 1, dtype=torch.float32, device=alpha.device)
            .unsqueeze(1)
            .repeat(1, len(alpha))
        )
        # Compute the pdf using the Gamma distribution formula
        coeff = 1 / (beta**alpha * torch.lgamma(alpha).exp())
        gamma_pdf = coeff * (x ** (alpha - 1)) * torch.exp(-x / beta)
        # Normalize data so the sum of the pdf equals 1
        uh = gamma_pdf / torch.sum(gamma_pdf, dim=0)
        return uh.unsqueeze(2)

    def _uh_conv(self, discharge: torch.Tensor, unit_hydrograph: torch.Tensor) -> torch.Tensor:
        """
        Convolution of discharge series and unit hydrograph.

        Parameters
        ----------
        discharge : torch.Tensor
            Discharge series.
        unit_hydrograph : torch.Tensor
            Unit hydrograph.

        Returns
        -------
        torch.Tensor
            Routed discharge.

        """
        batch_size = discharge.shape[0]
        kernel_size = unit_hydrograph.shape[0]
        padding_size = kernel_size - 1

        # Transpose unit_hydrograph to shape (batch_size, 1, kernel_size)
        unit_hydrograph = unit_hydrograph.permute(1, 2, 0)

        # Reshape discharge to shape (1, batch_size, timesteps)
        discharge = discharge.permute(2, 0, 1)

        # Perform the convolution
        routed_discharge = torch.nn.functional.conv1d(
            discharge, torch.flip(unit_hydrograph, [2]), groups=batch_size, padding=padding_size
        )
        # Remove padding from the output
        routed_discharge = routed_discharge[:, :, :-padding_size]

        return routed_discharge.permute(1, 2, 0)  # Shape: (batch_size, timesteps, 1)

    @property
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {"alpha": (0.0, 2.9), "beta": (0.0, 6.5)}
