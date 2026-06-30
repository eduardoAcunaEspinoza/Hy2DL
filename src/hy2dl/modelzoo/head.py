import torch
import torch.nn as nn

from hy2dl.utils import get_distribution
from hy2dl.utils.config import Config


class Regression(nn.Module):
    """Regression head layer.

    This class implements a regression head, which maps the hidden states produced by the LSTM into predictions.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.output_features = cfg.output_features
        self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.output_features)

    def forward(self, x) -> dict:
        """Forward pass through the regression head.

        Parameters
        ----------
        x : torch.Tensor (..., hidden_size)

        Returns
        -------
        dict
            Dictionary containing the output tensor and hidden states.
        """
        return {"y_hat": self.linear(x), "hs": x}


class MDN(nn.Module):
    """Mixture Density Network (MDN) head layer.

    This class implements a MDN head layer [1]_, to map the hidden states produced by the LSTM into the parameters of a
    mixture distribution.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.

    References
    ----------
    .. [1] Klotz, D., Kratzert, F., Gauch, M., Keefe Sampson, A., Brandstetter, J., Klambauer, G., Hochreiter, S., and
        Nearing, G. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. Hydrology and Earth
        System Sciences, 26(6), 1673–1693. https://doi.org/10.5194/hess-26-1673-2022
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.distribution = get_distribution(distribution=cfg.distribution)
        self.num_mixture_components = cfg.num_mixture_components
        self.output_features = cfg.output_features
        # fully connected (fc) layer to map the hidden states to the parameters of the mixture distribution
        self.fc_params = nn.Linear(
            cfg.hidden_size, len(self.distribution.parameters) * cfg.num_mixture_components * cfg.output_features
        )
        # fully connected layer to map the hidden states to the mixture weights, followed by a softmax activation to
        # ensure they sum to 1
        self.fc_weights = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.num_mixture_components * cfg.output_features),
            nn.Unflatten(-1, (cfg.num_mixture_components, cfg.output_features)),
            nn.Softmax(dim=-2),
        )

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization.

        Initializes the mixture weights from -0.5 to 0.5 to avoid symmetry issues at the beginning of training.

        Parameters
        ----------
        cfg : Config
            Configuration object containing model hyperparameters and settings.
        """

        # mixture weights
        bias_init = torch.linspace(0.5, -0.5, cfg.num_mixture_components).repeat(cfg.output_features)
        self.fc_weights[0].bias.data = bias_init

    def forward(self, x) -> dict:
        """Forward pass through the MDN head.

        Parameters
        ----------
        x : torch.Tensor (..., hidden_size)


        """
        # Parameters of the mixture distribution
        params = self.distribution.map_parameters(
            raw_params=self.fc_params(x),
            num_mixture_components=self.num_mixture_components,
            num_targets=self.output_features,
        )
        # Mixture weights
        weights = self.fc_weights(x)

        # expected value of the distribution -> deterministic prediction
        y_hat = self.distribution.mean(params=params, weights=weights)

        return {"y_hat": y_hat, "params": params, "weights": weights}
