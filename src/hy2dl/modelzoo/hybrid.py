import torch
import torch.nn as nn

# Conceptual models
from hy2dl.modelzoo.baseconceptualmodel import BaseConceptualModel
from hy2dl.modelzoo.hbv import HBV
from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.modelzoo.linear_reservoir import linear_reservoir
from hy2dl.modelzoo.nonsense import NonSense
from hy2dl.modelzoo.shm import SHM

# Routing models
from hy2dl.modelzoo.uh_routing import UH_routing
from hy2dl.utils.config import Config


class Hybrid(nn.Module):
    """Wrapper to combine a deep learning model with a conceptual hydrological models [#]_.

    Hybrid model in which a conceptual hydrological model is parameterized using a LSTM network.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    References
    ----------
    .. [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket?
        Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization,
        Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.

    """

    def __init__(self, cfg: Config):
        super().__init__()

        # LSTM model
        self.embedding_net = InputLayer(cfg)
        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size, batch_first=True)

        #  Conceptual model
        self.conceptual_model = _get_conceptual_model(cfg)
        self.n_conceptual_model_params = len(self.conceptual_model.parameter_ranges) * cfg.num_conceptual_models

        # Routing model
        self.routing_model = _get_routing_model(cfg) if cfg.routing_model is not None else None
        self.n_routing_params = len(self.routing_model.parameter_ranges) if self.routing_model is not None else 0

        # Linear layer
        self.linear = nn.Linear(
            in_features=cfg.hidden_size, out_features=self.n_conceptual_model_params + self.n_routing_params
        )

        self.cfg = cfg

    def forward(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Forward pass on hybrid model.

        In the forward pass, each element of the batch is associated with a basin. Therefore, the conceptual model is
        done to run multiple basins in parallel, and also multiple entities of the model at the same time.

        Parameters
        ----------
        sample: dict[str, torch.Tensor]
            Dictionary with the different tensors that will be used for the forward pass.

        Returns
        -------
        pred: dict[str, torch.Tensor]

        """

        # Preprocess data to be sent to the LSTM
        processed_sample = self.embedding_net(sample)
        x_lstm = self.embedding_net.assemble_sample(processed_sample)

        # Forward pass through the LSTM
        lstm_output, _ = self.lstm(x_lstm)

        # map lstm outputs to the dimension of the conceptual model´s parameters
        lstm_output = self.linear(lstm_output)

        # map lstm output to parameters of conceptual model
        warmup_period = self.cfg.seq_length - self.cfg.predict_last_n
        parameters_warmup, parameters_simulation = self.conceptual_model.map_parameters(
            lstm_out=lstm_output[:, :, : self.n_conceptual_model_params], warmup_period=warmup_period
        )

        # run conceptual model: warmup
        with torch.no_grad():
            pred = self.conceptual_model(
                x_conceptual={k: v[:, :warmup_period] for k, v in sample["x_d_conceptual"].items()},
                parameters=parameters_warmup,
            )

        # run conceptual model: simulation
        pred = self.conceptual_model(
            x_conceptual={k: v[:, warmup_period:] for k, v in sample["x_d_conceptual"].items()},
            parameters=parameters_simulation,
            initial_states=pred["final_states"],
        )
        # Conceptual routing
        if self.routing_model is not None:
            _, parameters_simulation = self.routing_model.map_parameters(
                lstm_out=lstm_output[:, :, self.n_conceptual_model_params :], warmup_period=warmup_period
            )
            # apply routing routine
            pred["y_hat"] = self.routing_model(discharge=pred["y_hat"], parameters=parameters_simulation)

        return pred


def _get_conceptual_model(cfg: Config) -> BaseConceptualModel:
    """Get conceptual model, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    BaseConceptualModel
        A new conceptual model instance of the type specified in the config.
    """
    if cfg.conceptual_model.lower() == "hbv":
        model = HBV(cfg=cfg)
    elif cfg.conceptual_model.lower() == "linear_reservoir":
        model = linear_reservoir(cfg=cfg)
    elif cfg.conceptual_model.lower() == "nonsense":
        model = NonSense(cfg=cfg)
    elif cfg.conceptual_model.lower() == "shm":
        model = SHM(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_conceptual_model()`")

    return model


def _get_routing_model(cfg: Config) -> BaseConceptualModel:
    """Get routing model, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    BaseConceptualModel
        A new conceptual model instance of the type specified in the config.
    """
    if cfg.routing_model.lower() == "uh_routing":
        model = UH_routing(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.routing_model} not implemented or not linked in `get_routing_model()`")

    return model
