import torch.nn as nn

# Deep learning methods
from hy2dl.modelzoo.cudalstm import CudaLSTM
from hy2dl.modelzoo.forecast_lstm import ForecastLSTM
from hy2dl.modelzoo.hybrid import Hybrid
from hy2dl.modelzoo.lstmmdn import LSTMMDN
from hy2dl.utils.config import Config


def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """

    if cfg.model.lower() == "cudalstm":
        model = CudaLSTM(cfg=cfg)
    elif cfg.model.lower() == "forecast_lstm":
        model = ForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == "hybrid":
        model = Hybrid(cfg=cfg)
    elif cfg.model.lower() == "lstmmdn":
        model = LSTMMDN(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
