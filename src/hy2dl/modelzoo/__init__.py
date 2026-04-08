import torch.nn as nn

# Deep learning methods
from hy2dl.modelzoo.cudalstm import CudaLSTM
from hy2dl.modelzoo.lstmmdn import LSTMMDN
from hy2dl.modelzoo.maskedarlstm import MaskedARLSTM
from hy2dl.utils.config import Config

# Define the registry mapping
model_registry = {
    "cudalstm": CudaLSTM,
    "lstmmdn": LSTMMDN,
    "maskedarlstm": MaskedARLSTM,
}


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
    model_name = cfg.model.lower()

    if model_name not in model_registry:
        available = list(model_registry.keys())
        raise NotImplementedError(f"'{model_name}' not implemented. Available models: {available}")

    # Instantiate the mapped class and return it
    model_class = model_registry[model_name]
    return model_class(cfg=cfg)
