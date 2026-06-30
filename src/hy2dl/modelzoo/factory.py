import torch.nn as nn

from hy2dl.utils.config import Config


def get_model_registry():
    from hy2dl.modelzoo.cudalstm import CudaLSTM
    from hy2dl.modelzoo.hybrid import Hybrid

    return {"cudalstm": CudaLSTM, "hybrid": Hybrid}


def get_head_registry():
    from hy2dl.modelzoo.head import MDN, Regression

    return {"regression": Regression, "mdn": MDN}


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
    registry = get_model_registry()
    model_name = cfg.model.lower()

    if model_name not in registry:
        available = list(registry.keys())
        raise NotImplementedError(f"'{model_name}' not implemented. Available models: {available}")

    # Instantiate the mapped class and return it
    model_class = registry[model_name]
    return model_class(cfg=cfg)


def get_head(cfg: Config) -> nn.Module:
    """Get head object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new head instance of the type specified in the config.
    """
    registry = get_head_registry()
    head_name = cfg.head.lower()

    if head_name not in registry:
        available = list(registry.keys())
        raise NotImplementedError(f"'{head_name}' not implemented. Available heads: {available}")

    # Instantiate the mapped class and return it
    head_class = registry[head_name]
    return head_class(cfg=cfg)
