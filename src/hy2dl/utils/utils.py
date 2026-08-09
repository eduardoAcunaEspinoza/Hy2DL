import random

import numpy as np
import torch

from hy2dl.utils.config import Config


def upload_to_device(sample: dict, device):
    """Upload the different tensors, contained in dictionaries, to the device (e.g. gpu).

    Parameters
    ----------
    cfg : Config
        Configuration file.
    sample : dict
        Dictionary with the different tensors that will be used for the forward pass.

    """
    for k, v in sample.items():
        if isinstance(v, dict) and k.startswith(("x_d", "x_conceptual")):
            sample[k] = {k: v.to(device) for k, v in v.items()}
        elif isinstance(v, torch.Tensor):
            sample[k] = v.to(device)
    return sample


def set_random_seed(cfg: Config):
    """Set a seed for various packages to be able to reproduce the results.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
