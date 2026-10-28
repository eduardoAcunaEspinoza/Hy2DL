import random

import numpy as np
import torch

from hy2dl.utils.config import Config


def upload_to_device(sample: dict, device):
    """Upload the different tensors, contained in dictionaries, to the device (e.g. gpu)."""

    def _to_device(v):
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            return {k: _to_device(val) for k, val in v.items()}
        return v  # leaves non-tensor entries (e.g. gauge_id, date, source_fc) untouched

    return {k: _to_device(v) for k, v in sample.items()}


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
