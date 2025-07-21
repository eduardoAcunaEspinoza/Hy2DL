import os
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
    for key in sample.keys():
        if isinstance(sample[key], dict) and key.startswith(("x_d", "x_ar", "x_conceptual")):
            sample[key] = {k: v.to(device) for k, v in sample[key].items()}
        elif isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key].to(device)
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


def write_report(cfg: Config, text: str):
    """Write a given text into a text file.

    If the file where one wants to write does not exists, it creates a new one.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    text : str
        Text that wants to be added

    """
    file_path = cfg.path_save_folder / "run_progress.txt"

    if os.path.exists(file_path):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not

    highscore = open(file_path, append_write)
    highscore.write(text + "\n")
    highscore.close()
