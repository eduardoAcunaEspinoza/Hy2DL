from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class MaskedARLSTM(nn.Module):
    """Autoregressive LSTM

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, cfg: Config):
        pass

    def forward(self, x):
        pass
