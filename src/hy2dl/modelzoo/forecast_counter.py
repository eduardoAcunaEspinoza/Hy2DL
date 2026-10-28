import torch
import torch.nn as nn

from hy2dl.utils.config import Config


class ForecastCounter(nn.Module):
    """Forecast step counter / lead-time feature across hindcast and forecast horizons.

    Class to add forecast positional information into the input sequence.

    Currently 2 counter types supported:
    - 'linear': 0.0 for hindcast, and 1, 2, ..., n_fc for forecast.
    - 'exponential': 1.0 for hindcast (exp(0)), and exp(-t / tau) for forecast (t = 1, ..., n_fc).

    Parameters
    ----------
    cfg : Config
        Configuration object.
    n_hc : int
        Number of hindcast steps.
    n_fc : int
        Number of forecast steps.
    """

    def __init__(
        self,
        cfg: Config,
        n_hc: int,
        n_fc: int,
    ):
        super().__init__()

        # Get type of counter from config
        model_type = cfg.forecast_counter
        if isinstance(model_type, bool): #default is linear
            model_type = "linear" if model_type else None
        elif isinstance(model_type, str):
            model_type = model_type.lower().strip()
            if model_type in ("none", "false", ""):
                model_type = None
        
        # If no forecast horizon, disable counter
        if n_fc == 0:
            model_type = None

        # Store attributes
        self.model_type = model_type
        self.n_hc = n_hc
        self.n_fc = n_fc
        self.n_channels = 1 if model_type is not None else 0

        if self.model_type is not None:
            counter = torch.zeros(n_hc + n_fc, dtype=torch.float32)
            t = torch.arange(1, n_fc + 1, dtype=torch.float32)

            if self.model_type == "linear":
                counter[:n_hc] = 0.0
                counter[n_hc:] = t
            elif self.model_type == "exponential":
                counter[:n_hc] = 1.0
                counter[n_hc:] = torch.exp(-t / cfg.forecast_counter_tau)
            else:
                raise ValueError(
                    f"Unknown forecast_counter mode '{model_type}'. Available options: ['linear', 'exponential']."
                )
            self.register_buffer("counter", counter)
        else:
            self.register_buffer("counter", None)

    def forward(self, x: torch.Tensor, period: str = "all", step: int | None = None) -> torch.Tensor:
        """Appends the counter feature to tensor `x` along the feature dimension (dim=2).

        If forecast_counter is disabled (size == 0), returns `x` unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, seq_len, features).
        period : str
            - 'all': uses full sequence counter (expects seq_len == n_hc + n_fc).
            - 'hc': uses hindcast slice (expects seq_len == n_hc).
            - 'fc': uses forecast slice (expects seq_len == n_fc).
        step : int, optional
            Step index t in [0, n_fc - 1]. Extracts step counter (expects seq_len == 1).
        """
        if self.counter is None:
            return x

        batch_size = x.shape[0]
        if step is not None:
            x_c = self.counter[self.n_hc + step : self.n_hc + step + 1].view(1, 1, 1).expand(batch_size, -1, -1)
        elif period == "hc":
            x_c = self.counter[: self.n_hc].view(1, -1, 1).expand(batch_size, -1, -1)
        elif period == "fc":
            x_c = self.counter[self.n_hc :].view(1, -1, 1).expand(batch_size, -1, -1)
        else:  # 'all'
            x_c = self.counter.view(1, -1, 1).expand(batch_size, -1, -1)

        return torch.cat([x, x_c], dim=2)
