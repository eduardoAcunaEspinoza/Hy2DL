"""Self-supervised LSTM with channel masking for multi-target reconstruction."""

from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class LSTM_SSL(nn.Module):
    """Self-supervised LSTM with configurable channel masking.

    During training, a fixed number of *units* per masking group is randomly selected for each sample in the batch
    and replaced by a learnable mask token (optionally perturbed with per-variable Gaussian noise). A unit is either
    a single variable or a nested list of variables that should be masked jointly (conditional masking).

    Targets are predicted by a linear head with one output per ``cfg.target`` entry, in target order. The model
    returns ``y_hat`` together with two boolean tensors aligned to the targets:

    * ``input_mask`` -- ``True`` where the channel was replaced by the mask token at the input (either because of
      NaN, or because of self-supervised masking).
    * ``nan_mask`` -- ``True`` where the original input was NaN (subset of ``input_mask``). Provided for inspection.

    ``input_mask`` is intended to be consumed by ``MaskedMSE``. Targets that do not appear in ``dynamic_input`` are
    never masked at input and therefore have ``input_mask=False`` everywhere.

    Required configuration fields
    -----------------------------
    ``dynamic_input`` : list[str]
        Flat list of dynamic input variables.
    ``target`` : list[str]
        Variables to predict, in output order.
    ``hidden_size`` : int
    ``dropout_rate`` : float
    ``predict_last_n`` : int
    ``mask_groups`` : dict[str, list]
        Mapping ``group_name -> list of units``. A unit is either a variable name (string) or a nested list of
        variable names (conditional bundle, masked jointly).
    ``mask_counts`` : dict[str, int]
        Mapping ``group_name -> number of units to mask per sample``. Must not exceed the number of units in the
        group.

    Optional configuration fields
    -----------------------------
    ``mask_noise_std_factor`` : float, default 0.0
        Standard-deviation factor for additive Gaussian noise on top of the mask token. The per-variable noise is
        scaled by the per-batch per-variable standard deviation, so that ``factor=0.01`` adds roughly 1% of the
        variable's natural variability.
    ``initial_forget_bias`` : float, optional
        Forget-gate bias initialization of 3

    Constraints
    -----------
    * ``cfg.dynamic_input`` must be a flat ``list`` (no group dict). Channel masking operates by variable name.
    * ``cfg.dynamic_embedding`` must be ``None`` (channel masking only makes sense in the raw variable space).
    * ``cfg.nan_handling_method`` must be ``None`` (NaN handling is done by this model via the mask token).
    * ``cfg.custom_seq_processing`` must be ``None`` (multi-frequency setups are out of scope for this PR).
    * Each variable may appear in at most one masking group; only ``dynamic_input`` variables are valid in
      ``mask_groups``.

    Notes
    -----
    Masking is currently *full-channel*: when a unit is selected, the entire sequence of the underlying variables is
    replaced with the mask token. A future extension could add a ``mask_temporal_strategy`` configuration (e.g.
    ``'full_channel' | 'random_substring'``) to enable MAE-style patch masking within selected channels.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        if not isinstance(cfg.dynamic_input, list):
            raise ValueError(
                "LSTM_SSL requires `dynamic_input` to be a flat list. "
                f"Got {type(cfg.dynamic_input).__name__}."
            )
        if cfg.dynamic_embedding is not None:
            raise ValueError(
                "LSTM_SSL requires `dynamic_embedding` to be None; channel masking operates in the raw variable "
                "space."
            )
        if cfg.nan_handling_method is not None:
            raise ValueError(
                "LSTM_SSL handles NaN inputs internally via the learnable mask token. "
                "Set `nan_handling_method` to None."
            )
        if cfg.custom_seq_processing is not None:
            raise ValueError(
                "LSTM_SSL does not currently support `custom_seq_processing` (multi-frequency setups)."
            )

        self._dynamic_input = list(cfg.dynamic_input)
        self._target = list(cfg.target)
        self._var_name_to_idx = {v: i for i, v in enumerate(self._dynamic_input)}
        self._target_in_input_idx = [
            self._var_name_to_idx[t] if t in self._var_name_to_idx else -1 for t in self._target
        ]

        self.emb_hc = InputLayer(cfg)
        self.lstm = nn.LSTM(
            input_size=self.emb_hc.output_size,
            hidden_size=cfg.hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=len(self._target))
        self.predict_last_n = cfg.predict_last_n

        self.mask_token = nn.Parameter(torch.zeros(()))
        self.mask_noise_std_factor = float(cfg.mask_noise_std_factor or 0.0)

        self._build_mask_units(cfg)

        self._reset_parameters(cfg)

    def _build_mask_units(self, cfg: Config) -> None:
        """Parse ``cfg.mask_groups`` and ``cfg.mask_counts`` into validated structures."""
        mask_groups_cfg = cfg.mask_groups or {}
        mask_counts_cfg = cfg.mask_counts or {}

        # group_name -> list of tuples of input indices. Each tuple represents one unit:
        # a single-element tuple for a plain variable, a multi-element tuple for a conditional bundle.
        self._mask_units: dict[str, list[tuple[int, ...]]] = {}
        self._mask_counts: dict[str, int] = {}

        seen: set[str] = set()
        for group_name, var_list in mask_groups_cfg.items():
            if not isinstance(var_list, list):
                raise TypeError(
                    f"mask_groups[{group_name!r}] must be a list, got {type(var_list).__name__}."
                )
            units: list[tuple[int, ...]] = []
            for item in var_list:
                if isinstance(item, str):
                    bundle = [item]
                elif isinstance(item, list):
                    bundle = list(item)
                else:
                    raise TypeError(
                        f"Entries of mask_groups[{group_name!r}] must be strings or lists of strings, "
                        f"got {type(item).__name__}."
                    )
                idxs: list[int] = []
                for var in bundle:
                    if var in seen:
                        raise ValueError(f"Variable {var!r} is listed more than once across mask_groups.")
                    if var not in self._var_name_to_idx:
                        raise ValueError(f"mask_groups variable {var!r} is not in dynamic_input.")
                    seen.add(var)
                    idxs.append(self._var_name_to_idx[var])
                units.append(tuple(idxs))

            count = int(mask_counts_cfg.get(group_name, 0))
            if count < 0:
                raise ValueError(f"mask_counts[{group_name!r}] must be non-negative.")
            if count > len(units):
                raise ValueError(
                    f"mask_counts[{group_name!r}] = {count} exceeds the number of units in that group "
                    f"({len(units)})."
                )
            self._mask_units[group_name] = units
            self._mask_counts[group_name] = count

    def _reset_parameters(self, cfg: Config) -> None:
        """Special initialization of certain model weights."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias

    def _sample_channel_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample per-sample, per-channel boolean mask for SSL training.

        Currently masks the entire sequence for selected channels (full-channel masking). Future extensions could
        add a configurable ``mask_temporal_strategy`` (e.g. ``'full_channel' | 'random_substring'``) to enable
        MAE-style patch masking within selected channels.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``[batch_size, len(dynamic_input)]``.
        """
        n_dynamic = len(self._dynamic_input)
        mask = torch.zeros(batch_size, n_dynamic, dtype=torch.bool, device=device)
        for b in range(batch_size):
            for group_name, units in self._mask_units.items():
                count = self._mask_counts[group_name]
                if count == 0:
                    continue
                perm = torch.randperm(len(units), device=device)[:count]
                for unit_idx in perm.tolist():
                    for var_idx in units[unit_idx]:
                        mask[b, var_idx] = True
        return mask

    @staticmethod
    def _per_variable_std(t: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Standard deviation of ``t`` over valid positions (scalar). ``t`` and ``valid`` are ``[B, L]``."""
        t_filled = torch.where(valid, t, torch.zeros_like(t))
        count = valid.sum().clamp(min=1).to(t.dtype)
        mean = t_filled.sum() / count
        var = (((t_filled - mean) ** 2) * valid).sum() / count
        return torch.sqrt(var + 1e-8)

    def _fill_with_token(
        self, t: torch.Tensor, where: torch.Tensor, std: torch.Tensor | None
    ) -> torch.Tensor:
        """Replace positions selected by ``where`` with the mask token (with optional per-variable noise)."""
        if std is None or self.mask_noise_std_factor == 0:
            return torch.where(where, self.mask_token, t)
        noise = self.mask_noise_std_factor * std * torch.randn_like(t)
        return torch.where(where, self.mask_token + noise, t)

    def forward(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        sample : dict
            Sample dictionary as produced by the dataset / dataloader. Must contain ``"x_d"`` as a per-variable
            dict ``{var_name: tensor[B, L]}``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``y_hat`` of shape ``[B, predict_last_n, len(target)]`` plus ``input_mask`` and ``nan_mask`` of the same
            shape (boolean), aligned to the target order.
        """
        if "x_d" not in sample:
            raise KeyError(
                "LSTM_SSL expects 'x_d' in sample; multi-frequency setups are not supported."
            )

        x_d_orig = sample["x_d"]
        first_var = next(iter(x_d_orig.values()))
        B, L = first_var.shape
        device = first_var.device

        nan_masks = {
            v: torch.isnan(t) for v, t in x_d_orig.items() if v in self._var_name_to_idx
        }

        if self.mask_noise_std_factor > 0:
            stds_per_var = {
                v: self._per_variable_std(x_d_orig[v], ~nan_masks[v]) for v in nan_masks
            }
        else:
            stds_per_var = None

        if self.training:
            ssl_mask = self._sample_channel_mask(B, device)
        else:
            ssl_mask = torch.zeros(B, len(self._dynamic_input), dtype=torch.bool, device=device)

        modified_x_d: dict[str, torch.Tensor] = {}
        for var_name, t in x_d_orig.items():
            if var_name not in self._var_name_to_idx:
                raise ValueError(
                    f"Variable {var_name!r} present in sample['x_d'] but not declared in "
                    f"cfg.dynamic_input. Expected: {self._dynamic_input}."
                )
            var_idx = self._var_name_to_idx[var_name]
            nan_mask_v = nan_masks[var_name]
            ssl_expanded = ssl_mask[:, var_idx].unsqueeze(1).expand(-1, L)
            combined = nan_mask_v | ssl_expanded
            std_v = stds_per_var[var_name] if stds_per_var is not None else None
            modified_x_d[var_name] = self._fill_with_token(t, combined, std_v)

        modified_sample = {**sample, "x_d": modified_x_d}

        x_lstm = self.emb_hc(modified_sample)
        hs, _ = self.lstm(x_lstm)
        hs = hs[:, -self.predict_last_n :, :]
        y_hat = self.linear(self.dropout(hs))

        n_target = len(self._target)
        input_mask = torch.zeros(B, self.predict_last_n, n_target, dtype=torch.bool, device=device)
        nan_mask_target = torch.zeros_like(input_mask)
        for t_idx, in_idx in enumerate(self._target_in_input_idx):
            if in_idx < 0:
                continue
            var_name = self._dynamic_input[in_idx]
            nan_v = nan_masks[var_name]
            ssl_v = ssl_mask[:, in_idx].unsqueeze(1).expand(-1, L)
            combined_v = nan_v | ssl_v
            input_mask[:, :, t_idx] = combined_v[:, -self.predict_last_n :]
            nan_mask_target[:, :, t_idx] = nan_v[:, -self.predict_last_n :]

        return {"y_hat": y_hat, "input_mask": input_mask, "nan_mask": nan_mask_target}
