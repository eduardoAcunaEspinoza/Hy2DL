from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo import get_head
from hy2dl.modelzoo.forecast_counter import ForecastCounter
from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class MF2LSTM(nn.Module):
    """Multi-Frequency Masked-Forecasting LSTM (MF2LSTM)

    This architecture integrates multi-frequency [1]_ hindcast processing with a sequential-forecast framework [2]_ for
    computationally efficient hourly hydrological forecasting. The model utilizes a masked-mean embedding strategy [3]_
    to handle missing data and assimilate real-time discharge signals.

    Key Features:

    * Training Mode (Teacher-Forcing): Ingests observed discharge targets across the forecast horizon to accelerate
      model training. To mitigate exposure bias, a sigmoid-scaled synthetic noise profile is injected over the
      forecast sequence.
    * Inference Mode (Autoregressive): Manually unrolls the LSTM cell across the forecast horizon, recursively feeding
      predicted discharges back into subsequent timesteps.
    * Missing Data Handling: Due to the masked-mean embedding, the model can operate both with and without real-time
      discharge assimilation.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.

    References
    ----------
    .. [1] Acuna Espinoza, E., Kratzert, F., Klotz, D., Gauch, M., Alvarez Chaves, M., Loritz, R., & Ehret, U. (2025).
        Technical note: An approach for handling multiple temporal frequencies with different input dimensions using a
        single LSTM cell. Hydrology and Earth System Sciences, 29(21), 1749–1758. https://doi.org/10.5194/hess-29-1749-2025
    .. [2] Cohen, D., Amira, R., Aschner, R., Carny, Y., Feinstein, B., Fester, H., Fronman, S., Gauch, M., Gilon, O.,
        Green, R., Hassidim, A., Klotz, D., Kratzert, F., Korenfeld, D., Loike, G., Markel, A., Matias, Y., Mayo, R.,
        Metzger, A., . . . Nearing, G. (2026). Extending medium-range global flood forecasts: The Google Global Flood
        Forecasting Model Version 2. EGUsphere, 2026, 1–31. https://doi.org/10.5194/egusphere-2026-2283
    .. [3] Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Cohen, D., & Gilon, O. (2025). How to deal with missing
        input data. Hydrology and Earth System Sciences, 29(21), 6221–6235. https://doi.org/10.5194/hess-29-6221-2025
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.emb_hc = InputLayer(cfg)
        self.emb_fc = InputLayer(cfg, embedding_type="forecast", shared_embeddings=self.emb_hc.shared_embeddings)
        self.fc_counter = ForecastCounter(cfg, n_hc=self.emb_hc.input_seq_length, n_fc=self.emb_fc.input_seq_length)

        # Cuda LSTM
        self.lstm = nn.LSTM(
            input_size=self.emb_hc.output_size + self.fc_counter.n_channels,
            hidden_size=cfg.hidden_size,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(p=cfg.dropout_rate)
        self.predict_last_n = cfg.predict_last_n

        # Set teacher-forcing
        self.ar_teacher_forcing = cfg.ar_teacher_forcing

        # Head layer
        self.head = get_head(cfg=cfg)
        self.head_type = cfg.head.lower()

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of certain model weights."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, x):
        """Forward pass of the MF2LSTM network.

        Uses teacher-forcing mode (`forward_tf`) during training if `ar_teacher_forcing` is True, otherwise runs
        autoregressive cycle (`forward_ar`).
        """

        return self.forward_tf(x) if self.training and self.ar_teacher_forcing else self.forward_ar(x)

    def forward_tf(self, sample):
        """Teacher-forcing mode"""

        x_hc = self.emb_hc(sample)
        x_hc = self.fc_counter(x_hc, period="hc")
        out_hc, (hs, cs) = self.lstm(x_hc)

        group_mask = getattr(self.emb_hc, "group_mask", None)
        x_fc = self.emb_fc(sample, group_mask=group_mask)
        x_fc = self.fc_counter(x_fc, period="fc")
        out_fc, _ = self.lstm(x_fc, (hs, cs))

        out = torch.cat([out_hc, out_fc], dim=1)[:, -self.predict_last_n :, :]
        return self.head(self.dropout(out))

    def forward_ar(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Autoregressive mode"""
        model_out = {}
        ### ---------------
        ### Hindcast period
        ### ---------------
        hc_info = self.emb_hc(sample, assemble=False)
        x_hc = torch.cat([hc_info["x_d"], hc_info["freq_flag"], hc_info["x_s"]], dim=2)
        x_hc = self.fc_counter(x_hc, period="hc")

        # Forward pass of hindcast period
        out_hc, (hs, cs) = self.lstm(x_hc)

        ### ---------------
        ### Forecast period
        ### ---------------
        dynamic_inputs = self.emb_fc.dynamic_input
        x_fc_groups = {}
        group_masks = {}

        # Pre-extract sequences and masks for all groups
        for g_name in dynamic_inputs:
            g_vars = sample["x_d_fc"][g_name] if isinstance(sample["x_d_fc"], dict) else sample["x_d_fc"]

            # Groups and masks
            if g_name == self.emb_fc.cfg.ar_group_name:
                ar_t = g_vars[:, :1, :]
                # Keep ar masked if it was dropped probabilistically, or if it is NaN
                is_ar_absent = self._ar_masked(ar_t) | ar_t.isnan().all(dim=-1, keepdim=True)
                group_masks[g_name] = is_ar_absent
            else:
                x_fc_groups[g_name] = g_vars
                if self.emb_fc.cfg.nan_probabilistic_masking:
                    group_masks[g_name] = self.emb_hc.group_mask[g_name].unsqueeze(1).unsqueeze(2)
                else:
                    group_masks[g_name] = g_vars.new_zeros(g_vars.shape[0], 1, 1, dtype=torch.bool)

        # Forecast frequency flags
        batch_size = x_hc.shape[0]
        fc_freq_flag = (
            self.emb_fc.flag_info["flag"].unsqueeze(0).expand(batch_size, -1, -1)
            if self.emb_fc.flag_info.get("flag") is not None
            else x_hc.new_zeros(batch_size, self.emb_fc.input_seq_length, 0)
        )

        # Preallocate prediction tensor
        pred = torch.empty((batch_size, self.emb_fc.input_seq_length, self.head.output_features), device=x_hc.device)
        if self.head_type == "mdn":
            preds_weights = []
            preds_params = {k: [] for k in self.head.distribution.parameters}

        # Manually rollout the LSTM for the forecast period
        for t in range(self.emb_fc.input_seq_length):
            x_d = []

            for group_name in dynamic_inputs:
                if group_name == self.emb_fc.cfg.ar_group_name:
                    x_t = ar_t
                else:
                    x_t = x_fc_groups[group_name][:, t : t + 1, :]

                # Mask values. First using the group probability mask. Then recomputing a new mask to consider NaNs
                # in the original data as well.
                x_t = torch.where(group_masks[group_name], torch.nan, x_t)
                step_mask = x_t.isnan().any(dim=-1, keepdim=True)

                # Masked-mean embedding
                x_t = torch.where(step_mask, 0.0, x_t)
                x_t_emb = self.emb_fc.emb_x_d["x_d_fc"][group_name](x_t)
                x_t_emb = torch.where(step_mask, torch.nan, x_t_emb)
                x_d.append(x_t_emb)

            x_d_fc = torch.nanmean(torch.stack(x_d, dim=0), dim=0)
            x_d_fc = torch.where(torch.isnan(x_d_fc), 0.0, x_d_fc)

            # Concatenate remaining information
            flag_t = fc_freq_flag[:, t : t + 1, :]
            x_s_t = hc_info["x_s"][:, :1, :]
            x_d_fc = torch.cat([x_d_fc, flag_t, x_s_t], dim=2)
            x_d_fc = self.fc_counter(x_d_fc, step=t)

            # Forward pass of forecast
            out_fc, (hs, cs) = self.lstm(x_d_fc, (hs, cs))
            out_fc = self.dropout(out_fc)

            # Retrieve Q_t
            step_out = self.head(out_fc)
            Q_t = step_out["y_hat"]
            pred[:, t : t + 1, :] = Q_t

            # In case of MDN head, store the weights and parameters for later use
            if self.head_type == "mdn":
                preds_weights.append(step_out["weights"])
                for k, v in preds_params.items():
                    v.append(step_out["params"][k])

            ar_t = torch.cat([Q_t, ar_t[:, :, :-1]], dim=2)

        model_out["y_hat"] = pred[:, -self.predict_last_n :, :]

        if self.head_type == "mdn":
            model_out["weights"] = torch.cat(preds_weights, dim=1)[:, -self.predict_last_n :, :, :]
            model_out["params"] = {
                k: torch.cat(v, dim=1)[:, -self.predict_last_n :, :, :] for k, v in preds_params.items()
            }

        return model_out

    def _ar_masked(self, ref: torch.Tensor) -> torch.Tensor:
        """Check if the autoregressive discharge group is dropped or masked."""
        if self.emb_hc.cfg.nan_probabilistic_masking:
            return self.emb_hc.group_mask[self.emb_fc.cfg.ar_group_name].reshape(-1, 1, 1)
        return torch.zeros_like(ref, dtype=torch.bool)
