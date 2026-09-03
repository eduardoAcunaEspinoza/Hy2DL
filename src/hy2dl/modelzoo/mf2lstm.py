from typing import Any

import torch
import torch.nn as nn

from hy2dl.modelzoo import get_head
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
        self.emb_fc = InputLayer(cfg, embedding_type="forecast")

        # Create forecast counter if specified
        self.forecast_counter = 0
        if cfg.forecast_counter:
            self.forecast_counter = 1
            fc_counter = torch.zeros(self.emb_hc.input_seq_length + self.emb_fc.input_seq_length, dtype=torch.float32)
            fc_counter[self.emb_hc.input_seq_length :] = torch.arange(
                1, self.emb_fc.input_seq_length + 1, dtype=torch.float32
            )
            self.register_buffer("fc_counter", fc_counter)

        # Cuda LSTM
        self.lstm = nn.LSTM(
            input_size=self.emb_hc.output_size + self.forecast_counter,
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
        return self.forward_tf(x) if self.training and self.ar_teacher_forcing else self.forward_ar(x)

    def forward_tf(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Teacher-forcing mode"""
        # Preprocess data for hindcast and forecast periods
        x_lstm = self.emb_hc(sample)
        x_fc = self.emb_fc(sample)

        # Concatenate both periods along the sequence dimension
        x_lstm = torch.cat((x_lstm, x_fc), dim=1)

        # Include forecast counter if specified
        if self.forecast_counter == 1:
            x_lstm = torch.cat((x_lstm, self.fc_counter.view(1, -1, 1).expand(x_lstm.shape[0], -1, -1)), dim=2)

        # Forward pass through the LSTM
        hs, _ = self.lstm(x_lstm)
        # Extract sequence of interest
        hs = hs[:, -self.predict_last_n :, :]
        hs = self.dropout(hs)

        return self.head(hs)

    def forward_ar(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Autoregressive mode"""
        model_out = {}
        ### ---------------
        ### Hindncast period
        ### ---------------
        hc_info = self.emb_hc(sample, assemble=False)
        x_hc = torch.cat([hc_info["x_d"], hc_info["freq_flag"], hc_info["x_s"]], dim=2)
        batch_size = x_hc.shape[0]

        if self.forecast_counter == 1:
            x_hc = torch.cat(
                [x_hc, self.fc_counter[: self.emb_hc.input_seq_length].view(1, -1, 1).expand(batch_size, -1, -1)],
                dim=2,
            )

        # Forward pass of hindcast period
        _, (hs, cs) = self.lstm(x_hc)

        ### ---------------
        ### Forecast period
        ### ---------------
        dynamic_inputs = self.emb_fc.dynamic_input
        x_fc_groups = {}
        group_masks = {}

        # Pre-extract sequences and masks for all groups
        for g_name, vars_list in dynamic_inputs.items():
            g_vars = torch.stack([sample["x_d_fc"][var] for var in vars_list], dim=-1)

            # Group masks to be consistent with hindcast period
            if self.emb_fc.cfg.nan_probabilistic_masking:
                g_mask = self.emb_hc.group_mask[g_name].unsqueeze(1).unsqueeze(2)
            else:
                g_mask = g_vars.new_zeros(g_vars.shape[0], 1, 1, dtype=torch.bool)

            group_masks[g_name] = g_mask

            if g_name == "ar":
                ar_t = g_vars[:, :1, :]  # Past discharges used as input in first forecast time
            else:
                x_fc_groups[g_name] = g_vars

        # Forecast counter
        if self.forecast_counter == 1:
            fc_counter = self.fc_counter[self.emb_hc.input_seq_length :].view(1, -1, 1).expand(batch_size, -1, -1)

        # Preallocate prediction tensor
        pred = torch.empty((batch_size, self.emb_fc.input_seq_length, self.head.output_features), device=x_hc.device)
        if self.head_type == "mdn":
            preds_weights = []
            preds_params = {k: [] for k in self.head.distribution.parameters}

        # Manually rollout the LSTM for the forecast period
        for t in range(self.emb_fc.input_seq_length):
            x_d = []

            for group_name in dynamic_inputs:
                if group_name == "ar":
                    x_t = ar_t
                else:
                    x_t = x_fc_groups[group_name][:, t : t + 1, :]

                # Mask values. First using the group probability mask. Then recomputing a new mask to consider NaNs
                # in the original data as well.
                x_t = torch.where(group_masks[group_name], torch.nan, x_t)
                step_mask = x_t.isnan().any(dim=-1, keepdim=True)

                # Masked-mean embedding
                x_t = torch.where(step_mask, 0.0, x_t)
                # Using bracket notation for the ModuleDict key lookup
                x_t_emb = self.emb_fc.emb_x_d["x_d_fc"][group_name](x_t)
                x_t_emb = torch.where(step_mask, torch.nan, x_t_emb)
                x_d.append(x_t_emb)

            x_d_fc = torch.nanmean(torch.stack(x_d, dim=0), dim=0)
            x_d_fc = torch.where(torch.isnan(x_d_fc), 0.0, x_d_fc)

            # Concatenate remaining information
            flag_t = self.emb_fc.flag_info["flag"][t : t + 1, :].unsqueeze(0).expand(batch_size, -1, -1)
            x_s_t = hc_info["x_s"][:, :1, :]
            x_d_fc = torch.cat([x_d_fc, flag_t, x_s_t], dim=2)

            if self.forecast_counter == 1:
                x_d_fc = torch.cat([x_d_fc, fc_counter[:, t : t + 1, :]], dim=2)

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
