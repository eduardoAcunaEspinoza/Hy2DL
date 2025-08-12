import torch
import torch.nn as nn

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config


class ARLSTM(nn.Module):
    """Autoregressive LSTM.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size, batch_first=True)

        self.dropout = torch.nn.Dropout(p=cfg.dropout_rate)

        self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.output_features)

        self.predict_last_n = cfg.predict_last_n

        self.teacher_forcing = True if cfg.teacher_forcing_scheduler is not None else False
        if self.teacher_forcing:
            self.teacher_forcing_scheduler = cfg.teacher_forcing_scheduler
            self.update_teacher_forcing_probability(epoch=1)
        else:
            self.teacher_forcing_probability = 0.0

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of certain model weights."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]])-> dict[str, torch.Tensor]:
        """Forward pass of lstm network

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        pred: Dict[str, torch.Tensor]

        """
        # Preprocess data to be sent to the LSTM
        processed_sample = self.embedding_net(sample)
        
        # Prepare tensor for hindcast period
        x_d_hc = processed_sample["x_d_hc"]
        # Concatenate flags in hindcast period
        if self.embedding_net.flag_info.get("flag_hc") is not None:
            x_d_hc = torch.cat([x_d_hc, self.embedding_net.flag_info["flag_hc"].unsqueeze(0).expand(x_d_hc.shape[0], -1, -1)], dim=2)
        # Concatenate static features
        if processed_sample.get("x_s") is not None:
            x_d_hc = torch.cat([x_d_hc, processed_sample["x_s"].unsqueeze(1).expand(-1, x_d_hc.shape[1], -1)], dim=2)

        # Forward pass of hindcast period
        out_hc, (h_n, c_n) = self.lstm(x_d_hc)

        # Calculate last simulated value for hindcast period
        Q_t = self.linear(self.dropout(out_hc[:, -1 :, :]))
        
        # run forecast period
        pred = []
        for t in range(processed_sample["x_d_fc"].shape[1]):
            x_t = processed_sample["x_d_fc"][:, t:t+1, :]

            # Teacher-forcing
            if torch.rand(1).item() < self.teacher_forcing_probability:
                Q_t = sample["x_ar_fc"][:, t:t+1, :]
            
            # Mean masked embedding
            Q_input = self.embedding_net.emb_fc_x_ar(torch.nan_to_num(Q_t, nan=0.0)).masked_fill(torch.isnan(Q_t), float('nan'))
            x_d_fc = InputLayer.masked_mean_embedding([x_t], [Q_input])[0]

            # Concatenate flags in hindcast period
            if self.embedding_net.flag_info.get("flag_fc") is not None:
                x_d_fc = torch.cat([x_d_fc, self.embedding_net.flag_info["flag_fc"][t:t+1, :].unsqueeze(0).expand(x_d_fc.shape[0], -1, -1)], dim=2)
            # Concatenate static features
            if processed_sample.get("x_s") is not None:
                x_d_fc = torch.cat([x_d_fc, processed_sample["x_s"].unsqueeze(1)], dim=2)

            # Forward pass of forecast
            out_fc, (h_n, c_n) = self.lstm(x_d_fc, (h_n, c_n))
            # Retrieve Q_t
            Q_t =  self.linear(self.dropout(out_fc[:, -1 :, :]))
            pred.append(Q_t)

        y_hat = torch.cat(pred, dim=1)[:, -self.predict_last_n :, :]
        return {"y_hat": y_hat}
    

    def update_teacher_forcing_probability(self, epoch: int) -> float:
        """Updates teacher_forcing_probability, based on a custom scheduler.

        Parameters
        ----------
        epoch: int
            Epoch for which the learning rate is needed

        """
        sorted_keys = sorted(self.teacher_forcing_scheduler.keys(), reverse=True)
        for key in sorted_keys:
            if epoch >= key:
                self.teacher_forcing_probability = self.teacher_forcing_scheduler[key]
                break
