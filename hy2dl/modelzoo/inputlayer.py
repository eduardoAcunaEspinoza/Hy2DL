from typing import List

import torch
import torch.nn as nn

from hy2dl.utils.config import Config


class InputLayer(nn.Module):
    """Input layer to preprocess what goes into the main model.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """

    def __init__(self, cfg: Config):
        super().__init__()

        # Get dynamic input size (if we use dynamic embeddings, all the embeddings have the same dimension)
        self.dynamic_input_size = (
            len(cfg.dynamic_input) if cfg.dynamic_embedding is None else cfg.dynamic_embedding["hiddens"][-1]
        )

        # Get static input size
        if not cfg.static_input:
            self.static_input_size = 0
        elif isinstance(cfg.static_input, list) and cfg.static_embedding is None:
            self.static_input_size = len(cfg.static_input)
        else:
            self.static_input_size = cfg.static_embedding["hiddens"][-1]

        # Get embedding network
        self._build_embeddings(cfg)

        # Get binary flags
        self.flag_info = InputLayer._build_flags(cfg)

        # Output size of the input layer
        self.output_size = self.dynamic_input_size + self.static_input_size + self.flag_info["n_flags"]
        self.cfg = cfg

    def _build_embeddings(self, cfg: Config):
        """Builds embedding networks based on the configuration.

        Parameters
        ----------
        cfg : Config
            Configuration file.
        """
        # dynamic embeddings ---------------------
        self.emb_hc_x_d = nn.ModuleDict()
        if cfg.custom_seq_processing is None:
            self.emb_hc_x_d["x_d"] = (
                InputLayer.build_ffnn(input_dim=len(cfg.dynamic_input),
                                      spec=cfg.dynamic_embedding["hiddens"],
                                      activation=cfg.dynamic_embedding["activation"],
                                      dropout=cfg.dynamic_embedding["dropout"]
                                      )
                if isinstance(cfg.dynamic_embedding, dict)
                else nn.Identity()
            )

        elif isinstance(cfg.custom_seq_processing , dict):
            for k in cfg.custom_seq_processing:
                self.emb_hc_x_d[f"x_d_{k}"] = (
                    InputLayer.build_ffnn(input_dim=len(cfg.dynamic_input) if isinstance(cfg.dynamic_input, list) else len(cfg.dynamic_input[k]),
                                          spec=cfg.dynamic_embedding["hiddens"],
                                          activation=cfg.dynamic_embedding["activation"],
                                          dropout=cfg.dynamic_embedding["dropout"]
                                          )
                    if isinstance(cfg.dynamic_embedding, dict)
                    else nn.Identity()
                )

        # static embeddings ---------------------
        if cfg.static_input:
            self.emb_x_s = (
                InputLayer.build_ffnn(input_dim=len(cfg.static_input),
                                      spec=cfg.static_embedding["hiddens"],
                                      activation=cfg.static_embedding["activation"],
                                      dropout=cfg.static_embedding["dropout"]
                                     )
               if isinstance(cfg.static_embedding, dict)
                else nn.Identity()
            )

        # autoregressive embeddings ---------------------
        if cfg.autoregressive_input:
            self.emb_hc_x_ar = nn.ModuleDict()
            if cfg.custom_seq_processing is None:
                self.emb_hc_x_ar["x_ar"] = InputLayer.build_ffnn(input_dim=1,
                                                                 spec=cfg.dynamic_embedding["hiddens"],
                                                                 activation=cfg.dynamic_embedding["activation"],
                                                                 dropout=cfg.dynamic_embedding["dropout"]
                                                                 )

            elif isinstance(cfg.custom_seq_processing , dict):
                for k in cfg.custom_seq_processing:
                    self.emb_hc_x_ar[f"x_ar_{k}"] = InputLayer.build_ffnn(input_dim=1,
                                                                          spec=cfg.dynamic_embedding["hiddens"],
                                                                          activation=cfg.dynamic_embedding["activation"],
                                                                          dropout=cfg.dynamic_embedding["dropout"]
                                                                          )
        # forecast embeddings ---------------------
        if cfg.forecast_input:
            self.emb_fc_x_d = (
                InputLayer.build_ffnn(input_dim=len(cfg.forecast_input),
                                      spec=cfg.dynamic_embedding["hiddens"],
                                      activation=cfg.dynamic_embedding["activation"],
                                      dropout=cfg.dynamic_embedding["dropout"]
                                      )
                if isinstance(cfg.dynamic_embedding, dict)
                else nn.Identity()
            )
             
            if cfg.autoregressive_input:
                self.emb_fc_x_ar = InputLayer.build_ffnn(input_dim=1,
                                                         spec=cfg.dynamic_embedding["hiddens"],
                                                         activation=cfg.dynamic_embedding["activation"],
                                                         dropout=cfg.dynamic_embedding["dropout"]
                                                         )
                                    


    @staticmethod
    def build_ffnn(input_dim: int, spec: List[int], activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
        """Builds a feedforward neural network based on the given specification.

        Parameters
        ----------
        input_dim: int
            Input dimension of the first layer.
        spec: List[int]
            Dimension of the different hidden layers.
        activation: str
            Activation function to use between layers (relu, linear, tanh, sigmoid).
            Default is 'relu'.
        dropout: float
            Dropout rate to apply after each layer (except the last one).
            Default is 0.0 (no dropout).

        Returns
        -------
        nn.Sequential
            A sequential model containing the feedforward neural network layers.
        """
        activation = InputLayer._get_activation_function(activation)
        ffnn_layers = []
        for i, out_dim in enumerate(spec):
            ffnn_layers.append(nn.Linear(input_dim, out_dim))
            if i != len(spec) - 1:  # add activation, except after the last linear
                ffnn_layers.append(activation)
                if dropout > 0.0:
                    ffnn_layers.append(nn.Dropout(dropout))

            input_dim = out_dim  # updates next layer’s input size

        return nn.Sequential(*ffnn_layers)

    @staticmethod
    def _build_flags(cfg: Config) -> dict[str, torch.Tensor]:
        """Builds flag channels.

        Parameters
        ----------
        cfg : Config
            Configuration file.

        Returns
        -------
        flag_info : dict[str, torch.Tensor]
            Dictionary containing the flag channels and their size.
            If no custom sequence processing is defined, returns a dictionary with n_flags = 0.
        """

        if not cfg.custom_seq_processing_flag:
            return {"n_flags":0}

        # Flags during hindcast period
        flag_info = {}
        mask_length = sum(v["n_steps"] for v in cfg.custom_seq_processing.values())
        flag_hc = torch.zeros((mask_length, len(cfg.custom_seq_processing)), device=cfg.device)
        i = 0
        for k, v in enumerate(cfg.custom_seq_processing.values()):
            flag_hc[i : i + v["n_steps"], k] = 1
            i += v["n_steps"]
        
        # If we only have two type of seq_processing, we only need one binary flag
        if len(cfg.custom_seq_processing) == 2:
            flag_hc = flag_hc[:, -1:]

        flag_info["flag_hc"] = flag_hc
        flag_info["n_flags"] = flag_hc.shape[1]

        # Flag during forecast period
        if cfg.seq_length_forecast > 0:
            flag_fc= flag_hc[-1, :].unsqueeze(0).repeat(cfg.seq_length_forecast, 1)
            flag_info["flag_fc"] = flag_fc
        

        return flag_info 

    @staticmethod
    def _get_activation_function(activation: str) -> nn.Module:
        """Returns the activation function based on the given string.

        Parameters
        ----------
        activation: str
            Name of the activation function (e.g., 'relu', 'linear', 'tanh', 'sigmoid').

        Returns
        -------
        nn.Module
            The corresponding activation function module.
        """
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation == "linear":
            return nn.Identity()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Forward pass of embedding networks.

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        processed_sample: dict[str, torch.Tensor]
            Dictionary with the different tensors processed by the embedding networks.
        """
        processed_sample = {}
        
        # Hindcast period, dynamic inputs
        x_d = []
        for k, v in self.emb_hc_x_d.items():
            x_d.append(v(torch.stack(list(sample[k].values()), dim=-1)))
            
        # Hindcast period, autoregressive inputs
        if self.cfg.autoregressive_input:
            x_ar = []
            for k, v in self.emb_hc_x_ar.items():
                x_ar.append(
                    v(torch.nan_to_num(sample[k], nan=0.0))
                    .masked_fill(torch.isnan(sample[k]), float('nan'))
                    )
                
            # Masked-mean between dynamic and autoregressive inputs
            x_d = InputLayer.masked_mean_embedding(x_d, x_ar)

        # concatenate different chuncks of sequence in case we have custom_seq_processing
        processed_sample["x_d_hc"] = torch.cat(x_d, dim=1)

        # Static inputs if available
        if self.cfg.static_input:
            processed_sample["x_s"] = self.emb_x_s(sample["x_s"])
        
        # Forecast period, dynamic inputs
        if self.cfg.forecast_input:
            processed_sample["x_d_fc"] = self.emb_fc_x_d(torch.stack(list(sample["x_d_fc"].values()), dim=-1))

        return processed_sample

    
    def assemble_sample(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """Assembles the sample for the forward pass.

        Parameters
        ----------
        sample: dict[str, torch.Tensor]
            Dictionary with the different tensors

        Returns
        -------
        x: torch.Tensor
            Asembled tensor

        """
        x_d = sample["x_d_hc"]

        # Add hindcast flags (if any)
        if self.flag_info.get("flag_hc") is not None:
            x_d = torch.cat([x_d, self.flag_info["flag_hc"].unsqueeze(0).expand(x_d.shape[0], -1, -1)], dim=2)
        
        # Add forecast input (if any)
        if sample.get("x_d_fc") is not None:
            x_d_fc = sample["x_d_fc"]
            # Add forecast flags (if any)
            if self.flag_info.get("flag_fc") is not None:
                x_d_fc = torch.cat([x_d_fc, self.flag_info["flag_fc"].unsqueeze(0).expand(x_d_fc.shape[0], -1, -1)], dim=2)

            x_d = torch.cat([x_d, x_d_fc], dim=1)

        # Add static inputs (if any)
        if sample.get("x_s") is not None:
            x_d = torch.cat([x_d, sample["x_s"].unsqueeze(1).expand(-1, x_d.shape[1], -1)], dim=2)

        return x_d
    
    @staticmethod
    def masked_mean_embedding(*tensor_lists: List[torch.Tensor]) -> List[torch.Tensor]:
        """Computes the element-wise mean of the different tensors, masking NaN values.

        Parameters
        ----------
        *tensor_lists: List[torch.Tensor]
            Lists of tensors

        Returns
        -------
        x_d: List[torch.Tensor]
        """
        x_d = [torch.nanmean(torch.stack(tensors, dim=0), dim=0) for tensors in zip(*tensor_lists)]
        return x_d
