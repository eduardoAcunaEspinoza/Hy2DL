from itertools import chain
from typing import Optional

import torch
import torch.nn as nn

from hy2dl.utils.config import Config


class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    This layer prepares the data before passing it to the main models. This can include running the dynamic and static
    attributes through embedding networks, preprocessing and assembling data at different temporal frequencies (e.g.
    daily, hourly), doing probabilistic masking and handling missing data.

    In the simplest case, the layer takes the dictionary containing the sample information and assembles the tensor to
    be sent to the main model.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    embedding_type : str
        Type of embedding to use (hindcast or forecast).

    """

    def __init__(self, cfg: Config, embedding_type: str = "hindcast"):
        super().__init__()

        self.embedding_type = embedding_type
        if embedding_type == "hindcast":
            self.dynamic_input = cfg.dynamic_input
            self._x_d_key = "x_d"
        elif embedding_type == "forecast":
            self.dynamic_input = cfg.forecast_input
            self._x_d_key = "x_d_fc"
        else:
            raise ValueError("embedding_type must be either 'hindcast' or 'forecast'")

        # Get dynamic input size
        self.dynamic_input_size = (
            len(self.dynamic_input) if cfg.dynamic_embedding is None else cfg.dynamic_embedding["hiddens"][-1]
        )

        # Get static input size
        if not cfg.static_input:
            self.static_input_size = 0
        elif isinstance(cfg.static_input, list) and cfg.static_embedding is None:
            self.static_input_size = len(cfg.static_input)
        else:
            self.static_input_size = cfg.static_embedding["hiddens"][-1]

        # Get embedding networks
        self._get_embeddings(cfg)

        # Get binary flags associated with custom_seq_processing
        self.flag_info = self._build_freq_flags(cfg)

        # Output size of the input layer
        self.output_size = self.dynamic_input_size + self.static_input_size + self.flag_info["n_flags"]

        # Save config
        self.cfg = cfg

    def forward(
        self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], assemble: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass of embedding networks.

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        assemble: bool
            Whether to assemble the different tensors into a single tensor or return a dictionary with the different

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Either the processed tensor or a dictionary with the different tensors that then have the be assembled
            manually

        """
        # -------------------------
        # Dynamic inputs
        # -------------------------
        if self.cfg.nan_handling_method == "masked_mean":
            x_d = self._masked_mean(sample)
        elif self.cfg.nan_handling_method == "input_replacement":
            x_d = self._input_replacement(sample)
        else:
            x_d = torch.cat([v(torch.stack(list(sample[k].values()), dim=-1)) for k, v in self.emb_x_d.items()], dim=1)

        # -------------------------
        # Frequency flags
        # -------------------------
        freq_flag = (
            self.flag_info["flag"].unsqueeze(0).expand(x_d.shape[0], -1, -1)
            if self.flag_info.get("flag") is not None
            else x_d.new_zeros(x_d.shape[0], x_d.shape[1], 0)
        )

        # -------------------------
        # Static inputs
        # -------------------------
        x_s = (
            self.emb_x_s(sample["x_s"]).unsqueeze(1).expand(-1, x_d.shape[1], -1)
            if self.cfg.static_input
            else x_d.new_zeros(x_d.shape[0], x_d.shape[1], 0)
        )

        return torch.cat([x_d, freq_flag, x_s], dim=2) if assemble else {"x_d": x_d, "freq_flag": freq_flag, "x_s": x_s}

    def _build_freq_flags(self, cfg: Config) -> dict[str, torch.Tensor]:
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
            return {"n_flags": 0}

        elif self.embedding_type == "hindcast":
            mask_length = sum(v["n_steps"] for v in cfg.custom_seq_processing.values())
            flag = torch.zeros((mask_length, len(cfg.custom_seq_processing)), device=cfg.device)
            i = 0
            for k, v in enumerate(cfg.custom_seq_processing.values()):
                flag[i : i + v["n_steps"], k] = 1
                i += v["n_steps"]

        elif self.embedding_type == "forecast":
            flag = torch.zeros((cfg.seq_length_forecast, len(cfg.custom_seq_processing)), device=cfg.device)
            flag[:, -1] = 1

        # One binary flag is enough when there are only two types of seq_processing
        if len(cfg.custom_seq_processing) == 2:
            flag = flag[:, -1:]

        return {"flag": flag, "n_flags": flag.shape[1]}

    def _get_embeddings(self, cfg: Config):
        """Build embedding networks based on the configuration.

        Parameters
        ----------
        cfg : Config
            Configuration file.

        """

        # -------------------------
        # Embeddings for dynamic variables
        # -------------------------
        self.emb_x_d = nn.ModuleDict()

        # Case 1: Single group of variables, same variables along the sequence length, and only one sequence processing.
        # Note: Forecast period is currently only supported for single frequency!
        if (cfg.custom_seq_processing is None or self.embedding_type == "forecast") and isinstance(
            self.dynamic_input, list
        ):
            self.emb_x_d[self._x_d_key] = InputLayer.build_embedding(
                input_dim=len(self.dynamic_input), embedding=cfg.dynamic_embedding
            )

        # Case 2. Multiple groups of variables along sequence length, but only one sequence processing type
        # Example: Single frequency case, but we have multiple group of variables.
        # Note 1: We have multiple groups, therefore we use nan_handling methods!
        # Note 2: Forecast period is currently only supported for single frequency!
        elif (cfg.custom_seq_processing is None or self.embedding_type == "forecast") and isinstance(
            self.dynamic_input, dict
        ):
            # With masked_mean architecture I need one embedding per group!
            if cfg.nan_handling_method == "masked_mean":
                self.emb_x_d[self._x_d_key] = nn.ModuleDict()
                for k, v in self.dynamic_input.items():
                    self.emb_x_d[self._x_d_key][k] = InputLayer.build_embedding(
                        input_dim=len(v), embedding=cfg.dynamic_embedding
                    )

            # With input replacement I concatenate all the groups together and add a flag per group. Then I pass
            # the result through an embedding network
            elif cfg.nan_handling_method == "input_replacement":
                i_dim = len(list(chain.from_iterable(self.dynamic_input.values()))) + len(self.dynamic_input)
                self.emb_x_d[self._x_d_key] = InputLayer.build_embedding(
                    input_dim=i_dim, embedding=cfg.dynamic_embedding
                )

        # Case 3. Custom sequence processing (e.g. multiple frequencies)
        elif isinstance(cfg.custom_seq_processing, dict) and self.embedding_type == "hindcast":
            # Iterate through each frequency
            for freq in cfg.custom_seq_processing:
                # Case 3.1. If we have a single group of variables, that does not change between frequencies.
                if isinstance(self.dynamic_input, list):
                    self.emb_x_d[f"{self._x_d_key}_{freq}"] = InputLayer.build_embedding(
                        input_dim=len(self.dynamic_input), embedding=cfg.dynamic_embedding
                    )

                # Case 3.2. If we use the different variables per frequency or have more than one group
                elif isinstance(self.dynamic_input, dict):
                    # Case 3.2.1  If we only have one group of variables
                    if isinstance(self.dynamic_input[freq], list):
                        self.emb_x_d[f"{self._x_d_key}_{freq}"] = InputLayer.build_embedding(
                            input_dim=len(self.dynamic_input[freq]), embedding=cfg.dynamic_embedding
                        )

                    # Case 3.2.2  If we have multiple groups of variables
                    # Note: We have multiple groups, therefore we use nan_handling methods!
                    elif isinstance(self.dynamic_input[freq], dict):
                        # With masked_mean architecture I need one embedding per group!
                        if cfg.nan_handling_method == "masked_mean":
                            self.emb_x_d[f"{self._x_d_key}_{freq}"] = nn.ModuleDict()
                            for k, v in self.dynamic_input[freq].items():
                                self.emb_x_d[f"{self._x_d_key}_{freq}"][k] = InputLayer.build_embedding(
                                    input_dim=len(v), embedding=cfg.dynamic_embedding
                                )

                        # With input replacement I concatenate all the groups together and add a flag per group. Then I
                        # pass the result through an embedding network
                        elif cfg.nan_handling_method == "input_replacement":
                            i_dim = len(list(chain.from_iterable(self.dynamic_input[freq].values()))) + len(
                                self.dynamic_input[freq]
                            )
                            self.emb_x_d[f"{self._x_d_key}_{freq}"] = InputLayer.build_embedding(
                                input_dim=i_dim, embedding=cfg.dynamic_embedding
                            )

        # -------------------------
        # Embeddings for static variables
        # -------------------------
       
        # Case 1: A set of static features is specified, but no static embedding is specified
        if cfg.static_input and cfg.static_embedding is None:
            self.emb_x_s = InputLayer.build_embedding(input_dim=len(cfg.static_input))

        # Case 2: A set of static features is specified as well as a static embedding
        elif cfg.static_input and cfg.static_embedding is not None:
            self.emb_x_s = InputLayer.build_embedding(input_dim=len(cfg.static_input), embedding=cfg.static_embedding)


    def _input_replacement(self, sample) -> torch.Tensor:
        """Apply input_replacement to handle missing inputs and add associated binary masks.

        Implementation based on Gauch2025 [#]_

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        torch.Tensor
            tensor of embedded dynamic inputs.

        References
        ----------
        .. [#] M. Gauch, F. Kratzert, D. Klotz, G. Nearing, D. Cohen and o. Gilon : How to deal w___ missing input data
            research in hydrology. EGUsphere, 1, 21, doi: 10.5194/egusphere-2025-1224, 2025

        """

        x_d = []
        # Compute group-level dropout mask
        if self.cfg.nan_probabilistic_masking:
            self.group_mask = self._mask_groups(sample=sample)

        for k, v in self.emb_x_d.items():
            freq_k = k.split(self._x_d_key + "_")[-1]
            # Case where I have multiple groups of variables. Can be either one frequency
            # with multiple groups or multiple frequencies with multiple groups.
            if (freq_k == self._x_d_key and isinstance(self.dynamic_input, dict)) or (
                freq_k != self._x_d_key and isinstance(self.dynamic_input[freq_k], dict)
            ):
                x_d_groups = []

                # Extract groups
                groups = self.dynamic_input[freq_k] if freq_k != self._x_d_key else self.dynamic_input
                for group_id, group_var in groups.items():
                    # Concatenate variables of the group into single tensor
                    x_d_group = torch.stack([sample[k][var] for var in group_var], dim=-1)

                    # Mask group based of nan_seq probability and nan_step probability
                    if self.cfg.nan_probabilistic_masking:
                        probability_mask = (
                            torch.rand(x_d_group.shape[0], x_d_group.shape[1], device=self.cfg.device)
                            < self.cfg.nan_probability[group_id]["nan_step"]
                        ) | self.group_mask[group_id].unsqueeze(1)
                        x_d_group[probability_mask] = torch.nan

                    # I recompute the mask, becase there can also be NaNs in the original data that we need to consider.
                    mask = x_d_group.isnan().any(dim=-1, keepdim=True)
                    x_d_group = torch.where(mask, 0.0, x_d_group)

                    # concatenate nan mask
                    x_d_group = torch.cat([x_d_group, mask.float()], dim=-1)
                    x_d_groups.append(x_d_group)

                # Concatenate groups and pass them through embedding
                x_d.append(v(torch.cat(x_d_groups, dim=2)))

            # Case in which I only have one group of variables. This can happen, for example, if for one frequency I
            # have groups and for another frequency I do not.
            else:
                x_d.append(v(torch.stack(list(sample[k].values()), dim=-1)))

        return torch.cat(x_d, dim=1)

    def _masked_mean(self, sample) -> torch.Tensor | list[torch.Tensor]:
        """Apply the masked-mean function to handle missing inputs.

        This architecture uses a masked mean approach to handle missing values in the dynamic inputs. It passes
        the different input groups through embedding networks and then apply the torch.nanmean function between
        the embeddings to get rid of nan values.

        Implementation based on Gauch2025 [#]_

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        torch.Tensor
            Masked-mean tensor of dynamic inputs.

        References
        ----------
        .. [#] M. Gauch, F. Kratzert, D. Klotz, G. Nearing, D. Cohen and o. Gilon : How to deal w___ missing input data
            research in hydrology. EGUsphere, 1, 21, doi: 10.5194/egusphere-2025-1224, 2025

        """

        # Process dynamic inputs ---
        x_d = []

        # Compute group-level dropout mask
        if self.cfg.nan_probabilistic_masking:
            self.group_mask = self._mask_groups(sample=sample)

        for k, v in self.emb_x_d.items():
            freq_k = k.split(self._x_d_key + "_")[-1]
            # Case where I have multiple groups of variables
            if isinstance(v, nn.ModuleDict):
                x_d_groups = []
                for group, emb in v.items():
                    # Extract variables of interest
                    group_var = (
                        self.dynamic_input[freq_k][group] if freq_k != self._x_d_key else self.dynamic_input[group]
                    )

                    # Concatenate variables of the group into single tensor
                    x_d_group = torch.stack([sample[k][var] for var in group_var], dim=-1)

                    # Mask group based of nan_seq probability and nan_step probability
                    if self.cfg.nan_probabilistic_masking:
                        probability_mask = (
                            torch.rand(x_d_group.shape[0], x_d_group.shape[1], device=self.cfg.device)
                            < self.cfg.nan_probability[group]["nan_step"]
                        ) | self.group_mask[group].unsqueeze(1)
                        x_d_group[probability_mask] = torch.nan

                    # Before the forward pass of the embedding we set Nans to zero to avoid NaN gradients. I recompute
                    # the mask, becase there can also be NaNs in the original data that we need to consider.
                    mask = x_d_group.isnan().any(dim=-1, keepdim=True)
                    x_d_group = torch.where(mask, 0.0, x_d_group)
                    # Forward pass
                    group_embedding = emb(x_d_group)
                    # After the forward pass, the introduced zeros are again substitued by NaNs, so they can be ignored
                    # by the masked-mean architecture
                    x_d_groups.append(torch.where(mask, torch.nan, group_embedding))

                x_d.append(x_d_groups)

            # Case in which I only have one group of variables. This can happen, for example, if for one frequency I
            # have groups and for another frequency I do not.
            else:
                x_d.append([v(torch.stack(list(sample[k].values()), dim=-1))])

        dynamic_output = torch.cat([torch.nanmean(torch.stack(x, dim=0), dim=0) for x in x_d], dim=1)
        # We can have nans, even with the masked-mean architecture, if for a given timestep all the groups had nans.
        # In this case we substitute the nans with zeros to make sure the model still runs.
        dynamic_output = torch.where(torch.isnan(dynamic_output), 0.0, dynamic_output)
        return dynamic_output

    def _mask_groups(self, sample):
        """Create a mask for the input groups based on their nan_seq probabilities.

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        Returns
        -------
        dict[str, bool]
            Dictionary with boolean values indicating whether to drop each input group.

        """
        nan_seq_probs = torch.tensor([v["nan_seq"] for v in self.cfg.nan_probability.values()], device=self.cfg.device)
        drop_group = torch.rand(sample["y_obs"].shape[0], len(nan_seq_probs), device=self.cfg.device) < nan_seq_probs
        all_dropped = drop_group.all(dim=1)
        if all_dropped.any():  # Don't allow all groups to be dropped out.
            # For samples with all groups dropped, randomly un-drop one group
            drop_group[
                torch.where(all_dropped)[0], torch.randint(0, drop_group.size(1), (1,), device=self.cfg.device)
            ] = False

        return dict(zip(self.cfg.nan_probability.keys(), drop_group.T, strict=True))

    @staticmethod
    def build_embedding(input_dim: int, embedding: Optional[dict[str, str | float | list[int]]]):
        """Build embedding

        Parameters
        ----------
        input_dim: int
            Input dimension of the first layer.
        embedding: dict[str, str | float | list[int]]
            Dictionary with the embedding characteristics

        Returns
        -------
        nn.Sequential | nn.Identity
            Embedding network or nn.Identity

        """

        return (
            InputLayer.build_ffnn(
                input_dim=input_dim,
                spec=embedding["hiddens"],
                activation=embedding["activation"],
                dropout=embedding["dropout"],
            )
            if isinstance(embedding, dict)
            else nn.Identity()
        )

    @staticmethod
    def build_ffnn(input_dim: int, spec: list[int], activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
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
