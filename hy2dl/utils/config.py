import os
import random
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import yaml
from hy2dl.utils.logging import get_logger


class Config(object):
    """Read run configuration from the specified path or dictionary and parse it into a configuration object.

    Parameters
    ----------
    yml_path_or_dict : Union[str, dict]
        Either a path to the config file or a dictionary of configuration values.

     This class and its methods are based on Neural Hydrology [#]_ and adapted for our specific case.

    References
    ----------
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022
    """

    def __init__(self, yml_path_or_dict: dict):
        # read the config from a dictionary
        self._cfg = self._read_yaml(yml_path_or_dict)

        # Check if the config contains any unknown keys
        Config._check_cfg_keys(cfg=self._cfg)

        # Check consistency in sequence length
        self._check_seq_length()

        # Check consistency in embeddings
        self._check_embeddings()

        # Check device
        self._device = Config._check_device(device=self._cfg.get("device", "cpu"))
        self._check_num_workers()

        # Create folder to save the results
        self._create_folder()

        # Initialize logger
        self.logger = get_logger(self.path_save_folder)

    def dump(self) -> None:
        """Write the current configuration to a YAML file."""
        temp_cfg = {}
        for key, value in self._cfg.items():
            if isinstance(value, Path):
                temp_cfg[key] = str(value)
            else:
                temp_cfg[key] = value

        with open(self.path_save_folder / "config.yml", "w") as file:
            yaml.dump(temp_cfg, file, default_flow_style=False, sort_keys=False)

    def _check_embeddings(self):
        if isinstance(self.dynamic_input, dict) and self.dynamic_embedding is None:
            raise ValueError("`dynamic_input` as dictionary is only supported when `dynamic_embedding` is specified")

        if self.autoregressive_input and self.dynamic_embedding is None:
            raise ValueError("`autoregressive_input` is only supported if `dynamic_embedding` is specified")

        if self.static_input is None and self.static_embedding is not None:
            raise ValueError("`static_embedding` requires specification of `static_input`")

        if isinstance(self.dynamic_input, dict) and isinstance(self.custom_seq_processing, dict):
            if set(self.dynamic_input.keys()) != set(self.custom_seq_processing.keys()):
                raise ValueError(
                    "The dictionaries `dynamic_input` and `custom_seq_processing` must have the same keys."
                )

        if (
            self.forecast_input
            and self.dynamic_embedding is None
            and len(self.forecast_input) != len(self.dynamic_input)
        ):
            raise ValueError(
                "`dynamic_input` and `forecast_input` have different dimensions. This is supported only if `dynamic_embedding` is specified"
            )

    def _check_num_workers(self):
        """Checks if the number of workers that will be used in the dataloaders is valid."""
        num_workers = self.num_workers
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
        elif num_workers > 0 and os.cpu_count() < num_workers:
            raise RuntimeError(f"num_workers ({num_workers}) must be less than number of cores ({os.cpu_count}).")

    def _check_seq_length(self):
        """Checks the consistency of sequence length when custom_seq_processing is used."""
        if self.custom_seq_processing:
            seq_length = sum(v["n_steps"] * v["freq_factor"] for v in self.custom_seq_processing.values())
            if seq_length != self.seq_length_hindcast:
                raise ValueError(
                    (
                        f"seq_length_hindcast: {self.seq_length_hindcast} does not match the sum "
                        f"of custom_seq_processing ({seq_length})."
                    )
                )

    def _create_folder(self):
        """Create a folder to store the results.

        Checks if the folder where one will store the results exist. If it does not, it creates it.

        Parameters
        ----------
        cfg : Config
            Configuration file.

        """
        # Create folder structure to store the results
        if not os.path.exists(self.path_save_folder):
            os.makedirs(self.path_save_folder)
            print(f"Folder '{self.path_save_folder}' was created to store the results.")

        if not os.path.exists(self.path_save_folder / "model"):
            os.makedirs(self.path_save_folder / "model")

    def _read_yaml(self, yml_path_or_dict: Union[str, dict]) -> dict:
        """Read the configuration from a YAML file or a dictionary."""
        if isinstance(yml_path_or_dict, (Path, str)):
            with open(yml_path_or_dict, "r") as file:
                return yaml.safe_load(file)
        elif isinstance(yml_path_or_dict, dict):
            return yml_path_or_dict
        else:
            raise ValueError("yml_path_or_dict must be a Path (path to YAML file) or a dictionary.")

    @staticmethod
    def _as_default_list(value: Any) -> list:
        """Convert a value to a list if it is not already a list."""
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    @staticmethod
    def _check_cfg_keys(cfg: dict):
        """Checks the config for unknown keys."""
        property_names = [p for p in dir(Config) if isinstance(getattr(Config, p), property)]

        unknown_keys = [k for k in cfg.keys() if k not in property_names]
        if unknown_keys:
            raise ValueError(f"{unknown_keys} are not recognized config keys.")

    @staticmethod
    def _check_device(device: str) -> str:
        """Checks the device specification and returns a valid device string."""
        if device.lower() == "cpu":
            return device.lower()

        elif device.lower() == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no CUDA devices available.")

            return "cuda:0"  # Default to the first CUDA device

        elif device.lower().startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no CUDA devices available.")

            try:
                device_index = int(device.lower().split(":")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid device format: '{device}'. Expected format 'cuda:<index>'.") from None

            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {device_index} is out of range. "
                    f"Only {torch.cuda.device_count()} CUDA device(s) available."
                )

            return device.lower()

        else:
            print(
                f"Invalid device specification: '{device}'. Expected 'cpu', 'gpu' or "
                f"'cuda[:<index>]'. CPU will be used by default."
            )
            return "cpu"

    @staticmethod
    def _get_embedding_spec(embedding: dict) -> dict:
        """Extracts the embedding specification from the configuration."""
        return {
            "hiddens": Config._as_default_list(embedding.get("hiddens", None)),
            "activation": embedding.get("activation", "relu"),
            "dropout": embedding.get("dropout", 0.0),
        }

    # From this point forward, we define properties to access the configuration values.
    @property
    def autoregressive_input(self) -> bool:
        return self._cfg.get("autoregressive_input", False)

    @property
    def batch_size_training(self) -> int:
        return self._cfg.get("batch_size_training")

    @property
    def batch_size_evaluation(self) -> int:
        return self._cfg.get("batch_size_evaluation", self.batch_size_training)

    @property
    def conceptual_model(self) -> str:
        return self._cfg.get("conceptual_model")

    @property
    def custom_seq_processing(self) -> Optional[dict[str, dict[str, int]]]:
        return self._cfg.get("custom_seq_processing")

    @property
    def custom_seq_processing_flag(self) -> bool:
        return self._cfg.get("custom_seq_processing_flag")

    @property
    def dataset(self) -> str:
        return self._cfg.get("dataset")

    @property
    def device(self) -> str:
        return self._device

    @property
    def dropout_rate(self) -> float:
        return self._cfg.get("dropout_rate", 0.0)

    @property
    def dynamic_input(self) -> Union[list[str], dict[str, list[str]]]:
        return self._cfg.get("dynamic_input")

    @property
    def dynamic_input_conceptual_model(self) -> dict[str, str | list[str]]:
        return self._cfg.get("dynamic_input_conceptual_model")

    @property
    def dynamic_parameterization_conceptual_model(self) -> List[str]:
        return Config._as_default_list(self._cfg.get("dynamic_parameterization_conceptual_model"))

    @property
    def dynamic_embedding(self) -> Optional[dict[str, Union[str, float, List[int]]]]:
        embedding = self._cfg.get("dynamic_embedding")
        return None if embedding is None else Config._get_embedding_spec(embedding)

    @property
    def epochs(self) -> int:
        return self._cfg.get("epochs")

    @property
    def experiment_name(self) -> str:
        return self._cfg.get("experiment_name", "experiment_" + str(random.randint(0, 10_000)))

    @property
    def forcings(self) -> List[str]:
        return Config._as_default_list(self._cfg.get("forcings"))

    @property
    def forecast_input(self) -> List[str]:
        return Config._as_default_list(self._cfg.get("forecast_input"))

    @property
    def hidden_size(self) -> int:
        return self._cfg.get("hidden_size")

    @property
    def initial_forget_bias(self) -> float:
        return self._cfg.get("initial_forget_bias", None)

    @property
    def learning_rate(self) -> Union[float, dict[str, float]]:
        return self._cfg.get("learning_rate", 0.001)

    @property
    def max_updates_per_epoch(self) -> int:
        return self._cfg.get("max_updates_per_epoch")

    @property
    def model(self) -> str:
        return self._cfg.get("model")

    @property
    def nan_sequence_probability(self) -> float:
        return self._cfg.get("nan_sequence_probability", None)

    @property
    def nan_step_probability(self) -> float:
        return self._cfg.get("nan_step_probability", None)

    @property
    def num_conceptual_models(self) -> int:
        return self._cfg.get("num_conceptual_models", 1)

    @property
    def num_workers(self) -> int:
        return self._cfg.get("num_workers", 0)

    @property
    def path_data(self) -> Path:
        if self._cfg.get("path_data"):
            return Path(self._cfg.get("path_data"))
        else:
            return None

    @property
    def path_additional_features(self) -> Path:
        path = self._cfg.get("path_additional_features")
        return Path(path) if path else None

    @property
    def path_entities(self) -> Path:
        path = self._cfg.get("path_entities")
        return Path(path) if path else None

    @property
    def path_entities_testing(self) -> Path:
        path = self._cfg.get("path_entities_testing")
        return Path(path) if path else self.path_entities

    @property
    def path_entities_training(self) -> Path:
        path = self._cfg.get("path_entities_training")
        return Path(path) if path else self.path_entities

    @property
    def path_entities_validation(self) -> Path:
        path = self._cfg.get("path_entities_validation")
        return Path(path) if path else self.path_entities

    @property
    def path_save_folder(self) -> Path:
        path = self._cfg.get("path_save_folder")
        return Path(path) if path else Path(f"../results/{self.experiment_name}_seed_{self.random_seed}")

    @property
    def predict_last_n(self) -> int:
        return self._cfg.get("predict_last_n", 1)

    @predict_last_n.setter
    def predict_last_n(self, value: int):
        self._cfg["predict_last_n"] = value

    @property
    def optimizer(self) -> str:
        return self._cfg.get("optimizer", "adam")

    @property
    def output_features(self) -> int:
        return self._cfg.get("output_features", 1)

    @property
    def random_seed(self) -> int:
        return self._cfg.get("random_seed", int(np.random.uniform(0, 1e6)))

    @property
    def routing_model(self) -> str:
        return self._cfg.get("routing_model", None)

    @property
    def static_embedding(self) -> Optional[dict[str, Union[str, float, List[int]]]]:
        embedding = self._cfg.get("static_embedding")
        return None if embedding is None else Config._get_embedding_spec(embedding)

    @property
    def static_input(self) -> List[str]:
        return Config._as_default_list(self._cfg.get("static_input"))

    @property
    def seq_length(self) -> int:
        return self._cfg.get("seq_length")

    @seq_length.setter
    def seq_length(self, value: int):
        self._cfg["seq_length"] = value

    @property
    def seq_length_hindcast(self) -> int:
        return self._cfg.get("seq_length_hindcast", self.seq_length)

    @property
    def seq_length_forecast(self) -> int:
        return self._cfg.get("seq_length_forecast", 0)

    @property
    def steplr_step_size(self) -> Optional[int]:
        return self._cfg.get("steplr_step_size", None)

    @property
    def steplr_gamma(self) -> Optional[float]:
        return self._cfg.get("steplr_gamma", None)

    @property
    def target(self) -> List[str]:
        return Config._as_default_list(self._cfg.get("target"))

    @property
    def teacher_forcing_scheduler(self) -> dict[str, float]:
        return self._cfg.get("teacher_forcing_scheduler", None)

    @property
    def testing_period(self) -> List[str]:
        return self._cfg.get("testing_period")

    @property
    def training_period(self) -> List[str]:
        return self._cfg.get("training_period")

    @property
    def unique_prediction_blocks(self) -> bool:
        return self._cfg.get("unique_prediction_blocks", False)

    @unique_prediction_blocks.setter
    def unique_prediction_blocks(self, value: bool) -> None:
        self._cfg["unique_prediction_blocks"] = value

    @property
    def validate_every(self) -> int:
        return self._cfg.get("validate_every", 1)

    @property
    def validate_n_random_basins(self) -> int:
        return self._cfg.get("validate_n_random_basins", 0)

    @property
    def validation_period(self) -> List[str]:
        return self._cfg.get("validation_period")
