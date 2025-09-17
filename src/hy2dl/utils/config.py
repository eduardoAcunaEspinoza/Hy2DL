import os
import random
from pathlib import Path
from typing import Any, Optional

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

    def __init__(self, yml_path_or_dict: dict, dev_mode: bool = False):
        # read the config from a dictionary
        self._cfg = self._read_yaml(yml_path_or_dict)

        # Check if the config contains any unknown keys
        if not dev_mode:
            Config._check_cfg_keys(cfg=self._cfg)

        # Multiple checks to ensure valid configuration
        self._check_dynamic_inputs()
        self._check_seq_length()
        self._check_embeddings()
        self._check_models()
        self._check_nan_settings()
        self._check_num_workers()
        self._device = Config._check_device(device=self._cfg.get("device", "cpu"))

    def init_experiment(self):
        """Create folder structure and get the logger where the experiment progress will be reported"""
        # Create folder to store the results and initialize logger
        self._create_folder()
        # Create logger
        self.logger = get_logger(self.path_save_folder, f"{self.experiment_name}_{self.random_seed}")

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

    def _check_dynamic_inputs(self):
        if isinstance(self.dynamic_input, dict):
            if self.custom_seq_processing is None and self.nan_handling_method is None:
                raise ValueError("Groups of variables are only supported with a `nan_handling_method`")
            elif self.custom_seq_processing is not None:
                for v in self.dynamic_input.values():
                    if isinstance(v, dict) and self.nan_handling_method is None:
                        raise ValueError("Groups of variables are only supported with a `nan_handling_method`")

    def _check_embeddings(self):
        if isinstance(self.dynamic_input, dict) and self.dynamic_embedding is None:
            raise ValueError("`dynamic_input` as dictionary is only supported when `dynamic_embedding` is specified")

        if self.static_input is None and self.static_embedding is not None:
            raise ValueError("`static_embedding` requires specification of `static_input`")

        if isinstance(self.dynamic_input, dict) and isinstance(self.custom_seq_processing, dict):
            if set(self.dynamic_input.keys()) != set(self.custom_seq_processing.keys()):
                raise ValueError("`dynamic_input` and `custom_seq_processing` must have the same keys.")

        if isinstance(self.nan_handling_method, str) and self.dynamic_embedding is None:
            raise ValueError("`dynamic_embedding` must be specified when using `nan_handling_method`")

        if (
            self.forecast_input
            and self.dynamic_embedding is None
            and len(self.forecast_input) != len(self.dynamic_input)
        ):
            raise ValueError(
                (
                    "`dynamic_input` and `forecast_input` have different dimensions. "
                    "This is supported only if `dynamic_embedding` is specified"
                )
            )

    def _check_models(self):
        """Check for specific configurations required by certain models."""
        # Check forecast configuration
        if self.model == "forecast_lstm" and (self.seq_length_forecast == 0 or len(self.forecast_input) == 0):
            raise ValueError("`forecast_lstm` requires `seq_length_forecast > 0` and `forecast_input` to be specified.")
        if self.model == "hybrid" and (self.conceptual_model is None or self.dynamic_input_conceptual_model is None):
            raise ValueError(
                "`hybrid` model requires `conceptual_model` and `dynamic_input_conceptual_model` to be specified."
            )

    def _check_nan_settings(self):
        """Check settings when working with nan handling methods"""
        if self.nan_handling_method is not None:
            if self.nan_handling_method not in ["masked_mean", "input_replacement"]:
                raise ValueError(
                    "Unknown `nan_handling_method`. Available options: ['masked_mean', 'input_replacement']"
                )
            if isinstance(self.nan_probability, dict):
                nan_groups = list(self.nan_probability.keys())

                input_groups = []
                # One frequency, multiple groups
                if self.custom_seq_processing is None and isinstance(self.dynamic_input, dict):
                    input_groups.extend(k for k in self.dynamic_input)
                # Multiple frequencies with groups. if I have multi-frequency the groups are defined in a nested dict.
                elif isinstance(self.custom_seq_processing, dict) and isinstance(self.dynamic_input, dict):
                    for v in self.dynamic_input.values():
                        if isinstance(v, dict):
                            input_groups.extend(k2 for k2 in v)
                # Groups of forecast
                if isinstance(self.forecast_input, dict):
                    input_groups.extend(k for k in self.forecast_input)

                if set(nan_groups) != set(input_groups):
                    raise ValueError(
                        "All groups contained in `dynamic_input` and `forecast_input` "
                        "must be specified in `nan_probability`"
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

    def _read_yaml(self, yml_path_or_dict: str | dict) -> dict:
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

    # -----------------
    # From this point forward, we define properties to access the configuration values.
    # -----------------
    @property
    def batch_size_training(self) -> int:
        return self._cfg.get("batch_size_training")

    @property
    def batch_size_evaluation(self) -> int:
        return self._cfg.get("batch_size_evaluation", self.batch_size_training)

    @property
    def conceptual_model(self) -> Optional[str]:
        return self._cfg.get("conceptual_model")

    @property
    def custom_seq_processing(self) -> Optional[dict[str, dict[str, int]]]:
        return self._cfg.get("custom_seq_processing")

    @property
    def custom_seq_processing_flag(self) -> bool:
        return self._cfg.get("custom_seq_processing_flag", False)

    @property
    def dataset(self) -> str:
        return self._cfg.get("dataset")

    @property
    def device(self) -> str:
        return self._device

    @property
    def distribution(self) -> str:
        return self._cfg.get("distribution")

    @property
    def dropout_rate(self) -> float:
        return self._cfg.get("dropout_rate", 0.0)

    @property
    def dynamic_embedding(self) -> Optional[dict[str, str | float | list[int]]]:
        embedding = self._cfg.get("dynamic_embedding")
        return None if embedding is None else Config._get_embedding_spec(embedding)

    @property
    def dynamic_input(self) -> list[str] | dict[str, list[str] | dict[str, list[str]]]:
        return self._cfg.get("dynamic_input")

    @property
    def dynamic_input_conceptual_model(self) -> Optional[dict[str, str | list[str]]]:
        return self._cfg.get("dynamic_input_conceptual_model")

    @property
    def dynamic_parameterization_conceptual_model(self) -> list[str]:
        return Config._as_default_list(self._cfg.get("dynamic_parameterization_conceptual_model"))

    @property
    def epochs(self) -> int:
        return self._cfg.get("epochs")

    @property
    def experiment_name(self) -> str:
        # If experiment_name is not set, create a random one
        if self._cfg.get("experiment_name") is None:
            self._cfg["experiment_name"] = "experiment_" + str(random.randint(0, 10_000))
        return self._cfg.get("experiment_name")

    @property
    def forcings(self) -> list[str]:
        return Config._as_default_list(self._cfg.get("forcings"))

    @property
    def forecast_input(self) -> list[str] | dict[str, list[str]]:
        return self._cfg.get("forecast_input", [])

    @property
    def hidden_size(self) -> int:
        return self._cfg.get("hidden_size")

    @property
    def initial_forget_bias(self) -> float:
        return self._cfg.get("initial_forget_bias", None)

    @property
    def lagged_features(self) -> Optional[dict[str, int | list[int]]]:
        return self._cfg.get("lagged_features")

    @property
    def learning_rate(self) -> float | dict[str, float]:
        return self._cfg.get("learning_rate", 0.001)

    @property
    def max_updates_per_epoch(self) -> int:
        return self._cfg.get("max_updates_per_epoch")

    @property
    def model(self) -> str:
        return self._cfg.get("model")

    @property
    def nan_handling_method(self) -> Optional[str]:
        return self._cfg.get("nan_handling_method")

    @property
    def nan_probability(self) -> Optional[dict[str, dict[str, float]]]:
        return self._cfg.get("nan_probability")

    @property
    def nan_probabilistic_masking(self) -> bool:
        return self._cfg.get("nan_probabilistic_masking", False)

    @nan_probabilistic_masking.setter
    def nan_probabilistic_masking(self, value: bool) -> None:
        self._cfg["nan_probabilistic_masking"] = value

    @property
    def num_mixture_components(self) -> int:
        return self._cfg.get("num_mixture_components")

    @property
    def num_conceptual_models(self) -> int:
        return self._cfg.get("num_conceptual_models", 1)

    @property
    def num_workers(self) -> int:
        return self._cfg.get("num_workers", 0)

    @property
    def path_data(self) -> Path:
        path = self._cfg.get("path_data")
        return Path(path) if path else None

    @property
    def path_additional_features(self) -> Optional[Path]:
        path = self._cfg.get("path_additional_features")
        return Path(path) if path else None

    @property
    def path_entities(self) -> Optional[Path]:
        path = self._cfg.get("path_entities")
        return Path(path) if path else None

    @property
    def path_entities_testing(self) -> Optional[Path]:
        path = self._cfg.get("path_entities_testing")
        return Path(path) if path else self.path_entities

    @property
    def path_entities_training(self) -> Optional[Path]:
        path = self._cfg.get("path_entities_training")
        return Path(path) if path else self.path_entities

    @property
    def path_entities_validation(self) -> Optional[Path]:
        path = self._cfg.get("path_entities_validation")
        return Path(path) if path else self.path_entities

    @property
    def path_save_folder(self) -> Path:
        path = self._cfg.get("path_save_folder")
        if path:
            return Path(f"{path}/{self.experiment_name}_seed_{self.random_seed}")
        else:
            return Path(f"../results/{self.experiment_name}_seed_{self.random_seed}")

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
        if self._cfg.get("random_seed") is None:
            self._cfg["random_seed"] = int(np.random.uniform(0, 1e6))
        return self._cfg.get("random_seed")

    @random_seed.setter
    def random_seed(self, value: int):
        self._cfg["random_seed"] = value

    @property
    def routing_model(self) -> Optional[str]:
        return self._cfg.get("routing_model")

    @property
    def static_embedding(self) -> Optional[dict[str, str | float | list[int]]]:
        embedding = self._cfg.get("static_embedding")
        return None if embedding is None else Config._get_embedding_spec(embedding)

    @property
    def static_input(self) -> list[str]:
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
        return self._cfg.get("steplr_step_size")

    @property
    def steplr_gamma(self) -> Optional[float]:
        return self._cfg.get("steplr_gamma")

    @property
    def target(self) -> list[str]:
        return Config._as_default_list(self._cfg.get("target"))

    @property
    def teacher_forcing_scheduler(self) -> Optional[dict[str, float]]:
        return self._cfg.get("teacher_forcing_scheduler")

    @property
    def testing_period(self) -> list[str]:
        return self._cfg.get("testing_period")

    @property
    def training_period(self) -> list[str]:
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
    def validation_period(self) -> list[str]:
        return self._cfg.get("validation_period")
