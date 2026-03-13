import os
import random
from itertools import chain
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from hy2dl.utils.distributions import Distribution
from hy2dl.utils.logging import get_logger


class Config(object):
    """Read run configuration from the specified path or dictionary and parse it into a configuration object.

    Parameters
    ----------
    yml_path_or_dict : Union[str, dict]
        Either a path to the config file or a dictionary of configuration values.
    base_dir : Path
        Base directory. It is use to resolve relative paths in the configuration.
    dev_mode : bool, default=False
        Whether to skip some checks for unknown keys in the configuration. This can be useful during development when
        the configuration is still changing frequently.

     This class is based on Neural Hydrology [#]_ and adapted for our specific case.

    References
    ----------
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022
    """

    def __init__(self, yml_path_or_dict: dict, base_dir: Path, dev_mode: bool = False):
        # Base directory, to resolve relative paths
        self.base_dir = base_dir

        # read the config from a dictionary
        self._cfg = self._read_yaml(yml_path_or_dict)

        # Check if the config contains any unknown keys
        if not dev_mode:
            Config._check_cfg_keys(cfg=self._cfg)

        # Multiple checks to ensure valid configuration
        self._check_dynamic_inputs()
        self._check_seq_length()
        self._check_embeddings()
        self._check_nan_settings()
        self._check_forecast()
        self._check_models()
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
        has_embedding = self.dynamic_embedding is not None

        if isinstance(self.dynamic_input, dict) and not has_embedding:
            raise ValueError("`dynamic_input` as dictionary is only supported when `dynamic_embedding` is specified")

        if self.static_input is None and self.static_embedding is not None:
            raise ValueError("`static_embedding` requires specification of `static_input`")

        if isinstance(self.dynamic_input, dict) and isinstance(self.custom_seq_processing, dict):
            if set(self.dynamic_input.keys()) != set(self.custom_seq_processing.keys()):
                raise ValueError("`dynamic_input` and `custom_seq_processing` must have the same keys.")

        if isinstance(self.nan_handling_method, str) and not has_embedding:
            raise ValueError("`dynamic_embedding` must be specified when using `nan_handling_method`")

    def _check_forecast(self):
        pfi = self.pseudo_forecast_input
        fi = self.forecast_input
        di = self.dynamic_input
        has_embedding = self.dynamic_embedding is not None

        # 1. Embedding check
        if (isinstance(pfi, dict) or isinstance(fi, dict)) and not has_embedding:
            raise ValueError(
                "`dynamic_embedding` must be specified when either "
                "`pseudo_forecast_input` or `forecast_input` are dictionaries."
            )

        # 2. Both are dictionaries
        if isinstance(pfi, dict) and isinstance(fi, dict):
            if not set(pfi).isdisjoint(fi):
                raise ValueError("Keys from `pseudo_forecast_input` and `forecast_input` must be different.")

            pfi_values = set(chain.from_iterable(pfi.values()))
            fi_values = set(chain.from_iterable(fi.values()))
            if not pfi_values.isdisjoint(fi_values):
                raise ValueError("Values from `pseudo_forecast_input` and `forecast_input` must be different.")

        # 3. Both are lists
        elif isinstance(pfi, list) and isinstance(fi, list) and pfi and fi:
            are_equal = pfi == fi
            are_disjoint = set(pfi).isdisjoint(fi)

            if not are_equal and not are_disjoint:
                raise ValueError(
                    "`pseudo_forecast_input` and `forecast_input` must be either identical "
                    "(including element order) or completely different. Partial overlap is not supported."
                )

            if are_disjoint and not has_embedding:
                raise ValueError(
                    "When `pseudo_forecast_input` and `forecast_input` are different, "
                    "`dynamic_embedding` must be specified."
                )

        # 4. Mixed Types (One Dict, One List)
        elif isinstance(pfi, dict) and isinstance(fi, list) and fi:
            pfi_values = set(chain.from_iterable(pfi.values()))
            if not pfi_values.isdisjoint(fi):
                raise ValueError("Values from `pseudo_forecast_input` and `forecast_input` must be different.")

        elif isinstance(fi, dict) and isinstance(pfi, list) and pfi:
            fi_values = set(chain.from_iterable(fi.values()))
            if not fi_values.isdisjoint(pfi):
                raise ValueError("Values from `pseudo_forecast_input` and `forecast_input` must be different.")

        # 4. Dimension checks
        if not has_embedding:
            if pfi and len(pfi) != len(di):
                raise ValueError(
                    "`dynamic_input` and `pseudo_forecast_input` have different dimensions. "
                    "This is supported only if `dynamic_embedding` is specified."
                )

            if fi and len(fi) != len(di):
                raise ValueError(
                    "`dynamic_input` and `forecast_input` have different dimensions. "
                    "This is supported only if `dynamic_embedding` is specified."
                )

        # Join forecast signals into single variable
        if not fi:
            self.forecast_signals = pfi
        elif not pfi:
            self.forecast_signals = fi
        elif isinstance(pfi, dict) and isinstance(fi, dict):
            self.forecast_signals = {**pfi, **fi}
        elif isinstance(pfi, dict) and isinstance(fi, list):
            self.forecast_signals = {**pfi, "forecast": fi}
        elif isinstance(fi, dict) and isinstance(pfi, list):
            self.forecast_signals = {**fi, "pseudo_forecast": pfi}
        elif isinstance(pfi, list) and isinstance(fi, list):
            if pfi == fi:
                self.forecast_signals = pfi
            else:
                self.forecast_signals = {"pseudo_forecast": pfi, "forecast": fi}

    def _check_models(self):
        """Check for specific configurations required by certain models."""
        # Check forecast configuration
        if self.model == "forecast_lstm" and (self.seq_length_forecast == 0 or len(self.pseudo_forecast_input) == 0):
            raise ValueError(
                "`forecast_lstm` requires `seq_length_forecast > 0` and `pseudo_forecast_input` to be specified."
            )
        if self.model == "hybrid" and (self.conceptual_model is None or self.dynamic_input_conceptual_model is None):
            raise ValueError(
                "`hybrid` model requires `conceptual_model` and `dynamic_input_conceptual_model` to be specified."
            )
        if self.model == "lstmmdn":
            if self.distribution not in [dist.value for dist in Distribution]:
                raise ValueError(f"`distribution`: {self.distribution} not supported.")
            if self.num_mixture_components is None:
                raise ValueError("`lstmmdn` model requires `num_mixture_components` to be specified.")

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
                if isinstance(self.pseudo_forecast_input, dict):
                    input_groups.extend(k for k in self.pseudo_forecast_input)
                if isinstance(self.forecast_input, dict):
                    input_groups.extend(k for k in self.forecast_input)

                if set(nan_groups) != set(input_groups):
                    raise ValueError(
                        "All groups contained in `dynamic_input`, `pseudo_forecast_input`, and `forecast_input` "
                        "must be specified in `nan_probability`"
                    )

    def _check_num_workers(self):
        """Checks if the number of workers that will be used in the dataloaders is valid."""
        num_workers = self.num_workers
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
        elif num_workers > 0 and os.cpu_count() < num_workers:
            raise RuntimeError(f"num_workers ({num_workers}) must be less than number of cores ({os.cpu_count()}).")

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

    def _prepare_path(self, k: str) -> Optional[Path]:
        """Prepare a path from the configuration, ensuring it is a Path object and is absolute.

        Parameters
        ----------
        k : str
            Key in the configuration dictionary corresponding to the path to be prepared.

        Returns
        -------
        Optional[Path]
            The absolute Path, or None if the path was not specified in the configuration.

        """
        # Get property
        path = self._cfg.get(k)

        if path is None:  # If path is None, return None
            return None

        # If path is not none, returns absolute path
        path = Path(path)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def _read_yaml(self, yml_path_or_dict: str | dict) -> dict:
        """Read the configuration from a YAML file or a dictionary."""
        if isinstance(yml_path_or_dict, (Path, str)):
            # Ensure absolute path
            yml_path = (self.base_dir / yml_path_or_dict).resolve()
            with open(yml_path, "r") as file:
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
            if torch.cuda.is_available():
                return "cuda:0"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                raise RuntimeError("GPU requested but no CUDA or MPS devices available.")

        elif device.lower() == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available.")
            return "mps"

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
                f"Invalid device specification: '{device}'. Expected 'cpu', 'gpu', 'mps' or "
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
    def dataset_in_ram(self) -> bool:
        return self._cfg.get("dataset_in_ram", True)

    @dataset_in_ram.setter
    def dataset_in_ram(self, value: bool):
        self._cfg["dataset_in_ram"] = value

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

    @experiment_name.setter
    def experiment_name(self, value: str):
        if not isinstance(value, str):
            raise ValueError("experiment_name must be a string.")
        # Set a custom experiment name, this also determines the folder where results are stored
        self._cfg["experiment_name"] = value

    @property
    def forcings(self) -> list[str]:
        return Config._as_default_list(self._cfg.get("forcings"))

    @property
    def forecast_dataset_in_ram(self) -> bool:
        return self._cfg.get("forecast_dataset_in_ram", True)

    @forecast_dataset_in_ram.setter
    def forecast_dataset_in_ram(self, value: bool):
        self._cfg["forecast_dataset_in_ram"] = value

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

    @num_workers.setter
    def num_workers(self, value: int):
        if value < 0:
            raise ValueError(f"num_workers must be non-negative, got {value}.")
        elif value > 0 and os.cpu_count() < value:
            raise RuntimeError(f"num_workers ({value}) must be less than number of cores ({os.cpu_count()}).")
        self._cfg["num_workers"] = value

    @property
    def path_additional_features(self) -> Optional[Path]:
        return self._prepare_path("path_additional_features")

    @property
    def path_data(self) -> Path:
        return self._prepare_path("path_data")

    @property
    def path_dataset_testing(self) -> Path:
        if self._cfg.get("path_dataset_testing"):
            return self._prepare_path("path_dataset_testing")
        return self.path_data / "dataset_testing.zarr"

    @path_dataset_testing.setter
    def path_dataset_testing(self, value: str):
        self._cfg["path_dataset_testing"] = value

    @property
    def path_dataset_training(self) -> Path:
        if self._cfg.get("path_dataset_training"):
            return self._prepare_path("path_dataset_training")
        return self.path_data / "dataset_training.zarr"

    @path_dataset_training.setter
    def path_dataset_training(self, value: str):
        self._cfg["path_dataset_training"] = value

    @property
    def path_dataset_validation(self) -> Path:
        if self._cfg.get("path_dataset_validation"):
            return self._prepare_path("path_dataset_validation")
        return self.path_data / "dataset_validation.zarr"

    @path_dataset_validation.setter
    def path_dataset_validation(self, value: str):
        self._cfg["path_dataset_validation"] = value

    @property
    def path_entities(self) -> Optional[Path]:
        return self._prepare_path("path_entities")

    @property
    def path_entities_testing(self) -> Optional[Path]:
        if self._cfg.get("path_entities_testing"):
            return self._prepare_path("path_entities_testing")
        return self.path_entities  # default to path_entities if not specified

    @path_entities_testing.setter
    def path_entities_testing(self, value: str):
        self._cfg["path_entities_testing"] = value

    @property
    def path_entities_training(self) -> Optional[Path]:
        if self._cfg.get("path_entities_training"):
            return self._prepare_path("path_entities_training")
        return self.path_entities  # default to path_entities if not specified

    @path_entities_training.setter
    def path_entities_training(self, value: str):
        self._cfg["path_entities_training"] = value

    @property
    def path_entities_validation(self) -> Optional[Path]:
        if self._cfg.get("path_entities_validation"):
            return self._prepare_path("path_entities_validation")
        return self.path_entities  # default to path_entities if not specified

    @path_entities_validation.setter
    def path_entities_validation(self, value: str):
        self._cfg["path_entities_validation"] = value

    @property
    def path_forecast_dataset(self) -> Optional[Path]:
        return self._prepare_path("path_forecast_dataset")

    @property
    def path_save_folder(self) -> Path:
        # experiment suffix to ensure that different runs of the same experiment do not overwrite each other
        suffix = f"{self.experiment_name}_seed_{self.random_seed}"

        if self._cfg.get("path_save_folder"):
            base = Path(self._cfg.get("path_save_folder"))
            folder = base if base.is_absolute() else (self.base_dir / base)
        else:
            folder = self.base_dir / "../results"

        return (folder / suffix).resolve()

    @property
    def predict_last_n(self) -> int:
        return self._cfg.get("predict_last_n", 1)

    @predict_last_n.setter
    def predict_last_n(self, value: int):
        self._cfg["predict_last_n"] = value

    @property
    def pseudo_forecast_input(self) -> list[str] | dict[str, list[str]]:
        return self._cfg.get("pseudo_forecast_input", [])

    @property
    def optimizer(self) -> str:
        return self._cfg.get("optimizer", "adam")

    @property
    def output_features(self) -> int:
        return self._cfg.get("output_features", 1)

    @property
    def ram_safety_factor(self) -> float:
        return self._cfg.get("ram_safety_factor", 1.5)

    @ram_safety_factor.setter
    def ram_safety_factor(self, value: float):
        self._cfg["ram_safety_factor"] = value

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
