from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.datasetzoo.camelsch import CAMELS_CH
from hy2dl.datasetzoo.camelsde import CAMELS_DE
from hy2dl.datasetzoo.camelsgb import CAMELS_GB
from hy2dl.datasetzoo.camelsus import CAMELS_US
from hy2dl.datasetzoo.caravan import CARAVAN
from hy2dl.datasetzoo.hourlycamelsde import Hourly_CAMELS_DE
from hy2dl.datasetzoo.hourlycamelsus import Hourly_CAMELS_US
from hy2dl.utils.config import Config

# Define the registry mapping
dataset_registry = {
    "camels_us": CAMELS_US,
    "camels_gb": CAMELS_GB,
    "camels_de": CAMELS_DE,
    "camels_ch": CAMELS_CH,
    "caravan": CARAVAN,
    "hourly_camels_us": Hourly_CAMELS_US,
    "hourly_camels_de": Hourly_CAMELS_DE,
}


def get_dataset(cfg: Config) -> type[BaseDataset]:
    """Get data set class, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    Returns
    -------
    BaseDataset
        The dataset class corresponding to the configuration.
    """
    dataset_name = cfg.dataset.lower()

    if dataset_name not in dataset_registry:
        available = list(dataset_registry.keys())
        raise NotImplementedError(f"No dataset class implemented for '{dataset_name}'. Available datasets: {available}")

    # Return the uninstantiated dataset class
    return dataset_registry[dataset_name]
