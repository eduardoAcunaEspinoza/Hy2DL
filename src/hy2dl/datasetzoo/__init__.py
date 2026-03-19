from hy2dl.datasetzoo.basedataset import BaseDataset
from hy2dl.datasetzoo.camelsch import CAMELS_CH
from hy2dl.datasetzoo.camelsde import CAMELS_DE
from hy2dl.datasetzoo.camelsgb import CAMELS_GB
from hy2dl.datasetzoo.camelsus import CAMELS_US
from hy2dl.datasetzoo.caravan import CARAVAN
from hy2dl.datasetzoo.hourlycamelsde import Hourly_CAMELS_DE
from hy2dl.datasetzoo.hourlycamelsus import Hourly_CAMELS_US
from hy2dl.utils.config import Config


def get_dataset(cfg: Config) -> BaseDataset:
    """Get data set instance, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """
    if cfg.dataset.lower() == "camels_us":
        Dataset = CAMELS_US
    elif cfg.dataset.lower() == "camels_gb":
        Dataset = CAMELS_GB
    elif cfg.dataset.lower() == "camels_de":
        Dataset = CAMELS_DE
    elif cfg.dataset.lower() == "camels_ch":
        Dataset = CAMELS_CH
    elif cfg.dataset.lower() == "caravan":
        Dataset = CARAVAN
    elif cfg.dataset.lower() == "hourly_camels_us":
        Dataset = Hourly_CAMELS_US
    elif cfg.dataset.lower() == "hourly_camels_de":
        Dataset = Hourly_CAMELS_DE
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

    return Dataset
