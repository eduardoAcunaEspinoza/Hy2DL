import hy2dl.training.loss as loss
from hy2dl.utils.config import Config


def get_loss(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance of the type specified in the config.
    """

    if cfg.loss.lower() == "nse_basin_averaged":
        loss = loss.NSEBasinAveraged(cfg=cfg)
    elif cfg.loss.lower() == "weighted_mse":
        loss = loss.WeightedMSE(cfg=cfg)
    elif cfg.loss.lower() == "feng2022":
        loss = loss.Feng2022(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss()`")

    return loss
