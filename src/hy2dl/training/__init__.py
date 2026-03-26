import hy2dl.training.loss as loss_module
from hy2dl.utils.config import Config

# Define the registry mapping
loss_registry = {
    "nll": loss_module.NLL,
    "nse_basin_averaged": loss_module.NSEBasinAveraged,
    "weighted_mse": loss_module.WeightedMSE,
}


def get_loss(cfg: Config) -> loss_module.BaseLoss:
    """Get loss object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss_module.BaseLoss
        A new loss instance of the type specified in the config.
    """
    loss_name = cfg.loss.lower()

    if loss_name not in loss_registry:
        available = list(loss_registry.keys())
        raise NotImplementedError(f"'{loss_name}' not implemented. Available losses: {available}")

    # Instantiate the mapped class and return it
    loss_class = loss_registry[loss_name]
    return loss_class(cfg=cfg)
