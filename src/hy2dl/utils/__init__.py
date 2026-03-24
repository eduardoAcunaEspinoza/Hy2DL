import hy2dl.utils.distributions as distribution
from hy2dl.utils.config import Config


def get_distribution(cfg: Config) -> distribution.BaseDistribution:
    """Get distribution object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    distribution.BaseDistribution
        A new distribution instance of the type specified in the config.
    """

    if cfg.distribution.lower() == "gaussian":
        distribution = distribution.GaussianMixture(cfg=cfg)
    elif cfg.distribution.lower() == "laplacian":
        distribution = distribution.AsymmetricLaplaceMixture(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.distribution} not implemented or not linked in `get_distribution()`")

    return distribution
