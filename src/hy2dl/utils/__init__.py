import hy2dl.utils.distributions as distribution_module
from hy2dl.utils.config import Config

# Define the registry mapping
distribution_registry = {
    "gaussian": distribution_module.GaussianMixture,
    "laplacian": distribution_module.AsymmetricLaplaceMixture,
}


def get_distribution(cfg: Config) -> distribution_module.BaseDistribution:
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
    dist_name = cfg.distribution.lower()

    if dist_name not in distribution_registry:
        available = list(distribution_registry.keys())
        raise NotImplementedError(f"'{dist_name}' not implemented. Available distributions: {available}")

    # Instantiate the mapped class and return it
    dist_class = distribution_registry[dist_name]
    return dist_class(cfg=cfg)
