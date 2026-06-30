import hy2dl.utils.distributions as distribution_module


def get_distribution_registry():
    return {
        "gaussian": distribution_module.GaussianMixture,
        "laplacian": distribution_module.AsymmetricLaplaceMixture,
    }


def get_distribution(distribution: str) -> distribution_module.BaseDistribution:
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
    registry = get_distribution_registry()
    dist_name = distribution.lower()

    if dist_name not in registry:
        available = list(registry.keys())
        raise NotImplementedError(f"'{dist_name}' not implemented. Available distributions: {available}")

    # Instantiate the mapped class and return it
    dist_class = registry[dist_name]
    return dist_class()
