from hy2dl.evaluation.basetester import BaseTester
from hy2dl.utils.config import Config


def get_tester(cfg: Config) -> BaseTester:
    """Get data set instance, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """
    if cfg.forecast_signals == []:
        if cfg.head.lower() == "mdn":
            from hy2dl.evaluation.simulation_tester_mdn import SimulationTesterMDN

            evaluator = SimulationTesterMDN
        elif cfg.model.lower() == "hybrid":
            from hy2dl.evaluation.hybridmodel_tester import HybridModelTester

            evaluator = HybridModelTester
        else:
            from hy2dl.evaluation.simulation_tester import SimulationTester

            evaluator = SimulationTester
    else:
        if cfg.head.lower() == "mdn":
            from hy2dl.evaluation.forecast_tester_mdn import ForecastTesterMDN

            evaluator = ForecastTesterMDN
        else:
            from hy2dl.evaluation.forecast_tester import ForecastTester

            evaluator = ForecastTester

    return evaluator
