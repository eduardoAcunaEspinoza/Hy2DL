from hy2dl.evaluation.basetester import BaseTester
from hy2dl.evaluation.evaluator import calculate_metrics
from hy2dl.evaluation.forecast_tester import ForecastTester
from hy2dl.evaluation.forecast_tester_mdn import ForecastTesterMDN
from hy2dl.evaluation.hybridmodel_tester import HybridModelTester
from hy2dl.evaluation.simulation_tester import SimulationTester
from hy2dl.evaluation.simulation_tester_mdn import SimulationTesterMDN
from hy2dl.utils.config import Config

__all__ = ["calculate_metrics", "get_tester"]


def get_tester(cfg: Config) -> BaseTester:
    """Get data set instance, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """
    if cfg.forecast_signals == []:
        if cfg.model.lower() in ["lstmmdn", "maskedarlstmmdn"]:
            evaluator = SimulationTesterMDN
        elif cfg.model.lower() == "hybrid":
            evaluator = HybridModelTester
        else:
            evaluator = SimulationTester
    else:
        if cfg.model.lower() in ["lstmmdn", "maskedarlstmmdn"]:
            evaluator = ForecastTesterMDN
        else:
            evaluator = ForecastTester

    return evaluator
