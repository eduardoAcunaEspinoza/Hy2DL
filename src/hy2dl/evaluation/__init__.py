from hy2dl.evaluation.basetester import BaseTester
from hy2dl.evaluation.forecast_tester import ForecastTester
from hy2dl.evaluation.simulation_tester import SimulationTester
from hy2dl.utils.config import Config


def get_tester(cfg: Config) -> BaseTester:
    """Get data set instance, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """
    if cfg.pseudo_forecast_input == [] and cfg.forecast_input == []:
        evaluator = SimulationTester
    else:
        evaluator = ForecastTester

    return evaluator
