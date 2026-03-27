from dataclasses import dataclass
from typing import Callable


@dataclass
class MetricMeta:
    func: Callable
    is_forecast_only: bool = False
    is_probabilistic: bool = False


class MetricRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, is_forecast_only: bool = False, is_probabilistic: bool = False):
        def decorator(func):
            self._registry[name.lower()] = MetricMeta(func, is_forecast_only, is_probabilistic)
            return func

        return decorator

    def get(self, name: str) -> MetricMeta:
        if name.lower() not in self._registry:
            raise ValueError(f"Unknown metric: {name}. Available: {list(self._registry.keys())}")
        return self._registry[name.lower()]

    def get_available(self, forecast_mode: bool, probabilistic: bool) -> list[str]:
        return [
            name
            for name, meta in self._registry.items()
            if (not meta.is_forecast_only or forecast_mode) and (not meta.is_probabilistic or probabilistic)
        ]


# Instantiate the single, global registry object.
registry = MetricRegistry()
