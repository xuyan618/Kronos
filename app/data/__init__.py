"""Data collection and normalized observation contracts for Kronos-Macro."""

from .config import DataConfig
from .schema import Observation, ObservationValidationError, filter_as_of, normalize_observations


def collect_market_data(*args, **kwargs):
    """Collect market data without importing the CLI module at package import time."""
    from .engine import collect_market_data as _collect_market_data
    return _collect_market_data(*args, **kwargs)


__all__ = [
    "DataConfig",
    "Observation",
    "ObservationValidationError",
    "collect_market_data",
    "filter_as_of",
    "normalize_observations",
]
