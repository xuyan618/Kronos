"""Release-aware macro factor and regime calculations."""

from .factors import FactorConfig, calculate_macro_factors, calculate_macro_history
from .regime import classify_regime
from .signals import Signal, SignalState, calculate_signal

__all__ = [
    "FactorConfig",
    "Signal",
    "SignalState",
    "calculate_macro_factors",
    "calculate_macro_history",
    "calculate_signal",
    "classify_regime",
]
