"""Small, deterministic signal primitives for normalized observations."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd

from app.data.schema import OBSERVATION_COLUMNS, filter_as_of


class SignalState(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Signal:
    name: str
    value: float | None
    state: SignalState
    confidence: float
    observations: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "state": self.state.value,
            "confidence": self.confidence,
            "observations": self.observations,
        }


def _series_rows(frame: pd.DataFrame, names: Iterable[str], target_date=None) -> pd.DataFrame:
    names = {str(name).lower() for name in names}
    result = frame.loc[
        frame["field"].astype(str).str.lower().isin(names)
        | frame["series_id"].astype(str).str.lower().isin(names)
    ].copy()
    if target_date is not None:
        result = result.loc[result["observation_date"] <= pd.Timestamp(target_date).date()]
    return result.sort_values(["observation_date", "available_at"])


def calculate_signal(
    frame: pd.DataFrame,
    names: Iterable[str],
    *,
    name: str | None = None,
    as_of=None,
    target_date=None,
    lookback: int = 1,
    threshold: float = 0.0,
    direction: float = 1.0,
) -> Signal:
    """Calculate a bounded directional signal from one normalized series.

    ``direction`` changes the economic interpretation (for example, higher
    volatility is a negative risk signal) without changing source values.
    """
    if set(frame.columns) != set(OBSERVATION_COLUMNS):
        raise ValueError("frame does not match the normalized observation contract")
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    rows = _series_rows(visible, names, target_date)
    label = name or next(iter(names), "series")
    if rows.empty:
        return Signal(label, None, SignalState.UNKNOWN, 0.0, 0)
    values = pd.to_numeric(rows["value"], errors="coerce").dropna()
    if len(values) <= lookback:
        return Signal(label, None, SignalState.UNKNOWN, 0.0, len(values))
    change = float(values.iloc[-1] - values.iloc[-1 - lookback]) * direction
    scale = max(abs(float(values.iloc[-1])), abs(float(values.iloc[-1 - lookback])), 1.0)
    value = change / scale
    if abs(value) <= threshold:
        state = SignalState.NEUTRAL
    else:
        state = SignalState.POSITIVE if value > 0 else SignalState.NEGATIVE
    confidence = min(1.0, len(values) / (lookback + 2))
    return Signal(label, value, state, confidence, len(values))
