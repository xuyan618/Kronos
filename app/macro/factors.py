"""Configurable macro factor calculations over normalized observations."""

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from app.data.schema import OBSERVATION_COLUMNS, filter_as_of
from .regime import classify_regime

FACTOR_ALIASES = {
    "growth": ("growth", "gdp_growth", "industrial_production"),
    "inflation": ("inflation", "cpi", "pce"),
    "liquidity": ("liquidity", "money_supply"),
    "rates": ("rates", "policy_rate", "fed_funds"),
    "yield_curve": ("yield_curve", "term_spread", "10y2y"),
    "financial_conditions": ("financial_conditions", "fci"),
    "credit": ("credit", "credit_spread"),
    "usd": ("usd", "dxy", "dollar"),
    "volatility": ("volatility", "vix"),
    "risk_appetite": ("risk_appetite", "equity_return"),
}
_DIRECTIONS = {
    "growth": 1, "inflation": -1, "liquidity": 1, "rates": -1,
    "yield_curve": 1, "financial_conditions": -1, "credit": -1,
    "usd": -1, "volatility": -1, "risk_appetite": 1,
}


@dataclass(frozen=True)
class FactorConfig:
    lookbacks: Mapping[str, int] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)

    def lookback(self, name): return max(1, int(self.lookbacks.get(name, 1)))
    def threshold(self, name): return max(0.0, float(self.thresholds.get(name, 0.0)))
    def weight(self, name): return max(0.0, float(self.weights.get(name, 1.0)))


def _validate(frame):
    if set(frame.columns) != set(OBSERVATION_COLUMNS):
        raise ValueError("frame does not match the normalized observation contract")


def _calculate_one(visible, target_date, config):
    target = pd.Timestamp(target_date).date()
    result = {"observation_date": target}
    weighted = total_weight = 0.0
    for name, aliases in FACTOR_ALIASES.items():
        mask = (
            (visible["observation_date"] <= target)
            & (visible["field"].astype(str).str.lower().isin(aliases)
               | visible["series_id"].astype(str).str.lower().isin(aliases))
        )
        values = pd.to_numeric(
            visible.loc[mask].sort_values(["observation_date", "available_at"])["value"],
            errors="coerce",
        ).dropna()
        lookback = config.lookback(name)
        if len(values) <= lookback:
            score, confidence, state = None, 0.0, "UNKNOWN"
        else:
            scale = max(abs(float(values.iloc[-1])), abs(float(values.iloc[-1 - lookback])), 1.0)
            score = max(-2.0, min(2.0, (float(values.iloc[-1] - values.iloc[-1 - lookback])
                                        * _DIRECTIONS[name] / scale * 2.0)))
            confidence = min(1.0, len(values) / (lookback + 2))
            threshold = config.threshold(name)
            state = "POSITIVE" if score > threshold else "NEGATIVE" if score < -threshold else "NEUTRAL"
            weighted += score * config.weight(name)
            total_weight += config.weight(name)
        result.update({f"{name}_score": score, f"{name}_state": state,
                       f"{name}_confidence": confidence})
    result["macro_score"] = None if not total_weight else max(0.0, min(1.0, (weighted / total_weight + 2) / 4))
    result["known_factors"] = sum(result[f"{name}_state"] != "UNKNOWN" for name in FACTOR_ALIASES)
    result["confidence"] = sum(result[f"{name}_confidence"] for name in FACTOR_ALIASES) / len(FACTOR_ALIASES)
    result.update(classify_regime(result))
    return result


def calculate_macro_factors(frame, *, as_of=None, target_date=None, config=None):
    """Return one factor row using only releases visible at ``as_of``."""
    _validate(frame)
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    if visible.empty:
        return pd.DataFrame()
    target = target_date if target_date is not None else visible["observation_date"].max()
    row = _calculate_one(visible, target, config or FactorConfig())
    row["as_of"] = None if as_of is None else pd.Timestamp(as_of).isoformat()
    return pd.DataFrame([row])


def calculate_macro_history(frame, *, as_of=None, config=None):
    """Return chronological factor rows for every visible observation date."""
    _validate(frame)
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    dates = sorted(visible["observation_date"].dropna().unique())
    return pd.concat(
        [calculate_macro_factors(visible, target_date=date, config=config) for date in dates],
        ignore_index=True,
    ) if dates else pd.DataFrame()
