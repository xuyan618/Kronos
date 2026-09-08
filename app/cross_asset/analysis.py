"""Explainable, point-in-time cross-asset relationship calculations.

The module deliberately uses descriptive relationships rather than forecasts.  Inputs
are normalized observations and are never filled across dates: a relationship is
unknown when its two legs cannot be aligned with enough observations.
"""

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

import pandas as pd

from app.data.schema import OBSERVATION_COLUMNS, filter_as_of


@dataclass(frozen=True)
class RelationshipDefinition:
    """Definition of a two-leg relationship and its economic interpretation."""

    name: str
    left_aliases: tuple[str, ...]
    right_aliases: tuple[str, ...]
    description: str
    direction: int = 1
    lookback: int = 20
    threshold: float = 0.01
    minimum_observations: int = 3


DEFAULT_RELATIONSHIPS = (
    RelationshipDefinition(
        "stocks_vs_bonds", ("equity", "stocks", "spy", "sp500", "qqq"),
        ("bond", "bonds", "treasury", "ief", "tlt"),
        "Equities outperforming duration is a descriptive risk-seeking signal.",
    ),
    RelationshipDefinition(
        "stocks_vs_usd", ("equity", "stocks", "spy", "sp500"),
        ("usd", "dxy", "dollar"),
        "Equity performance relative to the US dollar.",
    ),
    RelationshipDefinition(
        "gold_vs_rates", ("gold", "xau", "gld"),
        ("rates", "rate", "yield", "treasury"),
        "Gold performance relative to the rate leg; not a causal rate forecast.",
    ),
    RelationshipDefinition(
        "copper_vs_growth", ("copper", "hg", "metal"),
        ("growth", "industrial_production", "gdp"),
        "Copper performance relative to a growth proxy.",
    ),
    RelationshipDefinition(
        "oil_vs_inflation", ("oil", "crude", "wti", "brent"),
        ("inflation", "cpi", "pce"),
        "Oil performance relative to an inflation proxy.",
    ),
    RelationshipDefinition(
        "hyg_vs_lqd", ("hyg", "high_yield", "credit"),
        ("lqd", "investment_grade", "ig_credit"),
        "High-yield credit performance relative to investment-grade credit.",
    ),
    RelationshipDefinition(
        "vix_vs_equities", ("equity", "stocks", "spy", "sp500"),
        ("vix", "volatility"),
        "Equity performance relative to volatility; lower VIX is supportive.",
        direction=1,
    ),
    RelationshipDefinition(
        "dxy_vs_em", ("em", "emerging", "eem", "emerging_market"),
        ("usd", "dxy", "dollar"),
        "Emerging-market performance relative to the US dollar.",
    ),
    RelationshipDefinition(
        "usd_inr_vs_indian_assets", ("indian", "india", "nifty", "sensex", "inr_asset"),
        ("usd_inr", "usdinr", "usd/inr", "inr", "inr=x"),
        "Indian-asset performance relative to USD/INR where both are available.",
    ),
)


@dataclass(frozen=True)
class CrossAssetConfig:
    """Runtime controls; values are intentionally not hidden in calculations."""

    lookbacks: Mapping[str, int] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    minimum_observations: int | None = None

    def lookback(self, definition: RelationshipDefinition) -> int:
        return max(1, int(self.lookbacks.get(definition.name, definition.lookback)))

    def threshold(self, definition: RelationshipDefinition) -> float:
        return max(0.0, float(self.thresholds.get(definition.name, definition.threshold)))

    def minimum(self, definition: RelationshipDefinition) -> int:
        minimum = definition.minimum_observations if self.minimum_observations is None else self.minimum_observations
        return max(2, int(minimum))


@dataclass(frozen=True)
class RelationshipSignal:
    relationship: str
    state: str
    score: int | None
    metric: float | None
    observations: int
    left_series: tuple[str, ...]
    right_series: tuple[str, ...]
    input_dates: tuple[str, ...]
    as_of: str | None
    explanation: str
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CrossAssetReport:
    """Specialist output consumable by a future Chief Investment Agent."""

    as_of: str | None
    signals: tuple[RelationshipSignal, ...]
    known_relationships: int
    confidence: float
    summary: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "signals": [signal.to_dict() for signal in self.signals],
            "known_relationships": self.known_relationships,
            "confidence": self.confidence,
            "summary": self.summary,
            "limitations": list(self.limitations),
        }


def _validate(frame: pd.DataFrame) -> None:
    if set(frame.columns) != set(OBSERVATION_COLUMNS):
        raise ValueError("frame does not match the normalized observation contract")


def _matches(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.DataFrame:
    aliases = {str(alias).strip().lower() for alias in aliases}
    fields = frame["field"].astype(str).str.lower()
    series = frame["series_id"].astype(str).str.lower()
    assets = frame["asset_name"].astype(str).str.lower()
    identifier_match = series.map(lambda value: any(alias in value for alias in aliases))
    asset_match = assets.map(lambda value: any(alias in value for alias in aliases))
    return frame.loc[fields.isin(aliases) | identifier_match | asset_match]


def _leg(frame: pd.DataFrame, aliases: tuple[str, ...], target: pd.Timestamp.date,
         lookback: int) -> tuple[pd.Series, tuple[str, ...]]:
    selected = _matches(frame, aliases)
    selected = selected.loc[selected["observation_date"] <= target].copy()
    if selected.empty:
        return pd.Series(dtype=float), ()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected.dropna(subset=["value"])
    # Mean only multiple representations on the same date; never fill missing dates.
    values = selected.groupby("observation_date")["value"].mean().sort_index().tail(lookback + 1)
    series = tuple(sorted(selected["series_id"].astype(str).unique()))
    return values, series


def _unknown(definition: RelationshipDefinition, as_of: str | None, reason: str,
             left_series: tuple[str, ...] = (), right_series: tuple[str, ...] = (),
             observations: int = 0, dates: tuple[str, ...] = ()) -> RelationshipSignal:
    return RelationshipSignal(
        definition.name, "UNKNOWN", None, None, observations, left_series, right_series,
        dates, as_of, f"{definition.description} UNKNOWN: {reason}", (reason,),
    )


def _calculate_one(visible: pd.DataFrame, target: object, definition: RelationshipDefinition,
                   config: CrossAssetConfig, as_of: str | None) -> RelationshipSignal:
    target_date = pd.Timestamp(target).date()
    lookback = config.lookback(definition)
    left, left_series = _leg(visible, definition.left_aliases, target_date, lookback)
    right, right_series = _leg(visible, definition.right_aliases, target_date, lookback)
    aligned = pd.concat({"left": left, "right": right}, axis=1).dropna()
    minimum = config.minimum(definition)
    dates = tuple(index.isoformat() for index in aligned.index)
    if len(aligned) < minimum + 1:
        return _unknown(
            definition, as_of, f"only {len(aligned)} aligned observations; {minimum + 1} required",
            left_series, right_series, len(aligned), dates,
        )
    returns = aligned.pct_change().dropna()
    if returns.empty:
        return _unknown(definition, as_of, "returns could not be calculated",
                        left_series, right_series, len(aligned), dates)
    relative = float((returns["left"] - returns["right"]).tail(lookback).mean())
    metric = relative * definition.direction
    threshold = config.threshold(definition)
    if metric > threshold:
        state, score = "POSITIVE", 2 if metric > threshold * 2 else 1
    elif metric < -threshold:
        state, score = "NEGATIVE", -2 if metric < -threshold * 2 else -1
    else:
        state, score = "NEUTRAL", 0
    explanation = (
        f"{definition.description} Aligned mean relative return over {len(returns)} intervals "
        f"was {metric:.4f}; threshold is {threshold:.4f}."
    )
    return RelationshipSignal(
        definition.name, state, score, metric, len(aligned), left_series, right_series,
        dates, as_of, explanation,
    )


def analyze_cross_asset(
    frame: pd.DataFrame,
    *,
    as_of: object | None = None,
    target_date: object | None = None,
    config: CrossAssetConfig | None = None,
    relationships: Iterable[RelationshipDefinition] = DEFAULT_RELATIONSHIPS,
) -> CrossAssetReport:
    """Analyze configured relationships using only observations available at ``as_of``."""
    _validate(frame)
    config = config or CrossAssetConfig()
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    if visible.empty:
        signals = tuple(
            _unknown(definition, None if as_of is None else pd.Timestamp(as_of).isoformat(), "no visible observations")
            for definition in relationships
        )
    else:
        target = target_date if target_date is not None else visible["observation_date"].max()
        as_of_text = None if as_of is None else pd.Timestamp(as_of).isoformat()
        signals = tuple(_calculate_one(visible, target, definition, config, as_of_text)
                        for definition in relationships)
    known = sum(signal.state != "UNKNOWN" for signal in signals)
    confidence = known / len(signals) if signals else 0.0
    positive = sum(signal.state == "POSITIVE" for signal in signals)
    negative = sum(signal.state == "NEGATIVE" for signal in signals)
    summary = "No sufficiently aligned relationships are known."
    if known:
        summary = f"{known} relationships known: {positive} positive, {negative} negative, {known - positive - negative} neutral."
    limitations = (
        "Relationships are descriptive and not validated forecasts.",
        "Exact-date alignment is used; missing dates are not forward-filled.",
        "Provider availability_method and asset coverage determine historical usability.",
    )
    return CrossAssetReport(as_of=None if as_of is None else pd.Timestamp(as_of).isoformat(),
                            signals=signals, known_relationships=known,
                            confidence=confidence, summary=summary, limitations=limitations)


def calculate_cross_asset_history(
    frame: pd.DataFrame, *, as_of: object | None = None,
    config: CrossAssetConfig | None = None,
    relationships: Iterable[RelationshipDefinition] = DEFAULT_RELATIONSHIPS,
) -> pd.DataFrame:
    """Return one specialist report row per visible observation date."""
    _validate(frame)
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    dates = sorted(visible["observation_date"].dropna().unique())
    rows = []
    for target in dates:
        report = analyze_cross_asset(visible, as_of=as_of, target_date=target,
                                     config=config, relationships=relationships)
        rows.append({
            "observation_date": target,
            "as_of": report.as_of,
            "known_relationships": report.known_relationships,
            "confidence": report.confidence,
            "summary": report.summary,
            "signals": [signal.to_dict() for signal in report.signals],
            "limitations": list(report.limitations),
        })
    return pd.DataFrame(rows)
