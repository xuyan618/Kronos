"""Sector rotation analysis over normalized observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import pandas as pd

from app.data.schema import OBSERVATION_COLUMNS, filter_as_of

DEFAULT_SECTOR_MAPPINGS: dict[str, Sequence[str]] = {
    "technology": ("technology", "tech", "it", "xlk", "semis", "software"),
    "financials": ("financials", "fin", "banks", "xlf", "financial"),
    "healthcare": ("healthcare", "health", "xlv", "med"),
    "consumer_discretionary": ("consumer_discretionary", "discretionary", "xly", "consumer"),
    "industrials": ("industrials", "xli", "industrial"),
    "energy": ("energy", "xle", "oil"),
    "utilities": ("utilities", "xlu", "utility"),
    "materials": ("materials", "xlb", "materials"),
    "communication_services": ("communication_services", "communication", "xlc", "telecom"),
    "real_estate": ("real_estate", "reit", "xlre", "realty"),
    "consumer_staples": ("consumer_staples", "staples", "xlp", "consumer staples"),
}

DEFAULT_BENCHMARK_ALIASES: Sequence[str] = ("benchmark", "sp500", "spy", "market", "equity_benchmark")


@dataclass(frozen=True)
class SectorRotationConfig:
    """Configuration for sector rotation scoring and ranking."""

    benchmark_aliases: Sequence[str] = DEFAULT_BENCHMARK_ALIASES
    sector_mappings: Mapping[str, Sequence[str]] = field(default_factory=lambda: dict(DEFAULT_SECTOR_MAPPINGS))
    min_observations: int = 3
    volatility_floor: float = 1e-6
    relative_strength_threshold: float = 0.02
    minimum_coverage: float = 0.5
    max_staleness_days: int = 30

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")


def _validate_frame(frame: pd.DataFrame) -> None:
    if set(frame.columns) != set(OBSERVATION_COLUMNS):
        raise ValueError("frame does not match the normalized observation contract")


def _normalize_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _as_of_string(value: object) -> str | None:
    normalized = _normalize_timestamp(value)
    if normalized is None:
        return None
    return normalized.isoformat().replace("+00:00", "Z")


def _match_series(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.DataFrame:
    names = {str(alias).lower() for alias in aliases}
    return frame.loc[
        frame["field"].astype(str).str.lower().isin(names)
        | frame["series_id"].astype(str).str.lower().isin(names)
    ].copy()


def _aggregate_series(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["observation_date", "value", "source", "available_at", "series_id"])
    result = frame.sort_values(["observation_date", "available_at", "revision_id"]).copy()
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["value"]).drop_duplicates(subset=["observation_date"], keep="last")
    return result.sort_values("observation_date").reset_index(drop=True)


def _compute_metrics(series: pd.Series, benchmark: pd.Series, *, config: SectorRotationConfig) -> dict[str, float | None]:
    if len(series) < config.min_observations or len(benchmark) < config.min_observations:
        return {
            "relative_strength": None,
            "momentum": None,
            "trend": None,
            "volatility": None,
            "risk_adjusted_momentum": None,
        }

    series_value = pd.to_numeric(series, errors="coerce").dropna()
    benchmark_value = pd.to_numeric(benchmark, errors="coerce").dropna()
    if series_value.empty or benchmark_value.empty:
        return {
            "relative_strength": None,
            "momentum": None,
            "trend": None,
            "volatility": None,
            "risk_adjusted_momentum": None,
        }

    series_returns = series_value.pct_change().dropna()
    benchmark_returns = benchmark_value.pct_change().dropna()
    if series_returns.empty or benchmark_returns.empty:
        return {
            "relative_strength": None,
            "momentum": None,
            "trend": None,
            "volatility": None,
            "risk_adjusted_momentum": None,
        }

    recent = min(config.min_observations, len(series_returns))
    momentum = float(series_returns.tail(recent).mean())
    if len(series_value) >= 2:
        trend = float((series_value.iloc[-1] - series_value.iloc[0]) / max(abs(float(series_value.iloc[0])), 1e-6))
    else:
        trend = 0.0
    volatility = float(series_returns.tail(recent).std(ddof=0))
    volatility = max(volatility, config.volatility_floor)
    risk_adjusted = momentum / volatility if volatility > 0 else None
    relative_strength = float((series_value.iloc[-1] / series_value.iloc[0]) - (benchmark_value.iloc[-1] / benchmark_value.iloc[0]))
    return {
        "relative_strength": relative_strength,
        "momentum": momentum,
        "trend": trend,
        "volatility": volatility,
        "risk_adjusted_momentum": risk_adjusted,
    }


def _label_for(relative_strength: float | None, momentum: float | None, trend: float | None, *, threshold: float) -> str:
    if relative_strength is None or momentum is None or trend is None:
        return "UNKNOWN"
    if relative_strength > threshold and momentum > 0 and trend >= 0:
        return "LEADING"
    if relative_strength < -threshold and momentum < 0 and trend <= 0:
        return "LAGGING"
    return "NEUTRAL"


def _rank_active_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    active = frame.loc[frame["status"] == "ACTIVE"].copy()
    if active.empty:
        frame["rank"] = None
        return frame
    active_sorted = active.sort_values(["sector_score", "sector"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    active_sorted["rank"] = None
    score_to_rank: dict[float, int] = {}
    current_score: float | None = None
    current_rank = 0
    for idx, row in active_sorted.iterrows():
        score = row["sector_score"]
        if pd.isna(score):
            active_sorted.at[idx, "rank"] = None
            continue
        score_float = float(score)
        if current_score is None or abs(score_float - current_score) > 1e-9:
            current_rank += 1
            current_score = score_float
        active_sorted.at[idx, "rank"] = current_rank
        score_to_rank[score_float] = current_rank
    result = pd.concat([active_sorted, frame.loc[frame["status"] != "ACTIVE"]], ignore_index=True)
    result = result.sort_values(["status", "sector_score"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    return result


def calculate_sector_rotation(
    frame: pd.DataFrame,
    *,
    as_of: object | None = None,
    target_date: object | None = None,
    benchmark: str | Sequence[str] | None = None,
    sectors: Mapping[str, Sequence[str]] | None = None,
    config: SectorRotationConfig | None = None,
) -> pd.DataFrame:
    """Return one sector-rotation snapshot using only releases visible at ``as_of``."""
    _validate_frame(frame)
    cfg = config or SectorRotationConfig()
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    if visible.empty:
        return pd.DataFrame(columns=[
            "sector", "benchmark", "relative_strength", "momentum", "trend", "volatility",
            "risk_adjusted_momentum", "sector_score", "rank", "label", "status",
            "as_of", "limitations", "provenance", "source_series", "source_timestamps",
        ])

    target = pd.Timestamp(target_date).date() if target_date is not None else visible["observation_date"].max()
    cutoff = visible.loc[visible["observation_date"] <= target].copy()
    benchmark_aliases = tuple(benchmark) if isinstance(benchmark, (list, tuple, set)) else (benchmark,) if benchmark else cfg.benchmark_aliases
    sector_map = dict(sectors) if sectors is not None else dict(cfg.sector_mappings)
    benchmark_rows = _match_series(cutoff, benchmark_aliases)
    benchmark_frame = _aggregate_series(benchmark_rows)
    if benchmark_frame.empty:
        benchmark_name = benchmark_aliases[0] if benchmark_aliases else "benchmark"
        return pd.DataFrame([
            {
                "sector": sector_name,
                "benchmark": benchmark_name,
                "relative_strength": None,
                "momentum": None,
                "trend": None,
                "volatility": None,
                "risk_adjusted_momentum": None,
                "sector_score": None,
                "rank": None,
                "label": "UNKNOWN",
                "status": "UNKNOWN",
                "as_of": _as_of_string(as_of),
                "limitations": ["benchmark data unavailable at target date"],
                "provenance": [],
                "source_series": [],
                "source_timestamps": [],
            }
            for sector_name in sector_map
        ])

    benchmark_series = benchmark_frame.set_index("observation_date")["value"].sort_index()
    benchmark_name = benchmark_aliases[0] if benchmark_aliases else "benchmark"
    results = []
    for sector_name, aliases in sector_map.items():
        rows = _match_series(cutoff, aliases)
        if rows.empty:
            results.append({
                "sector": sector_name,
                "benchmark": benchmark_name,
                "relative_strength": None,
                "momentum": None,
                "trend": None,
                "volatility": None,
                "risk_adjusted_momentum": None,
                "sector_score": None,
                "rank": None,
                "label": "UNKNOWN",
                "status": "UNKNOWN",
                "as_of": _as_of_string(as_of),
                "limitations": ["sector series unavailable at target date"],
                "provenance": [],
                "source_series": [],
                "source_timestamps": [],
            })
            continue

        series_frame = _aggregate_series(rows)
        series = series_frame.set_index("observation_date")["value"].sort_index()
        metrics = _compute_metrics(series, benchmark_series, config=cfg)
        if metrics["momentum"] is None or metrics["relative_strength"] is None or metrics["trend"] is None:
            limitations = ["insufficient historical observations"]
            label = "UNKNOWN"
            status = "UNKNOWN"
            score = None
        else:
            volatility = float(metrics["volatility"])
            relative_strength = float(metrics["relative_strength"])
            momentum = float(metrics["momentum"])
            trend = float(metrics["trend"])
            score = 0.45 * relative_strength + 0.35 * momentum + 0.2 * trend
            label = _label_for(relative_strength, momentum, trend, threshold=cfg.relative_strength_threshold)
            status = "ACTIVE" if abs(score) >= cfg.relative_strength_threshold or label != "UNKNOWN" else "UNKNOWN"
            limitations = [] if status == "ACTIVE" else ["insufficient strength or coverage"]

        results.append({
            "sector": sector_name,
            "benchmark": benchmark_name,
            "relative_strength": None if metrics["relative_strength"] is None else float(metrics["relative_strength"]),
            "momentum": None if metrics["momentum"] is None else float(metrics["momentum"]),
            "trend": None if metrics["trend"] is None else float(metrics["trend"]),
            "volatility": None if metrics["volatility"] is None else float(metrics["volatility"]),
            "risk_adjusted_momentum": None if metrics["risk_adjusted_momentum"] is None else float(metrics["risk_adjusted_momentum"]),
            "sector_score": None if score is None else float(score),
            "rank": None,
            "label": label,
            "status": status,
            "as_of": _as_of_string(as_of),
            "limitations": limitations,
            "provenance": [
                {"series_id": row["series_id"], "field": row["field"], "available_at": row["available_at"]}
                for _, row in rows.drop_duplicates(subset=["series_id", "field"]).iterrows()
            ],
            "source_series": list(dict.fromkeys(rows["series_id"].astype(str).tolist())),
            "source_timestamps": sorted(rows["available_at"].astype(str).unique().tolist()),
        })

    output = pd.DataFrame(results)
    output = _rank_active_rows(output) if not output.empty else output
    return output


def calculate_sector_history(
    frame: pd.DataFrame,
    *,
    benchmark: str | Sequence[str] | None = None,
    sectors: Mapping[str, Sequence[str]] | None = None,
    as_of: object | None = None,
    config: SectorRotationConfig | None = None,
) -> pd.DataFrame:
    """Return chronological sector snapshots for every visible observation date."""
    _validate_frame(frame)
    visible = filter_as_of(frame, as_of) if as_of is not None else frame.copy()
    if visible.empty:
        return pd.DataFrame()
    dates = sorted(visible["observation_date"].dropna().unique())
    if not dates:
        return pd.DataFrame()
    frames = [
        calculate_sector_rotation(visible, as_of=as_of, target_date=value, benchmark=benchmark, sectors=sectors, config=config).assign(evaluation_date=value)
        for value in dates
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_sector_rotation_report(rotation: pd.DataFrame, *, as_of: object | None = None) -> dict[str, object]:
    """Turn a rotation snapshot into a readable specialist report."""
    if rotation.empty:
        return {
            "as_of": _as_of_string(as_of),
            "benchmark": "UNKNOWN",
            "leaders": [],
            "laggards": [],
            "ranking": [],
            "limitations": ["no sector observations available"],
            "specialist_report": "No sector rotation data available.",
        }

    leaders = rotation.loc[rotation["label"] == "LEADING"].sort_values("sector_score", ascending=False)
    laggards = rotation.loc[rotation["label"] == "LAGGING"].sort_values("sector_score", ascending=True)
    ranking = [
        {"sector": row["sector"], "rank": row.get("rank"), "score": row.get("sector_score"), "label": row.get("label")}
        for _, row in rotation.sort_values(["status", "sector_score"], ascending=[False, False], kind="mergesort").iterrows()
    ]
    limitations = []
    for values in rotation["limitations"].fillna([]).tolist():
        if isinstance(values, list):
            limitations.extend(values)
        elif values:
            limitations.append(str(values))
    if not limitations:
        limitations = ["no explicit limitations"]

    base_as_of = as_of if as_of is not None else rotation["as_of"].dropna().iloc[0] if not rotation["as_of"].dropna().empty else None
    report = (
        f"As of {base_as_of if base_as_of is not None else 'unknown'}, "
        f"leaders are {', '.join(leaders['sector'].astype(str).tolist()) if not leaders.empty else 'none'}; "
        f"laggards are {', '.join(laggards['sector'].astype(str).tolist()) if not laggards.empty else 'none'}."
    )
    return {
        "as_of": _as_of_string(base_as_of),
        "benchmark": rotation["benchmark"].dropna().iloc[0] if not rotation["benchmark"].dropna().empty else "UNKNOWN",
        "leaders": leaders["sector"].astype(str).tolist(),
        "laggards": laggards["sector"].astype(str).tolist(),
        "ranking": ranking,
        "limitations": limitations,
        "specialist_report": report,
    }


def generate_sector_rotation_report(rotation: pd.DataFrame, *, as_of: object | None = None) -> dict[str, object]:
    """Backward-compatible alias for specialist sector reports."""
    return build_sector_rotation_report(rotation, as_of=as_of)


__all__ = [
    "DEFAULT_BENCHMARK_ALIASES",
    "DEFAULT_SECTOR_MAPPINGS",
    "SectorRotationConfig",
    "build_sector_rotation_report",
    "calculate_sector_history",
    "calculate_sector_rotation",
    "generate_sector_rotation_report",
]
