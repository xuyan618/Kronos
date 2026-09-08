"""Deterministic reconciliation of Kronos-Macro specialist reports.

This module deliberately consumes specialist outputs; it does not recalculate
macro, cross-asset, sector, technical, Kronos, or risk metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, Iterable, Mapping

SPECIALISTS = ("macro", "cross_asset", "sectors", "technical", "kronos", "risk_decision")
_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL", "UNKNOWN"}
_EVIDENCE = {"KNOWN", "UNKNOWN"}


def _timestamp(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError(f"{field_name} must be an ISO timestamp")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SpecialistReport:
    """The minimum provenance and directional contract for one specialist."""

    specialist: str
    direction: str
    signal: float | None
    as_of: datetime | str
    report_timestamp: datetime | str
    provenance: str
    evidence: str = "KNOWN"
    summary: str = ""
    risks: tuple[str, ...] = ()
    critical: bool = False
    risk_valid: bool | None = None
    expected_value: float | None = None

    def __post_init__(self) -> None:
        name = self.specialist.strip().lower()
        direction = str(self.direction or "UNKNOWN").upper()
        evidence = str(self.evidence or "UNKNOWN").upper()
        if not name:
            raise ValueError("specialist is required")
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}")
        if evidence not in _EVIDENCE:
            raise ValueError("evidence must be KNOWN or UNKNOWN")
        as_of = _timestamp(self.as_of, "as_of")
        report_timestamp = _timestamp(self.report_timestamp, "report_timestamp")
        if not self.provenance.strip():
            raise ValueError("provenance is required")
        if self.signal is not None and (not isfinite(self.signal) or not -1 <= self.signal <= 1):
            raise ValueError("signal must be finite and between -1 and 1")
        if self.expected_value is not None and not isfinite(self.expected_value):
            raise ValueError("expected_value must be finite")
        object.__setattr__(self, "specialist", name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "report_timestamp", report_timestamp)
        object.__setattr__(self, "risks", tuple(self.risks))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["as_of"] = _iso(self.as_of)
        result["report_timestamp"] = _iso(self.report_timestamp)
        return result


@dataclass(frozen=True)
class ReconcileAction:
    """A request for a downstream specialist or data provider."""

    action: str
    specialist: str
    reason: str
    priority: str = "HIGH"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ChiefInvestmentReport:
    action: str
    score: float
    confidence: float
    confidence_is_win_probability: bool
    reasons: tuple[str, ...]
    risks: tuple[str, ...]
    evidence: tuple[str, ...]
    specialist_summaries: tuple[dict[str, Any], ...]
    score_contributions: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    additional_analysis: tuple[ReconcileAction, ...]
    vetoes: tuple[str, ...]
    report_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["additional_analysis"] = [
            action.to_dict() for action in self.additional_analysis
        ]
        return result

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)


def _coerce_report(report: SpecialistReport | Mapping[str, Any]) -> SpecialistReport:
    if isinstance(report, SpecialistReport):
        return report
    values = dict(report)
    if "name" in values and "specialist" not in values:
        values["specialist"] = values.pop("name")
    if "signal_state" in values and "direction" not in values:
        values["direction"] = values.pop("signal_state")
    if "report_at" in values and "report_timestamp" not in values:
        values["report_timestamp"] = values.pop("report_at")
    if "source" in values and "provenance" not in values:
        values["provenance"] = values.pop("source")
    return SpecialistReport(**values)


def reconcile(
    reports: Iterable[SpecialistReport | Mapping[str, Any]],
    *,
    as_of: datetime | str,
    weights: Mapping[str, float] | None = None,
    max_age: timedelta | float = timedelta(hours=24),
    critical_specialists: Iterable[str] = ("risk_decision",),
    report_timestamp: datetime | str | None = None,
) -> ChiefInvestmentReport:
    """Reconcile reports with deterministic scoring and hard vetoes."""
    cutoff = _timestamp(as_of, "as_of")
    age_limit = max_age.total_seconds() if isinstance(max_age, timedelta) else float(max_age)
    configured_weights = {key.lower(): float(value) for key, value in (weights or {}).items()}
    vetoes: list[str] = []
    actions: list[ReconcileAction] = []
    normalized: list[SpecialistReport] = []
    for raw in reports:
        try:
            normalized.append(_coerce_report(raw))
        except (TypeError, ValueError, KeyError) as exc:
            vetoes.append("invalid_report")
            actions.append(ReconcileAction("REPAIR_REPORT", "chief_investment", str(exc)))

    by_name = {report.specialist: report for report in normalized}
    critical = {name.lower() for name in critical_specialists}
    for name in sorted(critical - by_name.keys()):
        vetoes.append("missing_critical_data")
        actions.append(ReconcileAction("REQUEST_REPORT", name, "Critical specialist report is missing."))

    valid: list[SpecialistReport] = []
    for report in normalized:
        if report.report_timestamp > cutoff or report.as_of > cutoff:
            vetoes.append("future_report")
            actions.append(ReconcileAction("REQUEST_REPORT", report.specialist, "Report or as-of is after reconciliation cutoff."))
            continue
        if (cutoff - report.report_timestamp).total_seconds() > age_limit:
            vetoes.append("stale_report")
            actions.append(ReconcileAction("REQUEST_REPORT", report.specialist, "Report exceeds the configured freshness window."))
            continue
        if report.evidence == "UNKNOWN" or report.direction == "UNKNOWN":
            actions.append(ReconcileAction("REQUEST_ANALYSIS", report.specialist, "Directional evidence is unknown."))
        valid.append(report)

    if any(report.specialist == "risk_decision" and report.risk_valid is False for report in valid):
        vetoes.append("invalid_risk")
    if any(report.specialist == "risk_decision" and (
        report.expected_value is None or report.expected_value <= 0
    ) for report in valid):
        vetoes.append("non_positive_expected_value")

    known = [report for report in valid if report.evidence == "KNOWN" and report.direction != "UNKNOWN"]
    bullish = [report for report in known if report.direction == "BULLISH"]
    bearish = [report for report in known if report.direction == "BEARISH"]
    if bullish and bearish:
        vetoes.append("unresolved_contradiction")
        actions.append(ReconcileAction("REQUEST_ANALYSIS", "chief_investment", "Specialist directions contradict."))
    if not known:
        vetoes.append("missing_critical_data")
        actions.append(ReconcileAction("REQUEST_ANALYSIS", "chief_investment", "No known directional evidence is available."))

    contributions: list[dict[str, Any]] = []
    weighted_total = 0.0
    total_weight = 0.0
    for report in sorted(known, key=lambda item: item.specialist):
        weight = configured_weights.get(report.specialist, 1.0)
        if not isfinite(weight) or weight <= 0:
            vetoes.append("invalid_component_weight")
            continue
        signal = report.signal
        if signal is None:
            signal = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}[report.direction]
        weighted_total += signal * weight
        total_weight += weight
        contributions.append({"specialist": report.specialist, "weight": weight, "signal": signal, "contribution": signal * weight})
    score = max(-1.0, min(1.0, weighted_total / total_weight)) if total_weight else 0.0
    agreement = 1.0 if not (bullish and bearish) else 0.0
    coverage = len(known) / max(1, len(normalized))
    confidence = max(0.0, min(1.0, abs(score) * agreement * coverage))
    action = "BUY" if score > 0 and not vetoes else "SELL" if score < 0 and not vetoes else "NO_TRADE"
    reasons = [f"Weighted directional score is {score:.4f}."]
    reasons.extend(f"Veto: {veto}" for veto in sorted(set(vetoes)))
    risks = [risk for report in valid for risk in report.risks]
    evidence = [f"{report.specialist}: {report.direction} ({report.evidence})" for report in sorted(valid, key=lambda item: item.specialist)]
    summaries = tuple(report.to_dict() for report in sorted(normalized, key=lambda item: item.specialist))
    limitations = (
        "Confidence measures directional evidence agreement and coverage; it is not a win probability.",
        "The chief agent does not recalculate specialist indicators, risk, or expected value.",
    )
    timestamp = _timestamp(report_timestamp, "report_timestamp") if report_timestamp is not None else cutoff
    return ChiefInvestmentReport(
        action=action,
        score=score,
        confidence=confidence,
        confidence_is_win_probability=False,
        reasons=tuple(reasons),
        risks=tuple(risks),
        evidence=tuple(evidence),
        specialist_summaries=summaries,
        score_contributions=tuple(contributions),
        limitations=limitations,
        additional_analysis=tuple(actions),
        vetoes=tuple(dict.fromkeys(vetoes)),
        report_timestamp=_iso(timestamp),
    )


reconcile_reports = reconcile
serialize_report = lambda report, **kwargs: report.to_json(**kwargs)

__all__ = [
    "SPECIALISTS",
    "SpecialistReport",
    "ReconcileAction",
    "ChiefInvestmentReport",
    "reconcile",
    "reconcile_reports",
    "serialize_report",
]
