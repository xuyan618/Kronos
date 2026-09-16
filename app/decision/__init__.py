"""Transparent expected-value decisioning for Kronos-Macro."""

from dataclasses import dataclass

from app.decision.fusion import (
    FusedSignal,
    fuse_signals,
    generate_trade_signal,
    kronos_direction_from_forecast,
)
from app.risk import RiskResult
from app.schemas import Evidence

__all__ = [
    "Decision",
    "make_decision",
    "decide",
    "FusedSignal",
    "fuse_signals",
    "generate_trade_signal",
    "kronos_direction_from_forecast",
]


@dataclass(frozen=True)
class Decision:
    action: str
    net_expected_value: float
    expected_value_r: float
    score: float
    vetoes: tuple[str, ...]
    contributions: tuple[str, ...]

    @property
    def is_trade(self) -> bool:
        return self.action == "TRADE"

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.contributions


def make_decision(risk: RiskResult, evidence: Evidence) -> Decision:
    """Return TRADE only when risk and evidence are independently sufficient."""
    vetoes = list(risk.vetoes)
    contributions = list(risk.explanations)
    if evidence.sample_count < evidence.minimum_samples:
        vetoes.append("insufficient_evidence")
        contributions.append(
            f"Evidence has {evidence.sample_count} samples; {evidence.minimum_samples} are required."
        )
    win = evidence.win_probability * evidence.average_win_r
    loss = (1.0 - evidence.win_probability) * evidence.average_loss_r
    expected_value_r = win - loss
    net_expected_value = (
        risk.total_risk * expected_value_r - risk.total_cost
        if risk.total_risk > 0
        else 0.0
    )
    contributions.extend(
        (
            f"Win contribution: {evidence.win_probability:.4f} × {evidence.average_win_r:.4f}R = {win:.4f}R.",
            f"Loss contribution: {1.0 - evidence.win_probability:.4f} × {evidence.average_loss_r:.4f}R = {loss:.4f}R.",
            f"Net expected value: {expected_value_r:.4f}R × {risk.total_risk:g} risk - {risk.total_cost:g} costs = {net_expected_value:g}.",
        )
    )
    if expected_value_r <= 0:
        vetoes.append("non_positive_expected_value")
        contributions.append("Expected value is not positive after win and loss probabilities.")
    action = "TRADE" if not vetoes else "NO_TRADE"
    score = max(0.0, min(1.0, expected_value_r / max(1.0, risk.r_multiple))) if action == "TRADE" else 0.0
    contributions.append(f"Decision score is {score:.4f}; every veto forces NO_TRADE.")
    return Decision(
        action=action,
        net_expected_value=net_expected_value,
        expected_value_r=expected_value_r,
        score=score,
        vetoes=tuple(dict.fromkeys(vetoes)),
        contributions=tuple(contributions),
    )


decide = make_decision
