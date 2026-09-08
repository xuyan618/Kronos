"""Deterministic trade risk, sizing, and validation calculations."""

from dataclasses import dataclass
from math import floor, isfinite

from app.schemas import AccountRisk, InstrumentMetadata, TradeSpec


@dataclass(frozen=True)
class RiskResult:
    valid: bool
    quantity: float
    risk_per_contract: float
    total_risk: float
    gross_reward: float
    total_cost: float
    r_multiple: float
    vetoes: tuple[str, ...]
    explanations: tuple[str, ...]

    @property
    def position_size(self) -> float:
        return self.quantity

    @property
    def risk_amount(self) -> float:
        return self.total_risk


def assess_risk(
    instrument: InstrumentMetadata,
    account: AccountRisk,
    trade: TradeSpec,
) -> RiskResult:
    """Validate a trade and size it to the lesser of budget and portfolio capacity."""
    vetoes: list[str] = []
    explanations: list[str] = []
    direction = trade.direction.lower()
    stop_distance = (trade.entry - trade.stop) if direction == "long" else (trade.stop - trade.entry)
    reward_distance = (trade.target - trade.entry) if direction == "long" else (trade.entry - trade.target)
    if stop_distance <= 0:
        vetoes.append("stop_is_on_wrong_side")
        explanations.append("The stop must be below entry for long trades and above entry for short trades.")
    if reward_distance <= 0:
        vetoes.append("target_is_on_wrong_side")
        explanations.append("The target must be above entry for long trades and below entry for short trades.")
    if trade.slippage_per_side > instrument.tick_size * 20:
        vetoes.append("slippage_exceeds_tick_tolerance")
        explanations.append("Declared slippage is greater than twenty instrument ticks.")

    per_contract_move_risk = stop_distance * instrument.contract_multiplier
    gross_reward = reward_distance * instrument.contract_multiplier
    total_cost = (
        trade.fees_per_contract + 2.0 * trade.slippage_per_side * instrument.contract_multiplier
    )
    risk_per_contract = per_contract_move_risk + total_cost
    if risk_per_contract <= 0 or not isfinite(risk_per_contract):
        vetoes.append("non_positive_risk")
        explanations.append("Stop loss risk including fees and slippage must be positive and finite.")

    capacity = min(account.risk_budget, account.max_portfolio_risk - account.open_risk)
    if capacity <= 0:
        vetoes.append("portfolio_risk_limit_reached")
        explanations.append("No portfolio risk capacity remains after existing open risk.")
    quantity = 0.0
    if not vetoes or "non_positive_risk" not in vetoes:
        quantity = floor(capacity / risk_per_contract / instrument.lot_size) * instrument.lot_size
    if quantity < instrument.lot_size:
        vetoes.append("insufficient_risk_capacity_for_one_lot")
        explanations.append("The available risk capacity cannot fund one minimum lot.")
        quantity = 0.0

    total_risk = quantity * risk_per_contract
    reward = quantity * gross_reward
    r_multiple = (gross_reward / risk_per_contract) if risk_per_contract > 0 else 0.0
    if r_multiple <= 0:
        vetoes.append("non_positive_r_multiple")
        explanations.append("The target does not offer positive reward relative to total risk.")
    if not vetoes:
        explanations.extend(
            (
                f"Quantity is {quantity:g}, sized from {capacity:g} available risk capacity.",
                f"Risk is {total_risk:g} including {total_cost * quantity:g} round-trip costs.",
                f"Gross reward-to-risk is {r_multiple:.4f}R.",
            )
        )
    return RiskResult(
        valid=not vetoes,
        quantity=quantity,
        risk_per_contract=risk_per_contract,
        total_risk=total_risk,
        gross_reward=reward,
        total_cost=total_cost * quantity,
        r_multiple=r_multiple,
        vetoes=tuple(dict.fromkeys(vetoes)),
        explanations=tuple(explanations),
    )


calculate_position_size = assess_risk
