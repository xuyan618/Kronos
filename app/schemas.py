"""Shared, validated schemas used by the risk and decision components."""

from dataclasses import dataclass
from math import isfinite


def _positive(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _non_negative(value: float, name: str) -> float:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


@dataclass(frozen=True)
class InstrumentMetadata:
    symbol: str
    asset_class: str
    quote_currency: str
    contract_multiplier: float
    tick_size: float
    lot_size: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.asset_class.strip() or not self.quote_currency.strip():
            raise ValueError("symbol, asset_class, and quote_currency are required")
        _positive(self.contract_multiplier, "contract_multiplier")
        _positive(self.tick_size, "tick_size")
        _positive(self.lot_size, "lot_size")


@dataclass(frozen=True)
class AccountRisk:
    equity: float
    risk_budget: float
    max_portfolio_risk: float
    open_risk: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.equity, "equity")
        _positive(self.risk_budget, "risk_budget")
        _positive(self.max_portfolio_risk, "max_portfolio_risk")
        _non_negative(self.open_risk, "open_risk")
        if self.risk_budget > self.equity:
            raise ValueError("risk_budget cannot exceed equity")
        if self.max_portfolio_risk > self.equity:
            raise ValueError("max_portfolio_risk cannot exceed equity")
        if self.open_risk > self.max_portfolio_risk:
            raise ValueError("open_risk cannot exceed max_portfolio_risk")


@dataclass(frozen=True)
class TradeSpec:
    entry: float
    stop: float
    target: float
    direction: str
    fees_per_contract: float = 0.0
    slippage_per_side: float = 0.0

    def __post_init__(self) -> None:
        for value, name in ((self.entry, "entry"), (self.stop, "stop"), (self.target, "target")):
            _positive(value, name)
        if self.direction.lower() not in {"long", "short"}:
            raise ValueError("direction must be 'long' or 'short'")
        _non_negative(self.fees_per_contract, "fees_per_contract")
        _non_negative(self.slippage_per_side, "slippage_per_side")


@dataclass(frozen=True)
class Evidence:
    win_probability: float
    sample_count: int
    minimum_samples: int = 30
    average_win_r: float = 1.0
    average_loss_r: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.win_probability) or not 0.0 <= self.win_probability <= 1.0:
            raise ValueError("win_probability must be between 0 and 1")
        if self.sample_count < 0 or self.minimum_samples < 1:
            raise ValueError("sample_count must be non-negative and minimum_samples must be positive")
        _positive(self.average_win_r, "average_win_r")
        _positive(self.average_loss_r, "average_loss_r")

