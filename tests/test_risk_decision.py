import pytest

from app.decision import make_decision
from app.risk import assess_risk
from app.schemas import AccountRisk, Evidence, InstrumentMetadata, TradeSpec


@pytest.fixture
def instrument():
    return InstrumentMetadata("ES", "future", "USD", contract_multiplier=50, tick_size=0.25)


@pytest.fixture
def account():
    return AccountRisk(equity=100_000, risk_budget=1_000, max_portfolio_risk=2_000)


def test_long_trade_is_sized_with_costs_and_explained(instrument, account):
    risk = assess_risk(
        instrument,
        account,
        TradeSpec(100, 98, 104, "long", fees_per_contract=5, slippage_per_side=0.25),
    )
    assert risk.valid
    assert risk.quantity == 7
    assert risk.total_risk == pytest.approx(7 * (100 + 25 + 5))
    assert risk.r_multiple == pytest.approx(200 / 130)
    assert any("Quantity" in line for line in risk.explanations)


def test_short_trade_and_positive_ev_trade(instrument, account):
    risk = assess_risk(instrument, account, TradeSpec(100, 102, 96, "short"))
    decision = make_decision(risk, Evidence(0.6, sample_count=100, average_win_r=1.5))
    assert risk.valid
    assert decision.action == "TRADE"
    assert decision.net_expected_value == pytest.approx(500.0)
    assert decision.vetoes == ()
    assert len(decision.contributions) >= 4


@pytest.mark.parametrize(
    ("trade", "veto"),
    [
        (TradeSpec(100, 101, 104, "long"), "stop_is_on_wrong_side"),
        (TradeSpec(100, 98, 99, "long"), "target_is_on_wrong_side"),
    ],
)
def test_invalid_levels_are_deterministic_no_trade(instrument, account, trade, veto):
    risk = assess_risk(instrument, account, trade)
    decision = make_decision(risk, Evidence(0.9, 100))
    assert not risk.valid
    assert decision.action == "NO_TRADE"
    assert veto in decision.vetoes


def test_insufficient_evidence_and_non_positive_ev_veto(instrument, account):
    risk = assess_risk(instrument, account, TradeSpec(100, 98, 104, "long"))
    decision = make_decision(risk, Evidence(0.4, sample_count=2, average_win_r=1.0))
    assert decision.action == "NO_TRADE"
    assert {"insufficient_evidence", "non_positive_expected_value"} <= set(decision.vetoes)


def test_portfolio_capacity_veto(instrument):
    account = AccountRisk(100_000, risk_budget=1_000, max_portfolio_risk=500, open_risk=500)
    risk = assess_risk(instrument, account, TradeSpec(100, 98, 104, "short"))
    assert not risk.valid
    assert "portfolio_risk_limit_reached" in risk.vetoes


def test_metadata_and_probability_are_required():
    with pytest.raises(ValueError):
        InstrumentMetadata("", "future", "USD", 1, 0.01)
    with pytest.raises(ValueError):
        Evidence(1.1, 30)
