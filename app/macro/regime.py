"""Deterministic macro regime classification."""


def classify_regime(row: dict) -> dict:
    """Return one mutually exclusive primary regime and stable secondary traits."""
    state = lambda name: row.get(f"{name}_state", "UNKNOWN")
    growth, inflation = state("growth"), state("inflation")
    if growth == "UNKNOWN" or inflation == "UNKNOWN":
        primary = "UNKNOWN"
    elif growth == "POSITIVE" and inflation != "NEGATIVE":
        primary = "EXPANSION"
    elif growth == "NEGATIVE" and inflation == "POSITIVE":
        primary = "STAGFLATION"
    elif growth == "NEGATIVE":
        primary = "CONTRACTION"
    else:
        primary = "TRANSITION"
    secondary = []
    if state("liquidity") == "POSITIVE" or state("financial_conditions") == "POSITIVE":
        secondary.append("EASING_LIQUIDITY")
    if state("rates") == "NEGATIVE" or state("yield_curve") == "POSITIVE":
        secondary.append("EASING_RATES")
    if state("credit") == "NEGATIVE" or state("volatility") == "NEGATIVE":
        secondary.append("RISK_STRESS")
    if state("risk_appetite") == "POSITIVE" and state("volatility") != "NEGATIVE":
        secondary.append("RISK_SEEKING")
    if not secondary:
        secondary.append("NEUTRAL")
    confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
    return {
        "primary_regime": primary,
        "secondary_characteristics": tuple(secondary),
        "regime_confidence": confidence,
        "regime_known": primary != "UNKNOWN",
    }
