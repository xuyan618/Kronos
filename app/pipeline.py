"""Orchestration for the optional, report-producing Kronos-Macro pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.data.config import DataConfig
from app.data.engine import collect_market_data
from app.data.providers import CollectionResult, ProviderStatus
from app.data.yahoo import _default_downloader
from app.kronos import ForecastConfig, KronosForecastAdapter
from app.schemas import AccountRisk, Evidence, InstrumentMetadata, TradeSpec
from app.risk import assess_risk
from app.decision import make_decision


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _status(status: str, **values: Any) -> dict[str, Any]:
    return {"status": status, **values}


def _market_history(path: str, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[frame["asset_name"].eq(symbol)].copy()
    if selected.empty:
        raise ValueError(f"no normalized Yahoo observations found for symbol {symbol}")
    selected["timestamp"] = pd.to_datetime(selected["observation_date"], errors="raise")
    selected = selected.pivot_table(
        index="timestamp", columns="field", values="value", aggfunc="last"
    ).reset_index()
    selected.columns.name = None
    selected = selected.rename(columns={"adj_close": "adj_close"})
    missing = [column for column in ("open", "high", "low", "close", "volume")
               if column not in selected.columns]
    if missing:
        raise ValueError(f"normalized data is missing forecast columns: {missing}")
    return selected.sort_values("timestamp").reset_index(drop=True)


def _risk_report(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    risk_required = (
        "entry", "stop", "target", "direction", "equity", "risk_budget",
        "max_portfolio_risk", "contract_multiplier", "tick_size",
    )
    missing = [name for name in risk_required if inputs.get(name) is None]
    if missing:
        return (
            _status("skipped", reason="risk inputs are incomplete", missing=missing),
            _status("skipped", reason="risk inputs are incomplete", missing=missing),
        )
    instrument = InstrumentMetadata(
        inputs["symbol"], inputs.get("asset_class", "equity"),
        inputs.get("quote_currency", "USD"), inputs["contract_multiplier"],
        inputs["tick_size"], inputs.get("lot_size", 1.0),
    )
    account = AccountRisk(
        inputs["equity"], inputs["risk_budget"], inputs["max_portfolio_risk"],
        inputs.get("open_risk", 0.0),
    )
    trade = TradeSpec(
        inputs["entry"], inputs["stop"], inputs["target"], inputs["direction"],
        inputs.get("fees_per_contract", 0.0), inputs.get("slippage_per_side", 0.0),
    )
    risk = assess_risk(instrument, account, trade)
    risk_result = _status("success", **risk.__dict__)
    risk_result["vetoes"] = list(risk.vetoes)
    risk_result["explanations"] = list(risk.explanations)

    evidence_required = ("win_probability", "sample_count")
    missing_evidence = [name for name in evidence_required if inputs.get(name) is None]
    if missing_evidence:
        return risk_result, _status(
            "skipped", reason="evidence inputs were not supplied", missing=missing_evidence
        )
    evidence = Evidence(
        inputs["win_probability"], inputs["sample_count"],
        inputs.get("minimum_samples", 30), inputs.get("average_win_r", 1.0),
        inputs.get("average_loss_r", 1.0),
    )
    decision = make_decision(risk, evidence)
    decision_result = _status("success", **decision.__dict__)
    decision_result["vetoes"] = list(decision.vetoes)
    decision_result["contributions"] = list(decision.contributions)
    return risk_result, decision_result


def run_pipeline(
    *,
    data_config: DataConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    symbol: str | None = None,
    forecast_horizon: int | None = None,
    model_config: ForecastConfig | None = None,
    risk_inputs: dict[str, Any] | None = None,
    downloader: Callable[..., pd.DataFrame] | None = None,
    adapter_factory: Callable[..., KronosForecastAdapter] | None = None,
) -> dict[str, Any]:
    """Collect data and optionally produce forecast, risk, and decision sections."""
    started_at = _now()
    config = data_config or DataConfig.from_env()
    result: CollectionResult = collect_market_data(
        config, start=start, end=end, downloader=downloader or _default_downloader
    )
    report: dict[str, Any] = {
        "started_at": started_at,
        "completed_at": None,
        "data": {
            "yahoo": _status(result.status.value, provider=result.provider,
                             rows=result.rows, path=result.path, error=result.error),
            "macro": _status("skipped", reason="no macro provider configured; FRED credentials are not required"),
            "factor": _status("skipped", reason="no factor provider configured"),
        },
        "forecast": _status("skipped", reason="forecast was not requested"),
        "risk": _status("skipped", reason="risk inputs were not supplied"),
        "decision": _status("skipped", reason="risk inputs were not supplied"),
        "confidence": None,
        "limitations": [
            "Yahoo Finance data is external and may be delayed, revised, or unavailable.",
            "KronosForecastAdapter does not provide calibrated confidence or prediction ranges.",
            "A TRADE decision requires caller-supplied evidence; no probabilities are inferred.",
        ],
    }
    if result.status is ProviderStatus.SUCCESS and symbol and forecast_horizon:
        try:
            history = _market_history(result.path, symbol)
            if adapter_factory is None:
                adapter = KronosForecastAdapter.from_pretrained(config=model_config)
            else:
                adapter = adapter_factory(model_config)
            forecast = adapter.forecast(history, forecast_horizon)
            report["forecast"] = _status("success", symbol=symbol, result=forecast.to_dict())
        except (ImportError, ModuleNotFoundError) as exc:
            report["forecast"] = _status(
                "failed", reason="Kronos model dependencies are unavailable; install requirements.txt",
                error=str(exc),
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            report["forecast"] = _status("failed", reason="forecast could not be produced", error=str(exc))
        report["confidence"] = report["forecast"].get("result", {}).get("confidence")
    if risk_inputs:
        try:
            risk, decision = _risk_report({**risk_inputs, "symbol": symbol or risk_inputs.get("symbol", "unknown")})
            report["risk"], report["decision"] = risk, decision
        except (TypeError, ValueError) as exc:
            report["risk"] = _status("failed", reason="risk inputs are invalid", error=str(exc))
            report["decision"] = _status("skipped", reason="risk calculation failed")
    report["completed_at"] = _now()
    return report

