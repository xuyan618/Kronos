"""Run the optional Kronos-Macro integration and write a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.data.config import DataConfig
from app.pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--symbol", help="symbol to forecast (must be one of --ticker)")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--forecast-horizon", type=int)
    parser.add_argument("--report", type=Path, default=Path("macro_pipeline_report.json"))
    parser.add_argument("--entry", type=float)
    parser.add_argument("--stop", type=float)
    parser.add_argument("--target", type=float)
    parser.add_argument("--direction", choices=("long", "short"))
    parser.add_argument("--equity", type=float)
    parser.add_argument("--risk-budget", type=float)
    parser.add_argument("--max-portfolio-risk", type=float)
    parser.add_argument("--open-risk", type=float, default=0.0)
    parser.add_argument("--contract-multiplier", type=float)
    parser.add_argument("--tick-size", type=float)
    parser.add_argument("--lot-size", type=float, default=1.0)
    parser.add_argument("--fees-per-contract", type=float, default=0.0)
    parser.add_argument("--slippage-per-side", type=float, default=0.0)
    parser.add_argument("--win-probability", type=float)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--average-win-r", type=float, default=1.0)
    parser.add_argument("--average-loss-r", type=float, default=1.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    tickers = tuple(item.upper() for item in (args.tickers or ["SPY", "QQQ", "DIA"]))
    symbol = (args.symbol or tickers[0]).upper()
    inputs: dict[str, Any] = {
        key: getattr(args, key) for key in (
            "entry", "stop", "target", "direction", "equity", "risk_budget",
            "max_portfolio_risk", "open_risk", "contract_multiplier", "tick_size",
            "lot_size", "fees_per_contract", "slippage_per_side", "win_probability",
            "sample_count", "minimum_samples", "average_win_r", "average_loss_r",
        )
    }
    config = DataConfig(tickers=tickers, root=args.data_root or DataConfig.from_env().root)
    report = run_pipeline(
        data_config=config, start=args.start, end=args.end, symbol=symbol,
        forecast_horizon=args.forecast_horizon, risk_inputs=inputs,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

