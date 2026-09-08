# Kronos-Macro Implementation Specification

## 1. Scope and repository boundaries

Kronos-Macro is a research and backtesting system combining:

`data -> macro factors -> regime -> cross-asset -> sectors -> technicals -> Kronos forecast -> risk -> expected value -> decision`

The application code belongs in this repository. Any upstream Kronos checkout is an external, read-only dependency and must never be edited by application code, tests, or formatting tools. Its path and import mechanism must be configured; no source-tree path is assumed.

The system is analytical only. It must expose uncertainty and may return `NO_TRADE`; it must not represent a forecast as guaranteed or as investment advice.

## 2. Required data contract

Every normalized observation must contain:

| Field | Requirement |
|---|---|
| `series_id` | Stable provider-independent identifier |
| `observation_date` | Timestamp the value describes, timezone-aware |
| `available_at` | Earliest timestamp at which this exact value was available to the application |
| `revision_id` | Provider vintage/revision identifier when available |
| `value` | Numeric value; invalid values are errors |
| `source` | Provider name |
| `unit` | Explicit unit |
| `frequency` | `D`, `W`, `M`, `Q`, or provider-defined frequency |
| `asset_name`, `category`, `field` | Canonical metadata |
| `availability_method` | `release_timestamp`, `vintage`, `vendor_timestamp`, or `unknown` |

`available_at` is the only timestamp permitted for point-in-time filtering. `observation_date` must never be used as a substitute for publication time.

For FRED/ALFRED, preserve vintage/revision metadata and use the release timestamp when supplied. For providers without publication metadata, set `availability_method=unknown`, record the limitation, and exclude those observations from strict historical backtests unless the caller explicitly opts into an approximate mode. Never silently treat revised data as historically known.

All timestamps are stored in UTC. Provider-local timestamps must be converted before joins. Duplicate `(series_id, observation_date, revision_id)` records, invalid numeric values, future observations, and timezone-naive timestamps produce validation diagnostics and do not get silently discarded.

## 3. Data providers

Provider adapters are separate and report per-series status:

`SUCCESS`, `WARNING`, `SKIPPED`, or `FAILED`.

Yahoo Finance is used for market prices, yields, currencies, commodities, volatility, credit ETFs, indices, and sector proxies. FRED/ALFRED is used for US macro and liquidity series. Official Federal Reserve endpoints may be used only when documented and verified; otherwise use FRED. RBI and MoSPI are out of the active pipeline until an official, documented downloadable source is available.

Ticker mappings, FRED series IDs, units, adjustment policy, calendars, retry limits, and timeouts are centralized in configuration. Credentials come only from environment variables. `.env` is local and must not be committed; provide `.env.example`.

Market data must explicitly declare adjusted versus unadjusted prices, corporate-action treatment, and whether the dataset is suitable for backtesting. Sector and index data must document survivorship limitations.

## 4. Point-in-time processing

Every historical calculation accepts an `as_of` timestamp. A value is eligible only when:

`available_at <= as_of`

For mixed-frequency data, use release-aware as-of joins. Forward-filling is permitted only after a value becomes available, never before, and must be bounded by a configurable maximum staleness. Missing or stale inputs remain missing and reduce confidence; they are not replaced with current values.

The historical output must contain one row per requested evaluation timestamp and include the source timestamps used for every factor. Tests must prove that a release after `as_of` cannot affect the result.

## 5. Macro factor engine

Implement:

```text
app/macro/__init__.py
app/macro/signals.py
app/macro/factors.py
app/macro/regime.py
```

The engine produces historical and current rows for:

`growth`, `inflation`, `liquidity`, `rates`, `yield_curve`, `financial_conditions`, `credit`, `usd`, `volatility`, and `risk_appetite`.

Each factor output contains:

```text
factor
state
score             # integer in [-2, 2]
raw_metrics
input_series
data_coverage
confidence        # float in [0, 1]
as_of
limitations
```

### 5.1 Signal methodology

Thresholds and lookback windows are configuration, not literals hidden in functions. Signals use changes, slopes, z-scores, percentiles, and relative performance where appropriate. A signal is `UNKNOWN` when required inputs are unavailable or stale.

Scores are target-specific. The first target is broad US equity risk appetite; another asset class must use a separate mapping rather than reusing equity signs.

Default broad-equity score direction:

| Factor state | Score rule |
|---|---|
| Growth accelerating / stable / slowing | `+2/+1/0` when accelerating supports equities; `-1/-2` when materially slowing or contracting |
| Inflation cooling / stable / hot | `+1/+2/0` for cooling/stable; `-1/-2` for persistent hot inflation |
| Liquidity easing / neutral / tightening | `+2/0/-2` |
| Rates falling / stable / rising | `+1/0/-1`, with configurable regime overrides |
| Yield curve steepening / flat / inverting | `+1/0/-1`; normal/inverted is retained separately |
| Financial conditions easing / neutral / tightening | `+2/0/-2` |
| Credit risk-on / neutral / risk-off | `+2/0/-2` |
| USD weak / neutral / strong | `+1/0/-1` |
| Volatility low / normal / high | `+1/0/-2` |
| Risk appetite strong / neutral / weak | `+2/0/-2` |

The mapping is documented and configurable; it is not a universal economic truth.

## 6. Macro score, confidence, and regime

Weights are configured by factor and must be non-negative. The engine rejects configurations with zero total weight. The normalized macro score is:

```text
macro_score = sum(weight_i * score_i) / (2 * sum(weight_i for available factors))
```

This yields `[-1, 1]`. Missing factors are excluded from the denominator and separately reported. If coverage is below the configured minimum, the score is `UNKNOWN` and cannot authorize a trade.

Confidence combines data coverage, signal strength, factor agreement, freshness, and cross-factor confirmation. It is a diagnostic estimate, not a forecast probability, and is bounded to `[0, 1]`.

Primary regimes are mutually exclusive and selected by documented precedence:

1. `STAGFLATION`: growth weak and inflation hot.
2. `INFLATIONARY`: inflation hot without stagflation.
3. `DISINFLATIONARY`: inflation cooling and growth not weak.
4. `RISK_OFF`: risk appetite/credit/volatility confirmation is negative.
5. `RISK_ON`: risk appetite/credit/conditions confirmation is positive.
6. `GROWTH`: growth positive without a stronger classification.
7. `TRANSITION`: conflicting or rapidly changing factors.
8. `NEUTRAL`: insufficient strength for another regime.

Secondary characteristics, such as `LIQUIDITY_EASING` or `RATES_FALLING`, are retained independently. Historical regimes are generated with the same logic as the current regime.

## 7. Cross-asset and sector modules

Cross-asset analysis measures returns, correlations, spreads, ratios, and regime-conditioned relationships for equities, bonds, USD, gold, oil, copper, VIX, and credit. Relationships are descriptive unless validated out of sample.

Sector rotation ranks the configured 11 sectors using relative performance versus the selected benchmark, momentum, trend, volatility, and risk-adjusted momentum. Rankings use only data available at each evaluation timestamp. Macro-sector relationships are contextual scores and may be overridden by observed market data.

## 8. Technical structure

Technical calculations support daily and weekly data first, followed by configurable intraday timeframes. Calculate trend, swing levels, support/resistance, ATR, volatility, moving averages, momentum, breakouts, pullbacks, market structure, volume, and relative volume. Levels must derive from historical observations and be timestamped; future candles cannot influence them.

## 9. Kronos adapter

Kronos is wrapped by an application adapter and is never imported through an assumed `from model import ...` path. The adapter records model and tokenizer identifiers, versions, device, timeframe, context length, and data window.

Device selection is configurable and defaults to CPU. MPS is opt-in only after a compatibility check. The adapter validates the actual upstream API at startup and raises a clear configuration error when unavailable.

Forecast horizons are limited to those supported by the selected model and input timeframe. A context window must not exceed the model's supported maximum. “One year of hourly data” is a source-history requirement, not a claim that all rows are passed to a 512-token context. Resampling and window selection are explicit.

Kronos output may include OHLC, direction, and return only when produced by the model. Forecast ranges and confidence are `UNAVAILABLE` unless calibrated on out-of-sample residuals or a documented probabilistic method. No long-horizon accuracy is implied.

## 10. Risk and expected value

Risk calculations require instrument metadata: account equity, risk budget, contract/lot multiplier, tick size, leverage, fees, slippage, and currency. Validate long and short stop placement, positive stop distance, quantity rounding, maximum position risk, and portfolio exposure. ATR may inform stops, but the actual rule is configurable and recorded.

```text
reward = abs(target - entry)
risk = abs(entry - stop)
rr = reward / risk
position_size = floor(max_risk_cash / (risk * contract_multiplier))
```

Expected value uses net outcomes:

```text
EV = p_win * net_average_win - p_loss * net_average_loss
```

Probabilities must come from out-of-sample historical results, with `p_win + p_loss = 1` after accounting for other outcomes. Minimum sample size, confidence intervals, fees, slippage, gaps, partial exits, and uncertainty are required. If evidence is insufficient, EV is `UNKNOWN`, not positive by default.

## 11. Decision engine

Decision components have configurable, validated weights. Each component exposes its contribution and explanation. Scores are normalized to a common range before weighting.

Hard `NO_TRADE` vetoes include:

* insufficient or non-point-in-time data;
* invalid risk geometry or portfolio limits;
* unknown or uncalibrated required forecast probability;
* configured minimum confidence not met;
* non-positive or unknown net EV;
* conflicting evidence above the configured conflict threshold.

Otherwise, the engine returns `LONG`, `SHORT`, or `NO_TRADE` using documented score thresholds. The result includes reasons, vetoes, input timestamps, confidence, risk/reward, and limitations.

## 12. Backtesting and validation

Backtests use only observations with `available_at <= decision_timestamp`, include transaction costs and slippage, and document survivorship and corporate-action limitations. Walk-forward evaluation separates history, forecast, and trade periods.

Required metrics include total return, CAGR, win/loss rate, profit factor, expectancy, average R, maximum drawdown, Sharpe, Sortino, Calmar, average trade, largest win/loss, and consecutive wins/losses. Report results by macro regime, sector, direction, volatility regime, and trend regime.

Tests cover configuration, normalization, provider failures, validation, point-in-time joins, macro factors, regimes, technical calculations, risk, EV, and Kronos adapter behavior. Root pytest configuration must collect only application tests and must not modify or collect upstream Kronos tests.

## 13. Implementation order

1. Verify repository boundaries and inspect the existing data engine.
2. Stabilize the normalized observation schema and point-in-time joins.
3. Implement and test the macro factor engine.
4. Implement historical/current regime output and persistence.
5. Add cross-asset, sector, technical, Kronos, risk, EV, and decision modules incrementally.
6. Add backtesting and walk-forward evaluation before any live monitoring or dashboard.

Each phase must preserve existing behavior, add focused tests, and report unsupported data or model capabilities explicitly.
