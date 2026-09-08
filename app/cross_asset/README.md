# Cross-Asset Agent

`app.cross_asset` is a descriptive specialist for the future Chief Investment Agent.
It consumes the normalized observation frame from `app.data.schema`; it does not
collect data, call Kronos, or duplicate macro, risk, decision, or provider APIs.

`analyze_cross_asset(frame, as_of=...)` returns a typed `CrossAssetReport` containing
signals for stocks/bonds, stocks/USD, gold/rates, copper/growth, oil/inflation,
HYG/LQD, VIX/equities, DXY/emerging markets, and USD/INR/Indian assets. The latter
is `UNKNOWN` unless both legs are represented by available observations.

Each leg is filtered by `available_at <= as_of`, restricted to the target date, and
aligned on exact observation dates. Missing dates are not forward-filled. A signal
is `UNKNOWN` when it lacks enough aligned observations. Scores are bounded to
`[-2, 2]`, while `confidence` describes coverage of configured relationships; it
is not a probability or investment recommendation.

The output records input series, aligned dates, metric, threshold explanation, and
limitations so a downstream agent can apply its own policy. Provider observations
whose `availability_method` is `unknown` remain subject to the caller's
backtesting policy; this module does not silently promote them to point-in-time
data.
