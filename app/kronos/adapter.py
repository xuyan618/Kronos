"""A small, validated integration layer for the upstream Kronos predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
FORECAST_COLUMNS = PRICE_COLUMNS + ("volume", "amount")


@dataclass(frozen=True)
class ForecastConfig:
    """Runtime settings for a Kronos forecast."""

    model_id: str = "NeoQuasar/Kronos-small"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "cpu"
    max_context: int = 512
    timeframe: str = "1D"
    timestamp_column: str = "timestamp"
    validate_device: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 1


@dataclass(frozen=True)
class ForecastPoint:
    """One deterministic forecast row."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


@dataclass(frozen=True)
class ForecastResult:
    """Stable output contract for downstream consumers.

    KronosPredictor returns one averaged path when ``sample_count`` is used. It
    does not expose calibrated confidence or prediction intervals, so those
    fields are deliberately explicit ``None`` values rather than estimates.
    """

    timeframe: str
    context_length: int
    forecast_horizon: int
    forecasts: tuple[ForecastPoint, ...]
    confidence: None = None
    forecast_range: None = None
    unavailable: tuple[str, ...] = (
        "confidence",
        "forecast_range",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable field names."""

        result = asdict(self)
        result["forecasts"] = [asdict(point) for point in self.forecasts]
        result["unavailable"] = list(self.unavailable)
        return result


class KronosForecastAdapter:
    """Validate market history and invoke the repository's KronosPredictor."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ForecastConfig | None = None,
        predictor_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or ForecastConfig()
        self._validate_config()
        self._validate_device()

        if predictor_factory is None:
            from model import KronosPredictor

            predictor_factory = KronosPredictor
        self.predictor = predictor_factory(
            model,
            tokenizer,
            device=self.config.device,
            max_context=self.config.max_context,
        )

    @classmethod
    def from_pretrained(
        cls,
        config: ForecastConfig | None = None,
        *,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        predictor_factory: Callable[..., Any] | None = None,
    ) -> "KronosForecastAdapter":
        """Load the existing model and tokenizer, without changing upstream code."""

        effective_config = config or ForecastConfig()
        resolved_model_id = model_id or effective_config.model_id
        resolved_tokenizer_id = tokenizer_id or effective_config.tokenizer_id
        effective_config = replace(
            effective_config,
            model_id=resolved_model_id,
            tokenizer_id=resolved_tokenizer_id,
        )
        from model import Kronos, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(resolved_tokenizer_id)
        model = Kronos.from_pretrained(resolved_model_id)
        return cls(
            model,
            tokenizer,
            effective_config,
            predictor_factory=predictor_factory,
        )

    def forecast(
        self,
        history: pd.DataFrame,
        horizon: int,
        *,
        future_timestamps: Sequence[Any] | pd.Series | pd.DatetimeIndex | None = None,
    ) -> ForecastResult:
        """Forecast ``horizon`` rows from timestamped historical OHLCV data."""

        frame, x_timestamps = self._validate_history(history)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        if future_timestamps is None:
            offset = pd.tseries.frequencies.to_offset(self.config.timeframe)
            y_timestamps = pd.date_range(
                start=x_timestamps.iloc[-1] + offset,
                periods=horizon,
                freq=offset,
            )
        else:
            y_timestamps = pd.to_datetime(future_timestamps, errors="raise")
            if len(y_timestamps) != horizon:
                raise ValueError("future_timestamps length must equal horizon")
            if not y_timestamps.is_monotonic_increasing or y_timestamps.has_duplicates:
                raise ValueError("future_timestamps must be strictly increasing")

        prediction = self.predictor.predict(
            df=frame[list(FORECAST_COLUMNS)],
            x_timestamp=x_timestamps,
            y_timestamp=y_timestamps,
            pred_len=horizon,
            T=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            sample_count=self.config.sample_count,
            verbose=False,
        )
        return self._result_from_prediction(prediction, y_timestamps, len(frame), horizon)

    def _validate_history(self, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        timestamp_column = self.config.timestamp_column
        required = (timestamp_column,) + PRICE_COLUMNS + ("volume",)
        missing = [column for column in required if column not in history.columns]
        if missing:
            raise ValueError(f"history is missing required columns: {missing}")
        if history.empty:
            raise ValueError("history must contain at least one row")

        timestamps = pd.to_datetime(history[timestamp_column], errors="raise")
        if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError("history timestamps must be valid, increasing, and unique")

        frame = history.loc[:, list(PRICE_COLUMNS) + ["volume"]].copy()
        if "amount" in history.columns:
            frame["amount"] = history["amount"]
        else:
            frame["amount"] = frame["volume"] * frame["close"]
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("OHLCV values must be finite numbers")
        if (numeric["volume"] < 0).any() or (numeric["amount"] < 0).any():
            raise ValueError("volume and amount must be non-negative")
        if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any():
            raise ValueError("high must be at least open and close")
        if (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any():
            raise ValueError("low must be at most open and close")
        return numeric.astype(float), timestamps.reset_index(drop=True)

    def _result_from_prediction(
        self,
        prediction: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        context_length: int,
        horizon: int,
    ) -> ForecastResult:
        if not isinstance(prediction, pd.DataFrame):
            raise TypeError("KronosPredictor.predict must return a pandas DataFrame")
        missing = [column for column in FORECAST_COLUMNS if column not in prediction.columns]
        if missing or len(prediction) != horizon:
            raise ValueError("KronosPredictor returned an invalid forecast shape")
        values = prediction.loc[:, list(FORECAST_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("KronosPredictor returned non-finite forecast values")
        points = tuple(
            ForecastPoint(timestamp=timestamp.isoformat(), **row)
            for timestamp, row in zip(timestamps, values.to_dict(orient="records"))
        )
        return ForecastResult(
            timeframe=self.config.timeframe,
            context_length=context_length,
            forecast_horizon=horizon,
            forecasts=points,
        )

    def _validate_config(self) -> None:
        if self.config.max_context <= 0:
            raise ValueError("max_context must be positive")
        if self.config.sample_count <= 0 or self.config.top_k < 0:
            raise ValueError("sample_count must be positive and top_k non-negative")
        if not 0 < self.config.top_p <= 1 or self.config.temperature <= 0:
            raise ValueError("top_p must be in (0, 1] and temperature must be positive")
        pd.tseries.frequencies.to_offset(self.config.timeframe)

    def _validate_device(self) -> None:
        if not self.config.validate_device:
            return
        if self.config.device == "cpu":
            return
        import torch

        try:
            device = torch.device(self.config.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid torch device: {self.config.device}") from exc
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported torch device: {device.type}")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but MPS is unavailable")
