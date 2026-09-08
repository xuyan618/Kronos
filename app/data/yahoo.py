"""Yahoo Finance collection with an injectable downloader for testing."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd

from .schema import normalize_observations


def _default_downloader(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Yahoo collection requires the declared yfinance dependency") from exc
    return yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)


def collect_yahoo(
    tickers: tuple[str, ...],
    start: str | None = None,
    end: str | None = None,
    available_at: datetime | None = None,
    downloader: Callable[[str, str | None, str | None], pd.DataFrame] = _default_downloader,
) -> pd.DataFrame:
    available = available_at or datetime.now(timezone.utc)
    rows = []
    for ticker in tickers:
        data = downloader(ticker, start, end)
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=-1, drop_level=True) if ticker in data.columns.get_level_values(-1) else data.droplevel(-1, axis=1)
        for timestamp, values in data.iterrows():
            observation_date = pd.Timestamp(timestamp).date()
            for field, value in values.items():
                if field not in {"Open", "High", "Low", "Close", "Adj Close", "Volume"} or pd.isna(value):
                    continue
                numeric = float(value)
                unit = "shares" if field == "Volume" else "price"
                revision = hashlib.sha256(f"{ticker}|{observation_date}|{field}|{numeric}".encode()).hexdigest()[:16]
                rows.append({
                    "series_id": f"yahoo:{ticker}:{field.lower().replace(' ', '_')}",
                    "observation_date": observation_date, "available_at": available,
                    "revision_id": revision, "value": numeric, "source": "yahoo",
                    "unit": unit, "frequency": "daily", "asset_name": ticker,
                    "category": "market", "field": field.lower().replace(" ", "_"),
                    "availability_method": "provider_download",
                })
    return normalize_observations(rows)


def persist_yahoo(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
