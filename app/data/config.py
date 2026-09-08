"""Centralized data paths and ticker configuration."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class DataConfig:
    tickers: tuple[str, ...] = ("SPY", "QQQ", "DIA")
    root: Path = field(default_factory=lambda: Path(os.getenv("KRONOS_DATA_ROOT", "data")))

    @property
    def raw_path(self) -> Path:
        return self.root / "raw"

    @property
    def processed_path(self) -> Path:
        return self.root / "processed"

    @property
    def cache_path(self) -> Path:
        return self.root / "cache"

    def ensure_paths(self) -> None:
        for path in (self.raw_path, self.processed_path, self.cache_path):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "DataConfig":
        tickers = tuple(item.strip().upper() for item in os.getenv("KRONOS_TICKERS", "").split(",") if item.strip())
        return cls(tickers=tickers or cls().tickers, root=Path(os.getenv("KRONOS_DATA_ROOT", "data")))
