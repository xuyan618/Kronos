"""Provider status types and optional provider adapters."""

from dataclasses import dataclass
from enum import Enum
import os


class ProviderStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CollectionResult:
    status: ProviderStatus
    provider: str
    rows: int = 0
    path: str | None = None
    error: str | None = None


class FredAdapter:
    """Optional FRED adapter; credentials are read only from FRED_API_KEY."""

    provider = "fred"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def collect(self, *args, **kwargs):
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is required for the optional FRED adapter")
        raise NotImplementedError(
            "FRED support is not configured in this repository; use an approved FRED client adapter."
        )
