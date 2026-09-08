"""The release-aware normalized observation contract."""

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

OBSERVATION_COLUMNS = (
    "series_id", "observation_date", "available_at", "revision_id", "value",
    "source", "unit", "frequency", "asset_name", "category", "field",
    "availability_method",
)


class ObservationValidationError(ValueError):
    """Raised when data cannot be represented by the observation contract."""


def _utc(value: object, name: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ObservationValidationError(f"{name} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Observation:
    series_id: str
    observation_date: date
    available_at: str
    revision_id: str
    value: float
    source: str
    unit: str
    frequency: str
    asset_name: str
    category: str
    field: str
    availability_method: str

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, key), str) and getattr(self, key).strip()
                   for key in ("series_id", "revision_id", "source", "unit", "frequency",
                               "asset_name", "category", "field", "availability_method")):
            raise ObservationValidationError("all identifier and metadata fields must be non-empty")
        if not isinstance(self.observation_date, date):
            raise ObservationValidationError("observation_date must be a date")
        if self.observation_date > date.today():
            raise ObservationValidationError("observation_date cannot be in the future")
        if not isinstance(self.value, (int, float)) or pd.isna(self.value):
            raise ObservationValidationError("value must be a finite number")
        if self.value in (float("inf"), float("-inf")):
            raise ObservationValidationError("value must be a finite number")
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["observation_date"] = self.observation_date.isoformat()
        return result


def normalize_observations(rows: Iterable[Mapping[str, object] | Observation]) -> pd.DataFrame:
    """Validate rows and return a stable, contract-shaped UTC DataFrame."""
    normalized = []
    for row in rows:
        observation = row if isinstance(row, Observation) else Observation(**dict(row))
        normalized.append(observation.to_dict())
    frame = pd.DataFrame(normalized, columns=OBSERVATION_COLUMNS)
    if not frame.empty:
        frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.date
        frame["available_at"] = frame["available_at"].map(lambda value: _utc(value, "available_at"))
        frame = frame.sort_values(["series_id", "observation_date", "field", "available_at"])
        frame = frame.reset_index(drop=True)
    return frame


def filter_as_of(frame: pd.DataFrame, as_of: object) -> pd.DataFrame:
    """Keep the latest release visible at *as_of* for each observation."""
    if set(frame.columns) != set(OBSERVATION_COLUMNS):
        raise ObservationValidationError("frame does not match the observation contract")
    cutoff = pd.Timestamp(_utc(as_of, "as_of"))
    available = pd.to_datetime(frame["available_at"], utc=True)
    result = frame.loc[available <= cutoff].copy()
    if result.empty:
        return result.reset_index(drop=True)
    result["_available"] = pd.to_datetime(result["available_at"], utc=True)
    result = (result.sort_values(["series_id", "observation_date", "field", "_available", "revision_id"])
                    .drop_duplicates(["series_id", "observation_date", "field"], keep="last")
                    .drop(columns="_available")
                    .reset_index(drop=True))
    return result
