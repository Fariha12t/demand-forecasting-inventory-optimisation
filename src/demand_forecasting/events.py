"""Rule-based retail event calendar (used for history *and* future dates)."""
from __future__ import annotations

import pandas as pd


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def events_for_year(year: int) -> dict[pd.Timestamp, str]:
    thanksgiving = _nth_weekday(year, 11, 3, 4)  # 4th Thursday of November
    return {
        pd.Timestamp(year, 1, 1): "NewYear",
        pd.Timestamp(year, 2, 14): "Valentines",
        pd.Timestamp(year, 7, 4): "IndependenceDay",
        pd.Timestamp(year, 10, 31): "Halloween",
        thanksgiving: "Thanksgiving",
        thanksgiving + pd.Timedelta(days=1): "BlackFriday",
        pd.Timestamp(year, 12, 24): "ChristmasEve",
        pd.Timestamp(year, 12, 25): "Christmas",
    }


def event_names(dates) -> pd.Series:
    """Event name for each date ('' when there is no event)."""
    idx = pd.DatetimeIndex(dates)
    mapping: dict[pd.Timestamp, str] = {}
    for y in sorted(set(idx.year)):
        mapping.update(events_for_year(int(y)))
    return pd.Series([mapping.get(d, "") for d in idx], index=idx)


def add_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    names = event_names(df["date"]).to_numpy()
    df = df.copy()
    df["event_name"] = names
    df["is_event"] = (df["event_name"] != "").astype("int8")
    return df
