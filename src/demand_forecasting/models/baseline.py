"""Baseline forecasters. Signature: fn(train, future, **params) -> DataFrame[id, date, y_pred]."""
from __future__ import annotations

import pandas as pd


def moving_average(train: pd.DataFrame, future: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Flat forecast = mean of the last `window` days."""
    last = train["date"].max()
    mu = train[train["date"] > last - pd.Timedelta(days=window)].groupby("id")["sales"].mean()
    out = future[["id", "date"]].copy()
    out["y_pred"] = out["id"].map(mu).astype(float)
    return out


def seasonal_naive(train: pd.DataFrame, future: pd.DataFrame, season: int = 7) -> pd.DataFrame:
    """Repeat the most recent observation of the same weekday."""
    t = train.assign(dow=train["date"].dt.dayofweek).sort_values("date")
    last = t.groupby(["id", "dow"])["sales"].last()
    out = future[["id", "date"]].copy()
    keys = pd.MultiIndex.from_arrays([out["id"], out["date"].dt.dayofweek])
    out["y_pred"] = last.reindex(keys).to_numpy(float)
    return out


def seasonal_average(train: pd.DataFrame, future: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Average of the same weekday over the last `weeks` weeks."""
    last = train["date"].max()
    t = train[train["date"] > last - pd.Timedelta(days=7 * weeks)].copy()
    t["dow"] = t["date"].dt.dayofweek
    prof = t.groupby(["id", "dow"])["sales"].mean()
    out = future[["id", "date"]].copy()
    keys = pd.MultiIndex.from_arrays([out["id"], out["date"].dt.dayofweek])
    out["y_pred"] = prof.reindex(keys).to_numpy(float)
    return out
