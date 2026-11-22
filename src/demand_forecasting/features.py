"""Feature engineering and leak-free recursive multi-step forecasting.

Every feature for day *t* only uses information available before *t*:
  * lag_k          sales at t-k            (k = 7, 14, 28)
  * roll_mean_w    mean of sales t-w..t-1
  * calendar / event / price / promo features are *known in advance*
    (planned prices and promotions are inputs, as in the M5 competition).

The panel must be balanced (every series has every date) and sorted by (date, id).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "date", "sales", "sell_price", "promo", "is_event"]
CAT_SOURCE = {"item_code": "item_id", "store_code": "store_id", "dept_code": "dept_id", "cat_code": "cat_id"}
CAT_COLS = list(CAT_SOURCE)


class CategoryEncoder:
    """Stable integer codes for item / store / dept / category identifiers."""

    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        self.categories = {c: sorted(df[src].unique().tolist()) for c, src in CAT_SOURCE.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for code, src in CAT_SOURCE.items():
            out[code] = pd.Categorical(out[src], categories=self.categories[code]).codes.astype("int16")
        return out


def feature_columns(cfg: dict) -> list[str]:
    f = cfg["features"]
    cols = [f"lag_{k}" for k in f["lags"]]
    cols += [f"roll_mean_{w}" for w in f["windows"]]
    cols += [f"roll_std_{w}" for w in f["windows"] if w in (7, 28)]
    cols += ["dow", "is_weekend", "month", "quarter", "week_of_year", "day_of_month", "doy_sin", "doy_cos",
             "is_event", "sell_price", "price_rel", "discount", "price_change_7", "promo"]
    return cols + CAT_COLS


def make_features(df: pd.DataFrame, encoder: CategoryEncoder, cfg: dict) -> pd.DataFrame:
    """Return df (sorted by date,id) with all model features appended."""
    df = df.sort_values(["date", "id"]).reset_index(drop=True)
    n_series, n_dates = df["id"].nunique(), df["date"].nunique()
    if len(df) != n_series * n_dates:
        raise ValueError("Panel is not balanced: every series must have every date")
    f = cfg["features"]
    shape = (n_dates, n_series)

    sales = pd.DataFrame(df["sales"].to_numpy(float).reshape(shape))
    out: dict[str, np.ndarray] = {}
    for k in f["lags"]:
        out[f"lag_{k}"] = sales.shift(k).to_numpy().ravel()
    past = sales.shift(1)  # strictly excludes the current day
    for w in f["windows"]:
        out[f"roll_mean_{w}"] = past.rolling(w, min_periods=w).mean().to_numpy().ravel()
        if w in (7, 28):
            out[f"roll_std_{w}"] = past.rolling(w, min_periods=w).std().to_numpy().ravel()

    price = pd.DataFrame(df["sell_price"].to_numpy(float).reshape(shape))
    ref = price.rolling(f["price_ref_days"], min_periods=1).max()
    out["price_rel"] = (price / ref).to_numpy().ravel()
    out["discount"] = 1.0 - out["price_rel"]
    out["price_change_7"] = (price / price.shift(7)).fillna(1.0).to_numpy().ravel()

    d = df["date"].dt
    doy = d.dayofyear.to_numpy()
    out["dow"] = d.dayofweek.to_numpy()
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["month"] = d.month.to_numpy()
    out["quarter"] = d.quarter.to_numpy()
    out["week_of_year"] = d.isocalendar().week.to_numpy().astype(int)
    out["day_of_month"] = d.day.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    feats = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
    return encoder.transform(feats)


def training_frame(feats: pd.DataFrame, columns: list[str], date_from=None, date_to=None):
    """Rows with complete features within [date_from, date_to] -> (X, y, rows)."""
    m = np.ones(len(feats), dtype=bool)
    if date_from is not None:
        m &= (feats["date"] >= pd.Timestamp(date_from)).to_numpy()
    if date_to is not None:
        m &= (feats["date"] <= pd.Timestamp(date_to)).to_numpy()
    rows = feats[m]
    rows = rows.dropna(subset=columns + ["sales"])
    return rows[columns], rows["sales"].astype(float), rows


def recursive_forecast(predict_fn, history: pd.DataFrame, future: pd.DataFrame, encoder: CategoryEncoder,
                       cfg: dict, tail_days: int = 120) -> pd.DataFrame:
    """Multi-step forecast: predict day by day, feeding predictions back as lags.

    `future` carries the *known* exogenous inputs (price, promo, event) with no sales.
    """
    columns = feature_columns(cfg)
    last_dates = np.sort(history["date"].unique())[-tail_days:]
    hist = history[history["date"].isin(last_dates)][BASE_COLS]
    fut = future[[c for c in BASE_COLS if c != "sales" and c in future.columns]].copy()
    fut["sales"] = np.nan
    comb = pd.concat([hist, fut[BASE_COLS]], ignore_index=True).sort_values(["date", "id"]).reset_index(drop=True)
    for day in np.sort(fut["date"].unique()):
        feats = make_features(comb, encoder, cfg)
        mask = (feats["date"] == day).to_numpy()
        pred = np.clip(np.asarray(predict_fn(feats.loc[mask, columns])), 0, None)
        comb.loc[mask, "sales"] = pred
    res = comb[comb["date"].isin(fut["date"].unique())][["id", "date", "sales"]]
    return res.rename(columns={"sales": "y_pred"}).reset_index(drop=True)
