"""Statistical forecasters (statsmodels): Holt-Winters ETS and seasonal ARIMA.

The R pipeline (r/02_forecast_models.R) fits richer models (auto.arima, ETS, dynamic harmonic
regression) on the same splits; its forecasts are merged into the model comparison.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _fit_one(sid: str, y: np.ndarray, dates: pd.DatetimeIndex, kind: str, params: dict) -> pd.DataFrame:
    h = len(dates)
    fallback = np.full(h, float(np.mean(y[-28:])))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "holt_winters":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                m = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=7,
                                         initialization_method="estimated").fit()
                f = m.forecast(h)
            else:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                m = SARIMAX(y, order=tuple(params["order"]), seasonal_order=tuple(params["seasonal_order"]),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
                f = m.forecast(h)
        f = np.clip(np.asarray(f, float), 0, None)
        if not np.isfinite(f).all():
            f = fallback
    except Exception:
        f = fallback
    return pd.DataFrame({"id": sid, "date": dates, "y_pred": f})


def _run(train: pd.DataFrame, future: pd.DataFrame, kind: str, params: dict, train_days: int) -> pd.DataFrame:
    last = train["date"].max()
    tr = train[train["date"] > last - pd.Timedelta(days=train_days)].sort_values(["id", "date"])
    fut_dates = {sid: pd.DatetimeIndex(sorted(g["date"])) for sid, g in future.groupby("id")}
    jobs = [delayed(_fit_one)(sid, g["sales"].to_numpy(float), fut_dates[sid], kind, params)
            for sid, g in tr.groupby("id")]
    return pd.concat(Parallel(n_jobs=1)(jobs), ignore_index=True)


def holt_winters(train, future, train_days: int = 730):
    return _run(train, future, "holt_winters", {}, train_days)


def sarima(train, future, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7), train_days: int = 365):
    return _run(train, future, "sarima", {"order": order, "seasonal_order": seasonal_order}, train_days)
