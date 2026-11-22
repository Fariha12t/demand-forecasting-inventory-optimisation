"""Forecast accuracy metrics: MAE, RMSE, WAPE, MASE (+ bias)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p - y).sum() / max(y.sum(), 1e-9))


def mase_scale(train_y, season: int = 7) -> float:
    """In-sample MAE of the seasonal-naive forecast (denominator of MASE)."""
    y = np.asarray(train_y, float)
    if len(y) <= season:
        return float("nan")
    return float(np.mean(np.abs(y[season:] - y[:-season])))


def mase(y, p, scale: float) -> float:
    return mae(y, p) / scale if scale and scale > 0 else float("nan")


def series_scales(train: pd.DataFrame, season: int = 7) -> pd.Series:
    return train.sort_values("date").groupby("id")["sales"].apply(lambda s: mase_scale(s.to_numpy(), season))


def metrics_table(fc: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """fc columns: id, date, y_true, y_pred, model. Returns one row per model.

    MASE = mean over series of MAE_series / in-sample seasonal-naive MAE (scale-free; <1 beats seasonal naive).
    WAPE_weekly aggregates each series to weekly buckets first (inventory is ordered weekly).
    """
    scales = series_scales(train)
    rows = []
    for model, g in fc.groupby("model"):
        per = g.groupby("id").apply(lambda x: mase(x["y_true"], x["y_pred"], scales.get(x.name, np.nan)),
                                    include_groups=False)
        g = g.sort_values(["id", "date"]).copy()
        g["week"] = g.groupby("id").cumcount() // 7
        wk = g.groupby(["id", "week"])[["y_true", "y_pred"]].sum()
        rows.append({
            "model": model,
            "MAE": mae(g["y_true"], g["y_pred"]),
            "RMSE": rmse(g["y_true"], g["y_pred"]),
            "WAPE": wape(g["y_true"], g["y_pred"]),
            "MASE": float(per.mean()),
            "WAPE_weekly": wape(wk["y_true"], wk["y_pred"]),
            "Bias": bias(g["y_true"], g["y_pred"]),
        })
    return pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
