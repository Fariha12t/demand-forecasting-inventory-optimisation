"""Forecast monitoring and drift detection (PSI on inputs, rolling WAPE on errors) + retrain decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import wape


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index (<0.1 stable, 0.1-0.2 moderate, >0.2 significant drift)."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(feats: pd.DataFrame, reference_end, recent_start, columns: list[str], threshold: float) -> pd.DataFrame:
    ref = feats[feats["date"] <= pd.Timestamp(reference_end)]
    rec = feats[feats["date"] >= pd.Timestamp(recent_start)]
    rows = [{"feature": c, "psi": psi(ref[c].dropna(), rec[c].dropna())} for c in columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["drifted"] = out["psi"] >= threshold
    return out


def weekly_error(fc: pd.DataFrame) -> pd.DataFrame:
    g = fc.sort_values(["id", "date"]).copy()
    g["week"] = g["date"].dt.to_period("W").dt.start_time
    return g.groupby("week").apply(lambda x: wape(x["y_true"], x["y_pred"]), include_groups=False) \
        .rename("wape").reset_index()


def retrain_decision(drift: pd.DataFrame, baseline_wape: float, recent_wape: float, model_age_days: float,
                     cfg: dict) -> dict:
    m = cfg["monitoring"]
    reasons = []
    n_drift = int(drift["drifted"].sum())
    if n_drift:
        reasons.append(f"{n_drift} feature(s) with PSI >= {m['psi_threshold']}")
    if baseline_wape > 0 and recent_wape > baseline_wape * (1 + m["wape_degradation"]):
        reasons.append(f"WAPE degraded {recent_wape / baseline_wape - 1:.0%} vs validation")
    if model_age_days > m["max_model_age_days"]:
        reasons.append(f"model older than {m['max_model_age_days']} days")
    return {"needs_retrain": bool(reasons), "reasons": reasons, "n_drifted_features": n_drift,
            "baseline_wape": baseline_wape, "recent_wape": recent_wape, "model_age_days": model_age_days}
