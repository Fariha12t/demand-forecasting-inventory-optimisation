"""Demand anomaly detection: robust z-score against the same weekday's recent history.

Changes explained by a known event (e.g. Black Friday) or an active promotion are not flagged.

For each series and weekday we keep a rolling median / MAD of the previous `window_weeks` observations
(using only the past), so a one-day spike or collapse is flagged without being masked by weekly seasonality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_anomalies(df: pd.DataFrame, window_weeks: int = 8, threshold: float = 4.0,
                     overdispersion: float = 0.3) -> pd.DataFrame:
    extra = [c for c in ("anomaly_injected", "promo", "is_event") if c in df]
    d = df[["id", "date", "sales"] + extra].copy()
    d = d.sort_values(["id", "date"])
    d["dow"] = d["date"].dt.dayofweek
    g = d.groupby(["id", "dow"])["sales"]
    d["baseline"] = g.transform(lambda s: s.shift(1).rolling(window_weeks, min_periods=4).median())
    d["dev"] = (d["sales"] - d["baseline"]).abs()
    d["mad"] = d.groupby(["id", "dow"])["dev"].transform(lambda s: s.shift(1).rolling(window_weeks, min_periods=4).median())
    base = np.maximum(d["baseline"], 1.0)
    # noise scale: robust MAD, floored by an over-dispersed Poisson (negative-binomial-like) std
    scale = np.maximum(1.4826 * d["mad"], np.sqrt(base + overdispersion * base ** 2))
    d["z_score"] = (d["sales"] - d["baseline"]) / scale
    explained = ((d["promo"] == 1) | (d["is_event"] == 1)) if {"promo", "is_event"} <= set(d) else False
    d["explained_by_event_or_promo"] = explained
    d["is_anomaly"] = (d["z_score"].abs() >= threshold) & d["baseline"].notna() & ~explained
    d["type"] = np.where(d["z_score"] > 0, "spike", "drop")
    return d.sort_values(["date", "id"]).reset_index(drop=True)


def anomaly_scores(d: pd.DataFrame) -> dict:
    """Precision / recall against injected ground truth (synthetic data only)."""
    if "anomaly_injected" not in d:
        return {}
    ev = d[d["baseline"].notna()]
    tp = int(((ev["is_anomaly"]) & (ev["anomaly_injected"] == 1)).sum())
    fp = int(((ev["is_anomaly"]) & (ev["anomaly_injected"] == 0)).sum())
    fn = int(((~ev["is_anomaly"]) & (ev["anomaly_injected"] == 1)).sum())
    inj = ev[ev["anomaly_injected"] == 1]
    by_type = {t: float(inj.loc[inj["sales"].gt(inj["baseline"]) == (t == "spike"), "is_anomaly"].mean())
               for t in ("spike", "drop")}
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "recall_spikes": by_type["spike"], "recall_drops": by_type["drop"]}
