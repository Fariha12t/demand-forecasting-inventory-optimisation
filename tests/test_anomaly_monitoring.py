import numpy as np
import pandas as pd

from demand_forecasting.anomaly import detect_anomalies
from demand_forecasting.monitoring import psi, retrain_decision


def test_spike_is_detected_and_normal_days_are_not():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=200)
    sales = rng.poisson(20, 200)
    sales[150] = 120
    df = pd.DataFrame({"id": "A", "date": dates, "sales": sales})
    d = detect_anomalies(df, 8, 4.0)
    flagged = d[d["is_anomaly"]]
    assert dates[150] in set(flagged["date"])
    assert len(flagged) <= 3


def test_events_and_promos_are_not_flagged():
    dates = pd.date_range("2024-01-01", periods=120)
    sales = np.full(120, 20)
    sales[100] = 80
    df = pd.DataFrame({"id": "A", "date": dates, "sales": sales, "promo": 0, "is_event": 0})
    df.loc[100, "promo"] = 1
    assert not detect_anomalies(df)["is_anomaly"].any()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.2


def test_retrain_decision_triggers(tiny_cfg):
    import pandas as pd
    drift = pd.DataFrame({"feature": ["x"], "psi": [0.05], "drifted": [False]})
    ok = retrain_decision(drift, 0.30, 0.31, 3, tiny_cfg)
    assert not ok["needs_retrain"]
    bad = retrain_decision(drift, 0.30, 0.45, 3, tiny_cfg)
    assert bad["needs_retrain"] and "WAPE" in bad["reasons"][0]
    old = retrain_decision(drift, 0.30, 0.30, 90, tiny_cfg)
    assert old["needs_retrain"]
