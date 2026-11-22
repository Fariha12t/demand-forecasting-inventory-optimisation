import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demand_forecasting.config import resolve
from demand_forecasting.data.prepare import read_split


def test_split_is_chronological(trained, tiny_cfg):
    s = read_split(tiny_cfg)
    assert s["train_end"] < s["val_start"] <= s["val_end"] < s["test_start"] <= s["test_end"]


def test_models_forecast_only_their_window_and_no_leakage(trained, tiny_cfg):
    out, s = resolve(tiny_cfg, "outputs_dir"), read_split(tiny_cfg)
    val = pd.read_csv(out / "forecasts_val.csv", parse_dates=["date"])
    test = pd.read_csv(out / "forecasts_test.csv", parse_dates=["date"])
    assert val["date"].min() == pd.Timestamp(s["val_start"]) and val["date"].max() == pd.Timestamp(s["val_end"])
    assert test["date"].min() == pd.Timestamp(s["test_start"]) and test["date"].max() == pd.Timestamp(s["test_end"])
    assert {"LightGBM", "XGBoost", "RandomForest", "HoltWinters", "MovingAvg_28", "SeasonalNaive"} <= set(test["model"])
    assert test["y_pred"].ge(0).all() and test["y_pred"].notna().all()


def test_ml_beats_naive_baseline_on_test(trained):
    m = trained["metrics"]["test"].set_index("model")
    assert m.loc["LightGBM", "WAPE"] < m.loc["SeasonalNaive", "WAPE"]
    assert m.loc["LightGBM", "MASE"] < 1.0


def test_outputs_exist_and_recommendations_valid(trained, tiny_cfg):
    out = resolve(tiny_cfg, "outputs_dir")
    for f in ["metrics_val.csv", "metrics_test.csv", "recommendations.csv", "policy_simulation.csv", "price_elasticity.csv",
              "anomalies.csv", "monitoring_summary.json", "future_forecast.csv", "feature_importance.csv"]:
        assert (out / f).exists(), f
    r = pd.read_csv(out / "recommendations.csv")
    assert r["recommended_order_qty"].ge(0).all() and r["safety_stock"].ge(0).all()
    assert set(r["stockout_risk"]) <= {"Low", "Medium", "High"}
    assert (r["reorder_point"] >= r["demand_over_lead_time"]).all()
    assert len(json.loads((out / "monitoring_summary.json").read_text())) > 0


def test_price_elasticity_is_negative(trained):
    assert (trained["elasticity"]["elasticity"] < 0).all()


@pytest.fixture()
def client(trained, tiny_cfg, monkeypatch):
    import api.main as main
    monkeypatch.setattr(main, "ARTIFACTS", resolve(tiny_cfg, "models_dir") / "artifacts.joblib")
    monkeypatch.setattr(main, "PROJECT_ROOT", resolve(tiny_cfg, "models_dir").parent)
    with TestClient(main.app) as c:
        yield c


def test_api_endpoints(client):
    assert client.get("/health").json()["model_loaded"] is True
    sid = client.get("/series").json()["series_ids"][0]
    f = client.post("/forecast", json={"series_id": sid, "horizon": 14}).json()
    assert len(f["forecast"]) == 14 and f["total_units"] > 0
    r = client.post("/recommend", json={"series_id": sid, "current_stock": 5, "on_order": 0}).json()
    assert r["recommended_order_qty"] > 0 and r["stockout_risk"] in {"Low", "Medium", "High"}
    assert client.post("/forecast", json={"series_id": "NOPE"}).status_code == 404
    assert client.post("/forecast", json={"series_id": sid, "horizon": 999}).status_code == 422


def test_whatif_price_cut_raises_units(client):
    sid = client.get("/series").json()["series_ids"][0]
    w = client.post("/whatif", json={"series_id": sid, "price_change_pct": -25, "promo": True}).json()
    assert w["scenario"]["units"] > w["baseline"]["units"]


def test_database_roundtrip(tmp_path):
    from demand_forecasting.db import get_engine, read_table, write_table
    eng = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    df = pd.DataFrame({"id": ["a", "b"], "date": ["2025-01-01", "2025-01-02"], "y": [1.0, 2.0]})
    write_table(df, "forecasts", eng)
    assert len(read_table("forecasts", eng)) == 2
