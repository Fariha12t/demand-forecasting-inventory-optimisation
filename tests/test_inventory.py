import numpy as np
import pytest
from scipy.stats import norm

from demand_forecasting.inventory import InventoryParams, recommend, simulate_policy


def _prm(**kw):
    base = dict(service_level=0.95, lead_time_days=7, review_period_days=7, unit_cost=4.0, pack_size=1, min_order_qty=0)
    base.update(kw)
    return InventoryParams(**base)


def test_safety_stock_and_reorder_point_formulas():
    f = np.full(28, 10.0)
    r = recommend(f, sigma_daily=3.0, on_hand=50, on_order=0, prm=_prm())
    z = norm.ppf(0.95)
    assert r["demand_over_lead_time"] == pytest.approx(70.0)
    assert r["safety_stock"] == pytest.approx(z * 3 * np.sqrt(7), abs=0.06)
    assert r["reorder_point"] == pytest.approx(70 + z * 3 * np.sqrt(7), abs=0.06)
    assert r["predicted_weekly_demand"] == pytest.approx(70.0)


def test_order_when_below_reorder_point_and_pack_rounding():
    f = np.full(28, 10.0)
    r = recommend(f, 3.0, on_hand=20, on_order=0, prm=_prm(pack_size=12))
    assert r["recommended_order_qty"] > 0 and r["recommended_order_qty"] % 12 == 0
    assert r["stockout_risk"] == "High"
    assert r["action"].startswith("Order")


def test_no_order_when_well_stocked_and_risk_monotonic():
    f = np.full(28, 10.0)
    rich = recommend(f, 3.0, on_hand=400, on_order=0, prm=_prm())
    poor = recommend(f, 3.0, on_hand=40, on_order=0, prm=_prm())
    assert rich["recommended_order_qty"] == 0 and rich["stockout_risk"] == "Low"
    assert poor["stockout_probability"] > rich["stockout_probability"]


def test_on_order_counts_towards_inventory_position():
    f = np.full(28, 10.0)
    a = recommend(f, 3.0, on_hand=20, on_order=0, prm=_prm())
    b = recommend(f, 3.0, on_hand=20, on_order=200, prm=_prm())
    assert b["recommended_order_qty"] == 0 < a["recommended_order_qty"]


def test_higher_service_level_means_more_safety_stock():
    f = np.full(28, 10.0)
    lo = recommend(f, 3.0, 50, 0, _prm(service_level=0.90))
    hi = recommend(f, 3.0, 50, 0, _prm(service_level=0.99))
    assert hi["safety_stock"] > lo["safety_stock"]


def test_simulation_good_forecast_beats_no_forecast():
    rng = np.random.default_rng(0)
    demand = rng.poisson(10, 56).astype(float)
    prm = _prm()
    good = simulate_policy(demand, np.full(56, 10.0), 3.2, prm, initial_on_hand=140)
    bad = simulate_policy(demand, np.full(56, 1.0), 0.5, prm, initial_on_hand=140)
    assert good["filled"] / good["demanded"] > bad["filled"] / bad["demanded"]
