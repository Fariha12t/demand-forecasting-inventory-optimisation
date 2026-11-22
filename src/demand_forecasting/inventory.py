"""Inventory optimisation: safety stock, reorder point, order quantity, stockout risk, holding cost.

Continuous-demand (s, S) policy with periodic review:
    mu_L      expected demand over the supplier lead time L (from the forecast)
    sigma_L   sigma_d * sqrt(L)       (sigma_d = daily forecast-error std from validation residuals)
    SS        z * sigma_L             (z = Phi^-1(service level))
    ROP       mu_L + SS
    S         mu_{L+R} + z * sigma_d * sqrt(L+R)      (order-up-to level, R = review period)
    Order     S - inventory_position   if inventory_position <= ROP else 0   (rounded up to pack size / MOQ)
    Stockout risk  P(D_L > inventory_position) = 1 - Phi((IP - mu_L) / sigma_L)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class InventoryParams:
    service_level: float = 0.95
    lead_time_days: int = 7
    review_period_days: int = 7
    holding_cost_rate: float = 0.25
    order_cost: float = 25.0
    unit_cost: float = 5.0
    pack_size: int = 1
    min_order_qty: int = 0
    medium_risk: float = 0.10
    high_risk: float = 0.30


def params_from_config(cfg: dict, **overrides) -> InventoryParams:
    inv = cfg["inventory"]
    p = InventoryParams(
        service_level=inv["service_level"], lead_time_days=inv["default_lead_time_days"],
        review_period_days=inv["review_period_days"], holding_cost_rate=inv["holding_cost_rate"],
        order_cost=inv["order_cost"], medium_risk=inv["risk_thresholds"]["medium"],
        high_risk=inv["risk_thresholds"]["high"])
    for k, v in overrides.items():
        if v is not None and hasattr(p, k):
            setattr(p, k, v)
    return p


def _cum(daily: np.ndarray, n: int) -> float:
    """Expected demand over the first n days (extrapolates with the mean beyond the forecast horizon)."""
    daily = np.asarray(daily, float)
    n = int(n)
    if n <= len(daily):
        return float(daily[:n].sum())
    return float(daily.sum() + (n - len(daily)) * daily.mean())


def _risk_level(p: float, prm: InventoryParams) -> str:
    return "High" if p >= prm.high_risk else "Medium" if p >= prm.medium_risk else "Low"


def recommend(daily_forecast, sigma_daily: float, on_hand: float, on_order: float, prm: InventoryParams) -> dict:
    """Inventory recommendation for one SKU-location."""
    f = np.clip(np.asarray(daily_forecast, float), 0, None)
    L, R = max(int(prm.lead_time_days), 1), max(int(prm.review_period_days), 1)
    z = float(norm.ppf(prm.service_level))
    sigma_d = max(float(sigma_daily), 1e-6)

    mu_L = _cum(f, L)
    sigma_L = sigma_d * np.sqrt(L)
    safety = z * sigma_L
    rop = mu_L + safety
    order_up_to = _cum(f, L + R) + z * sigma_d * np.sqrt(L + R)
    ip = float(on_hand) + float(on_order)

    raw_qty = max(0.0, order_up_to - ip) if ip <= rop else 0.0
    qty = 0
    if raw_qty > 0:
        qty = max(raw_qty, prm.min_order_qty)
        pack = max(int(prm.pack_size), 1)
        qty = int(np.ceil(qty / pack) * pack)

    p_stockout = float(1.0 - norm.cdf((ip - mu_L) / sigma_L))
    mean_daily = max(float(f.mean()), 1e-6)
    h = prm.unit_cost * prm.holding_cost_rate                      # holding cost / unit / year
    cycle_stock = mean_daily * R / 2.0
    avg_inventory = safety + cycle_stock
    annual_demand = mean_daily * 365.0
    eoq = float(np.sqrt(2.0 * annual_demand * prm.order_cost / max(h, 1e-9)))
    risk = _risk_level(p_stockout, prm)

    if qty > 0:
        action = f"Order {qty} units now"
    elif risk != "Low":
        action = "Monitor closely (risk elevated, position above reorder point)"
    else:
        action = "No order needed"
    return {
        "predicted_weekly_demand": round(_cum(f, 7), 1),
        "demand_over_lead_time": round(mu_L, 1),
        "safety_stock": round(safety, 1),
        "reorder_point": round(rop, 1),
        "order_up_to_level": round(order_up_to, 1),
        "inventory_position": round(ip, 1),
        "recommended_order_qty": qty,
        "stockout_probability": round(p_stockout, 3),
        "stockout_risk": risk,
        "days_of_cover": round(float(on_hand) / mean_daily, 1),
        "est_weekly_holding_cost": round(avg_inventory * h / 52.0, 2),
        "est_annual_holding_cost": round(avg_inventory * h, 2),
        "current_weekly_holding_cost": round(float(on_hand) * h / 52.0, 2),
        "eoq": round(eoq, 1),
        "action": action,
        "service_level": prm.service_level,
        "lead_time_days": L,
    }


def recommend_all(forecast: pd.DataFrame, sigma: pd.Series, snapshot: pd.DataFrame, master: pd.DataFrame,
                  prices: pd.Series, cfg: dict, service_level: float | None = None) -> pd.DataFrame:
    """Recommendations for every series.

    forecast: id, date, y_pred (daily, future) | sigma: id -> daily error std
    snapshot: id, on_hand, on_order | master: item_id, lead_time_days, pack_size | prices: id -> current price
    """
    snap = snapshot.set_index("id")
    m = master.set_index("item_id")
    rows = []
    for sid, g in forecast.sort_values("date").groupby("id"):
        item = sid.rsplit("_", 2)[0]
        lead = int(m.loc[item, "lead_time_days"]) if item in m.index else None
        pack = int(m.loc[item, "pack_size"]) if item in m.index else None
        price = float(prices.get(sid, 5.0))
        prm = params_from_config(cfg, lead_time_days=lead, pack_size=pack, min_order_qty=pack,
                                 unit_cost=price * cfg["inventory"]["unit_cost_ratio"],
                                 service_level=service_level)
        rec = recommend(g["y_pred"].to_numpy(), float(sigma.get(sid, g["y_pred"].mean() * 0.5)),
                        float(snap.loc[sid, "on_hand"]), float(snap.loc[sid, "on_order"]), prm)
        rows.append({"id": sid, "on_hand": int(snap.loc[sid, "on_hand"]), "on_order": int(snap.loc[sid, "on_order"]),
                     "unit_price": price, **rec})
    return pd.DataFrame(rows)


def _cum_from(f: np.ndarray, start: int, n: int) -> float:
    """Forecast demand for days [start, start+n); days beyond the forecast horizon repeat the last week's mean."""
    tail = float(f[-7:].mean())
    seg = f[start:start + n]
    return float(seg.sum() + max(n - len(seg), 0) * tail)


def simulate_policy(demand: np.ndarray, forecast: np.ndarray, sigma_daily: float, prm: InventoryParams,
                    initial_on_hand: float) -> dict:
    """Back-test a forecast-driven (s, S) policy against *actual* demand.

    The forecast is fixed at the origin (no later updates). Every day s_t and S_t are recomputed from the
    forecast for the next L (and L+R) days; when the inventory position <= s_t an order up to S_t is placed
    and arrives after the lead time. Unmet demand is lost.
    """
    L, R = max(int(prm.lead_time_days), 1), max(int(prm.review_period_days), 1)
    z = norm.ppf(prm.service_level)
    f = np.clip(np.asarray(forecast, float), 0, None)
    on_hand, pipeline = float(initial_on_hand), []          # pipeline: (arrival_day, qty)
    demanded = filled = 0.0
    stockout_days, inv_sum = 0, 0.0
    pack = max(int(prm.pack_size), 1)
    for t, d in enumerate(demand):
        on_hand += sum(q for a, q in pipeline if a == t)
        pipeline = [(a, q) for a, q in pipeline if a > t]
        ip = on_hand + sum(q for _, q in pipeline)
        s_t = _cum_from(f, t, L) + z * sigma_daily * np.sqrt(L)
        if ip <= s_t:
            S_t = _cum_from(f, t, L + R) + z * sigma_daily * np.sqrt(L + R)
            q = np.ceil(max(S_t - ip, 0) / pack) * pack
            if q > 0:
                pipeline.append((t + L, q))
        sold = min(on_hand, d)
        demanded += d
        filled += sold
        stockout_days += int(d > on_hand)
        on_hand -= sold
        inv_sum += on_hand
    n = len(demand)
    return {"demanded": demanded, "filled": filled, "stockout_days": stockout_days, "days": n,
            "avg_on_hand": inv_sum / n}


def params_dict(prm: InventoryParams) -> dict:
    return asdict(prm)
