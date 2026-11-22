"""Synthetic retail generator that mimics the M5 structure.

The simulator has *known* ground truth (price elasticities, seasonality,
promo uplift, injected anomalies) so the pipeline can be validated end to end
without downloading Kaggle data. Swap to real data with `data.source: m5`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import add_event_columns

# dept -> (category, true price elasticity)
DEPTS = {
    "FOODS_1": ("FOODS", -1.8),
    "FOODS_2": ("FOODS", -1.2),
    "HOBBIES_1": ("HOBBIES", -0.7),
    "HOUSEHOLD_1": ("HOUSEHOLD", -1.4),
}
TRUE_ELASTICITY = {d: v[1] for d, v in DEPTS.items()}
STORES = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1", "WI_2"]
EVENT_MULT = {
    "NewYear": 0.6, "Valentines": 1.15, "IndependenceDay": 1.2, "Halloween": 1.25,
    "Thanksgiving": 0.7, "BlackFriday": 1.8, "ChristmasEve": 1.35, "Christmas": 0.3,
}
# (amplitude, peak day-of-year) of yearly seasonality per category
YEARLY = {"FOODS": (0.12, 190), "HOBBIES": (0.25, 350), "HOUSEHOLD": (0.15, 120)}
DOW_BASE = np.array([0.85, 0.85, 0.90, 0.95, 1.10, 1.30, 1.20])  # Mon..Sun


def generate(n_items=20, n_stores=3, start_date="2023-01-01", n_days=1095,
             anomaly_rate=0.004, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales_long, item_master)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    items = [f"ITEM_{i + 1:03d}" for i in range(n_items)]
    stores = STORES[:n_stores]
    dept_names = list(DEPTS)
    item_dept = {it: dept_names[i % len(dept_names)] for i, it in enumerate(items)}

    series = [(it, st) for it in items for st in stores]
    ids = [f"{it}_{st}" for it, st in series]
    S, T = len(series), n_days

    dept_of = np.array([item_dept[it] for it, _ in series])
    cat_of = np.array([DEPTS[d][0] for d in dept_of])
    elas = np.array([DEPTS[d][1] for d in dept_of])
    store_mult = dict(zip(stores, rng.uniform(0.7, 1.3, len(stores))))
    base = rng.lognormal(1.9, 0.55, S) * np.array([store_mult[st] for _, st in series])
    price0_item = dict(zip(items, rng.uniform(2, 20, n_items).round(2)))
    price0 = np.array([price0_item[it] for it, _ in series])

    t = np.arange(T)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    # weekly pattern with mild dept-specific tilt
    tilt = rng.normal(0, 0.05, (S, 7))
    dow_mult = (DOW_BASE[None, :] + tilt)[:, dow].T                      # (T,S)
    amp = np.array([YEARLY[c][0] for c in cat_of])
    peak = np.array([YEARLY[c][1] for c in cat_of])
    yearly = 1 + amp[None, :] * np.cos(2 * np.pi * (doy[:, None] - peak[None, :]) / 365.25)
    growth = rng.uniform(0.0, 0.15, S)
    trend = 1 + growth[None, :] * (t[:, None] / 365.0)

    ev = add_event_columns(pd.DataFrame({"date": dates}))
    ev_mult = ev["event_name"].map(EVENT_MULT).fillna(1.0).to_numpy()[:, None]

    # regular price: small quarterly step changes
    steps = rng.normal(0, 0.02, (T // 90 + 1, S)).cumsum(axis=0)
    reg_price = price0[None, :] * np.exp(steps[t // 90])

    # promotions: random 7/14 day discount windows
    promo = np.zeros((T, S), dtype=bool)
    disc = np.zeros((T, S))
    for s in range(S):
        for _ in range(rng.poisson(6 * T / 365)):
            a = rng.integers(0, T - 14)
            ln = rng.choice([7, 7, 14])
            promo[a:a + ln, s] = True
            disc[a:a + ln, s] = rng.uniform(0.10, 0.30)
    price = (reg_price * (1 - disc)).round(2)

    price_effect = (price / reg_price) ** elas[None, :]
    promo_lift = np.where(promo, 1.12, 1.0)
    lam = base[None, :] * dow_mult * yearly * trend * ev_mult * price_effect * promo_lift
    noise = rng.gamma(shape=10.0, scale=0.1, size=(T, S))
    sales = rng.poisson(lam * noise).astype(float)

    # inject anomalies (spikes and sudden drops) with ground-truth flags
    anomaly = rng.random((T, S)) < anomaly_rate
    anomaly[:28] = False
    spike = rng.random((T, S)) < 0.6
    mult = rng.uniform(2.5, 5.0, (T, S))
    sales = np.where(anomaly & spike, np.ceil(sales * mult + 5), sales)
    sales = np.where(anomaly & ~spike, 0.0, sales)

    df = pd.DataFrame({
        "id": np.tile(ids, T),
        "item_id": np.tile([s[0] for s in series], T),
        "dept_id": np.tile(dept_of, T),
        "cat_id": np.tile(cat_of, T),
        "store_id": np.tile([s[1] for s in series], T),
        "state_id": np.tile([s[1].split("_")[0] for s in series], T),
        "date": np.repeat(dates, S),
        "sales": sales.ravel().astype(int),
        "sell_price": price.ravel(),
        "promo": promo.ravel().astype("int8"),
        "anomaly_injected": anomaly.ravel().astype("int8"),
    })
    df = add_event_columns(df).sort_values(["date", "id"]).reset_index(drop=True)

    lead = rng.choice([3, 5, 7, 10, 14], n_items, p=[0.15, 0.25, 0.3, 0.2, 0.1])
    pack = rng.choice([1, 6, 12], n_items, p=[0.5, 0.3, 0.2])
    master = pd.DataFrame({
        "item_id": items,
        "dept_id": [item_dept[i] for i in items],
        "lead_time_days": lead,
        "pack_size": pack,
        "supplier": [f"SUP_{(i % 4) + 1}" for i in range(n_items)],
    })
    return df, master
