"""Create the processed datasets (sales, item master, inventory snapshot, split dates)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import resolve
from .synthetic import generate
from .m5_loader import load_m5

log = logging.getLogger(__name__)


def make_inventory_snapshot(sales: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulated current stock position (the M5 dataset has no inventory data).

    In production replace this with a feed from the ERP / WMS.
    """
    rng = np.random.default_rng(seed)
    last = sales["date"].max()
    recent = sales[sales["date"] > last - pd.Timedelta(days=28)]
    weekly = recent.groupby("id")["sales"].mean() * 7
    snap = pd.DataFrame({"id": weekly.index})
    wk = weekly.to_numpy()
    snap["on_hand"] = np.round(wk * rng.uniform(0.2, 2.2, len(snap))).astype(int)
    snap["on_order"] = np.round(wk * 0.5 * (rng.random(len(snap)) < 0.2)).astype(int)
    snap["as_of_date"] = last.date().isoformat()
    return snap


def compute_split(dates: pd.Series, val_days: int, test_days: int) -> dict:
    d = np.sort(dates.unique())
    d = pd.DatetimeIndex(d)
    test_start, val_start = d[-test_days], d[-(test_days + val_days)]
    return {
        "train_end": str(d[-(test_days + val_days) - 1].date()),
        "val_start": str(val_start.date()),
        "val_end": str(d[-test_days - 1].date()),
        "test_start": str(test_start.date()),
        "test_end": str(d[-1].date()),
        "first_date": str(d[0].date()),
        "val_days": val_days,
        "test_days": test_days,
    }


def prepare(cfg: dict) -> dict:
    processed = resolve(cfg, "processed_dir")
    dcfg = cfg["data"]
    if dcfg["source"] == "synthetic":
        s = dcfg["synthetic"]
        sales, master = generate(s["n_items"], s["n_stores"], s["start_date"], s["n_days"],
                                 s["anomaly_rate"], cfg["project"]["seed"])
    elif dcfg["source"] == "m5":
        m = dcfg["m5"]
        sales, master = load_m5(resolve(cfg, "raw_dir"), m["n_series"], m["history_days"], m.get("stores"))
    else:
        raise ValueError(f"Unknown data source {dcfg['source']}")

    n_series, n_dates = sales["id"].nunique(), sales["date"].nunique()
    assert len(sales) == n_series * n_dates, "panel must be balanced (every series has every date)"
    split = compute_split(sales["date"], cfg["split"]["val_days"], cfg["split"]["test_days"])
    snap = make_inventory_snapshot(sales, cfg["project"]["seed"])

    sales.to_csv(processed / "sales.csv", index=False)
    master.to_csv(processed / "item_master.csv", index=False)
    snap.to_csv(processed / "inventory_snapshot.csv", index=False)
    (processed / "split.json").write_text(json.dumps(split, indent=2))
    log.info("Prepared %d series x %d days (%s) -> %s", n_series, n_dates, dcfg["source"], processed)
    return split


def read_sales(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg, "processed_dir") / "sales.csv", parse_dates=["date"], low_memory=False)
    if "event_name" in df:
        df["event_name"] = df["event_name"].fillna("")
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def read_split(cfg: dict) -> dict:
    return json.loads((resolve(cfg, "processed_dir") / "split.json").read_text())
