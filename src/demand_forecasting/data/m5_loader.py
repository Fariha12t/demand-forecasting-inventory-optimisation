"""Load the Kaggle M5 Forecasting - Accuracy dataset into the project's long format.

Download from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data and put
`calendar.csv`, `sales_train_evaluation.csv` (or `sales_train_validation.csv`) and
`sell_prices.csv` in data/raw/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..events import add_event_columns


def load_m5(raw_dir: Path, n_series: int = 60, history_days: int = 1095,
            stores: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    sales_file = next((raw_dir / f for f in ("sales_train_evaluation.csv", "sales_train_validation.csv")
                       if (raw_dir / f).exists()), None)
    if sales_file is None or not (raw_dir / "calendar.csv").exists() or not (raw_dir / "sell_prices.csv").exists():
        raise FileNotFoundError(
            f"M5 files not found in {raw_dir}. Expected calendar.csv, sell_prices.csv and "
            "sales_train_evaluation.csv (see data/README.md).")

    header = pd.read_csv(sales_file, nrows=0).columns
    d_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: "int16" for c in d_cols}
    sales = pd.read_csv(sales_file, dtype=dtypes)
    if stores:
        sales = sales[sales["store_id"].isin(stores)]

    keep_d = d_cols[-history_days:]
    recent = sales[keep_d]
    # rank by volume in the most recent year, keep top-N
    vol = recent.iloc[:, -365:].sum(axis=1)
    sales = sales.loc[vol.sort_values(ascending=False).index[: n_series * 3]]  # headroom for filtering

    cal = pd.read_csv(raw_dir / "calendar.csv", usecols=["date", "wm_yr_wk", "d"])
    cal["date"] = pd.to_datetime(cal["date"])
    prices = pd.read_csv(raw_dir / "sell_prices.csv")

    long = sales.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                      value_vars=keep_d, var_name="d", value_name="sales")
    long = long.merge(cal, on="d", how="left").merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.rename(columns={"sell_price": "sell_price"})

    # keep only series that are on sale (price known) for the whole window -> balanced panel
    ok = long.groupby("id")["sell_price"].apply(lambda s: s.notna().all())
    ok_ids = ok[ok].index
    totals = long[long["id"].isin(ok_ids)].groupby("id")["sales"].sum().sort_values(ascending=False)
    chosen = totals.index[:n_series]
    long = long[long["id"].isin(chosen)].copy()

    long["id"] = long["id"].str.replace("_evaluation", "").str.replace("_validation", "")
    long = long.sort_values(["id", "date"])
    ref = long.groupby("id")["sell_price"].transform(lambda s: s.rolling(91, min_periods=1).max())
    long["promo"] = (long["sell_price"] < 0.95 * ref).astype("int8")  # M5 has no promo flag -> derive from discounts
    long = add_event_columns(long)
    long = long[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sales",
                 "sell_price", "promo", "event_name", "is_event"]]
    long = long.sort_values(["date", "id"]).reset_index(drop=True)

    items = long[["item_id", "dept_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(42)  # M5 has no supply data -> assumed lead times / pack sizes
    items["lead_time_days"] = rng.choice([3, 5, 7, 10, 14], len(items))
    items["pack_size"] = rng.choice([1, 6, 12], len(items), p=[0.5, 0.3, 0.2])
    items["supplier"] = [f"SUP_{(i % 4) + 1}" for i in range(len(items))]
    return long, items
