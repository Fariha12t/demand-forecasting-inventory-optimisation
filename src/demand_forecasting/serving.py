"""Model serving core: load artefacts, forecast, recommend inventory, run what-if scenarios.

Used by the FastAPI service, the Streamlit dashboard and the pipeline itself.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .events import add_event_columns
from .features import BASE_COLS, recursive_forecast
from .inventory import params_from_config, recommend as _recommend


def build_future_frame(history: pd.DataFrame, horizon: int, price_multiplier: float = 1.0, promo: bool = False,
                       start_offset: int = 0, duration: int | None = None) -> pd.DataFrame:
    """Known future inputs (price, promo, events) for the `horizon` days after the history ends.

    Baseline plan = regular price (max of the last 56 days) and no promotion. A scenario applies
    `price_multiplier` / `promo` on days [start_offset, start_offset + duration).
    """
    last = history["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    recent = history[history["date"] > last - pd.Timedelta(days=56)]
    ref = recent.groupby("id")["sell_price"].max().rename("ref_price")
    meta = history.drop_duplicates("id")[["id", "item_id", "dept_id", "cat_id", "store_id"]].set_index("id")
    n_s = len(meta)
    fut = pd.DataFrame({
        "id": np.tile(meta.index.to_numpy(), horizon),
        "date": np.repeat(dates, n_s),
        "offset": np.repeat(np.arange(horizon), n_s),
    })
    fut = fut.join(meta, on="id").join(ref, on="id")
    end = horizon if duration is None else start_offset + duration
    active = ((fut["offset"] >= start_offset) & (fut["offset"] < end)).to_numpy()
    fut["sell_price"] = np.where(active, fut["ref_price"] * price_multiplier, fut["ref_price"]).round(2)
    fut["promo"] = (active & bool(promo)).astype("int8")
    fut = add_event_columns(fut)
    fut["sales"] = np.nan
    return fut.sort_values(["date", "id"]).reset_index(drop=True)


class Forecaster:
    def __init__(self, artifacts_path: str | Path):
        art = joblib.load(artifacts_path)
        self.art = art
        self.model = art["model"]
        self.cfg = art["config"]
        self.encoder = art["encoder"]
        self.history: pd.DataFrame = art["history"]
        self.sigma: dict = art["sigma"]
        self.master: pd.DataFrame = art["item_master"]
        self.snapshot: pd.DataFrame = art["snapshot"]
        self.series_ids = sorted(self.history["id"].unique())

    def info(self) -> dict:
        a = self.art
        return {"model_name": a["model_name"], "trained_through": a["trained_through"], "created_at": a["created_at"],
                "n_series": len(self.series_ids), "features": a["features"], "validation_metrics": a.get("metrics_val", {})}

    def _history(self, series_id: str | None) -> pd.DataFrame:
        if series_id is None:
            return self.history
        if series_id not in self.series_ids:
            raise KeyError(f"Unknown series_id {series_id!r}")
        return self.history[self.history["id"] == series_id]

    def forecast(self, series_id: str | None = None, horizon: int = 28, price_multiplier: float = 1.0,
                 promo: bool = False, start_offset: int = 0, duration: int | None = None) -> pd.DataFrame:
        hist = self._history(series_id)
        fut = build_future_frame(hist, horizon, price_multiplier, promo, start_offset, duration)
        pred = recursive_forecast(self.model.predict, hist, fut, self.encoder, self.cfg)
        return pred.merge(fut[["id", "date", "sell_price", "promo"]], on=["id", "date"]) \
            .sort_values(["id", "date"]).reset_index(drop=True)

    def recommend(self, series_id: str, current_stock: float | None = None, on_order: float | None = None,
                  lead_time_days: int | None = None, service_level: float | None = None,
                  horizon: int = 28) -> dict:
        fc = self.forecast(series_id, horizon)
        snap = self.snapshot.set_index("id")
        item = series_id.rsplit("_", 2)[0]
        m = self.master.set_index("item_id")
        lead = lead_time_days or (int(m.loc[item, "lead_time_days"]) if item in m.index else None)
        pack = int(m.loc[item, "pack_size"]) if item in m.index else 1
        price = float(fc["sell_price"].iloc[0])
        on_hand = current_stock if current_stock is not None else float(snap.loc[series_id, "on_hand"]) if series_id in snap.index else 0.0
        ord_ = on_order if on_order is not None else float(snap.loc[series_id, "on_order"]) if series_id in snap.index else 0.0
        prm = params_from_config(self.cfg, lead_time_days=lead, pack_size=pack, min_order_qty=pack,
                                 unit_cost=price * self.cfg["inventory"]["unit_cost_ratio"], service_level=service_level)
        sigma = float(self.sigma.get(series_id, fc["y_pred"].mean() * 0.5))
        rec = _recommend(fc["y_pred"].to_numpy(), sigma, on_hand, ord_, prm)
        return {"series_id": series_id, "as_of": str(self.history["date"].max().date()), "on_hand": on_hand,
                "on_order": ord_, "unit_price": price, "daily_sigma": round(sigma, 3), **rec}

    def whatif(self, series_id: str, price_change_pct: float = 0.0, promo: bool = False, start_offset: int = 0,
               duration: int | None = None, horizon: int = 28) -> dict:
        base = self.forecast(series_id, horizon)
        scen = self.forecast(series_id, horizon, 1 + price_change_pct / 100.0, promo, start_offset, duration)
        unit_cost = float(base["sell_price"].iloc[0]) * self.cfg["inventory"]["unit_cost_ratio"]
        def tot(f):
            return {"units": float(f["y_pred"].sum()), "revenue": float((f["y_pred"] * f["sell_price"]).sum()),
                    "gross_margin": float((f["y_pred"] * (f["sell_price"] - unit_cost)).sum())}
        b, s = tot(base), tot(scen)
        return {"series_id": series_id, "baseline": b, "scenario": s,
                "delta": {k: s[k] - b[k] for k in b},
                "delta_pct": {k: (s[k] / b[k] - 1) * 100 if b[k] else 0.0 for k in b},
                "daily": [{"date": str(d.date()), "baseline": float(x), "scenario": float(y),
                           "baseline_price": float(pb), "scenario_price": float(ps)}
                          for d, x, y, pb, ps in zip(base["date"], base["y_pred"], scen["y_pred"],
                                                      base["sell_price"], scen["sell_price"])],
                "note": "Model-based estimate learned from historical price/promo variation; "
                        "validate with a controlled test before large price moves."}
