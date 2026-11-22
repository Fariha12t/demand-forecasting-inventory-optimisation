"""Forecasting & inventory API (FastAPI).

    uvicorn api.main:app --port 8000        ->  http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from demand_forecasting.config import PROJECT_ROOT
from demand_forecasting.serving import Forecaster

ARTIFACTS = Path(os.environ.get("DF_ARTIFACTS", PROJECT_ROOT / "models" / "artifacts.joblib"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ARTIFACTS.exists():
        state["fc"] = Forecaster(ARTIFACTS)
    yield
    state.clear()


app = FastAPI(title="Demand Forecasting & Inventory Optimisation API", version="1.0.0", lifespan=lifespan)


def _fc() -> Forecaster:
    if "fc" not in state:
        raise HTTPException(503, f"Model artefacts not found at {ARTIFACTS}. Run `make train` first.")
    return state["fc"]


def _check(f: Forecaster, series_id: str) -> None:
    if series_id not in f.series_ids:
        raise HTTPException(404, f"Unknown series_id '{series_id}'")


class ForecastRequest(BaseModel):
    series_id: str
    horizon: int = Field(28, ge=1, le=56, description="days ahead")


class RecommendRequest(BaseModel):
    series_id: str
    current_stock: float | None = Field(None, ge=0, description="units on hand (defaults to the latest snapshot)")
    on_order: float | None = Field(None, ge=0)
    lead_time_days: int | None = Field(None, ge=1, le=90)
    service_level: float | None = Field(None, gt=0.5, lt=0.9999)


class WhatIfRequest(BaseModel):
    series_id: str
    price_change_pct: float = Field(0.0, ge=-50, le=50, description="e.g. -10 = 10% price cut")
    promo: bool = False
    start_offset_days: int = Field(0, ge=0, le=27)
    duration_days: int | None = Field(None, ge=1, le=56)
    horizon: int = Field(28, ge=1, le=56)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "fc" in state}


@app.get("/model-info")
def model_info():
    return _fc().info()


@app.get("/series")
def series(limit: int = Query(100, ge=1, le=5000)):
    return {"count": len(_fc().series_ids), "series_ids": _fc().series_ids[:limit]}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    f = _fc()
    _check(f, req.series_id)
    fc = f.forecast(req.series_id, req.horizon)
    return {"series_id": req.series_id, "model": f.info()["model_name"],
            "forecast": [{"date": str(r.date.date()), "units": round(float(r.y_pred), 2), "price": float(r.sell_price)}
                         for r in fc.itertuples()],
            "total_units": round(float(fc["y_pred"].sum()), 1),
            "weekly_units": [round(float(x), 1) for x in fc["y_pred"].groupby(fc.index // 7).sum()]}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    f = _fc()
    _check(f, req.series_id)
    return f.recommend(req.series_id, req.current_stock, req.on_order, req.lead_time_days, req.service_level)


@app.post("/whatif")
def whatif(req: WhatIfRequest):
    f = _fc()
    _check(f, req.series_id)
    return f.whatif(req.series_id, req.price_change_pct, req.promo, req.start_offset_days, req.duration_days, req.horizon)


@app.get("/recommendations")
def recommendations(risk: str | None = Query(None, pattern="^(Low|Medium|High)$"), limit: int = Query(100, ge=1, le=5000)):
    """Pre-computed recommendations for all SKU-locations (from the latest pipeline run)."""
    path = PROJECT_ROOT / "outputs" / "recommendations.csv"
    if not path.exists():
        raise HTTPException(503, "Run `make train` first.")
    df = pd.read_csv(path)
    if risk:
        df = df[df["stockout_risk"] == risk]
    return df.sort_values("stockout_probability", ascending=False).head(limit).to_dict(orient="records")
