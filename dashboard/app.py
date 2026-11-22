"""Interactive forecasting & inventory dashboard (Streamlit).

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demand_forecasting.config import PROJECT_ROOT
from demand_forecasting.serving import Forecaster

OUT = Path(os.environ.get("DF_OUTPUTS", PROJECT_ROOT / "outputs"))
ARTIFACTS = Path(os.environ.get("DF_ARTIFACTS", PROJECT_ROOT / "models" / "artifacts.joblib"))
RISK_COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}

st.set_page_config(page_title="Demand Forecasting & Inventory", page_icon="📦", layout="wide")


@st.cache_data
def load_csv(name: str, parse_dates=None):
    p = OUT / name
    return pd.read_csv(p, parse_dates=parse_dates) if p.exists() else None


@st.cache_data
def load_json(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_forecaster():
    return Forecaster(ARTIFACTS) if ARTIFACTS.exists() else None


fc = load_forecaster()
if fc is None:
    st.title("📦 Demand Forecasting & Inventory Optimisation")
    st.warning("No trained model found. Run `make all` (or `python -m demand_forecasting.pipeline train`) first.")
    st.stop()

recs = load_csv("recommendations.csv")
mval, mtest = load_csv("metrics_val.csv"), load_csv("metrics_test.csv")
info = fc.info()

st.title("📦 Demand Forecasting & Inventory Optimisation")
st.caption(f"Serving model: **{info['model_name']}** · trained through {info['trained_through']} · "
           f"{info['n_series']} SKU-locations")

tabs = st.tabs(["Overview", "Forecast", "Inventory", "Model comparison", "Price & what-if", "Anomalies & monitoring"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    if recs is not None:
        c = st.columns(5)
        c[0].metric("SKU-locations", len(recs))
        c[1].metric("High stockout risk", int((recs["stockout_risk"] == "High").sum()))
        c[2].metric("Orders to place", int((recs["recommended_order_qty"] > 0).sum()))
        c[3].metric("Units to order", f"{int(recs['recommended_order_qty'].sum()):,}")
        c[4].metric("Est. weekly holding cost", f"${recs['est_weekly_holding_cost'].sum():,.0f}")
        left, right = st.columns([1, 2])
        counts = recs["stockout_risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        counts.columns = ["risk", "count"]
        left.plotly_chart(px.bar(counts, x="risk", y="count", color="risk", color_discrete_map=RISK_COLORS,
                                 title="Stockout risk (next lead time)").update_layout(showlegend=False),
                          width="stretch")
        right.subheader("Highest-risk SKU-locations")
        right.dataframe(recs.sort_values("stockout_probability", ascending=False).head(10)[
            ["id", "on_hand", "predicted_weekly_demand", "reorder_point", "stockout_probability", "stockout_risk",
             "recommended_order_qty", "action"]], hide_index=True, width="stretch")
    if mtest is not None:
        best = mtest.iloc[0]
        st.info(f"Best test-set model by WAPE: **{best['model']}** (WAPE {best['WAPE']:.1%}, weekly WAPE "
                f"{best['WAPE_weekly']:.1%}, MASE {best['MASE']:.2f}).")

# ------------------------------------------------------------------ Forecast
with tabs[1]:
    sid = st.selectbox("SKU-location", fc.series_ids, key="fc_sid")
    horizon = st.slider("Forecast horizon (days)", 7, 56, 28, key="fc_h")
    hist = fc.history[fc.history["id"] == sid].sort_values("date")
    pred = fc.forecast(sid, horizon)
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["sales"], name="Actual sales", line=dict(color="#555"))
    fig.add_scatter(x=pred["date"], y=pred["y_pred"], name="Forecast", line=dict(color="#1F4E79", width=3))
    fig.update_layout(title=f"{sid}: last 120 days and {horizon}-day forecast", height=420, yaxis_title="units / day")
    st.plotly_chart(fig, width="stretch")
    c = st.columns(3)
    c[0].metric("Next 7 days", f"{pred['y_pred'].head(7).sum():,.0f} units")
    c[1].metric(f"Next {horizon} days", f"{pred['y_pred'].sum():,.0f} units")
    c[2].metric("Avg daily", f"{pred['y_pred'].mean():.1f}")
    bt = load_csv("forecasts_test.csv", ["date"])
    if bt is not None:
        st.subheader("Back-test on the held-out test window")
        models = st.multiselect("Models", sorted(bt["model"].unique()),
                                default=[m for m in [info["model_name"], "SeasonalAvg_4w"] if m in set(bt["model"])])
        b = bt[(bt["id"] == sid) & (bt["model"].isin(models))]
        f2 = px.line(b, x="date", y="y_pred", color="model")
        truth = bt[(bt["id"] == sid)].drop_duplicates("date").sort_values("date")
        f2.add_scatter(x=truth["date"], y=truth["y_true"], name="Actual", line=dict(color="black", dash="dot"))
        f2.update_layout(height=350, yaxis_title="units / day")
        st.plotly_chart(f2, width="stretch")

# ------------------------------------------------------------------ Inventory
with tabs[2]:
    st.subheader("Replenishment recommendation")
    sid_i = st.selectbox("SKU-location", fc.series_ids, key="inv_sid")
    snap = fc.snapshot.set_index("id").loc[sid_i]
    c = st.columns(4)
    stock = c[0].number_input("Current stock (units)", 0, 100000, int(snap["on_hand"]))
    on_order = c[1].number_input("On order (units)", 0, 100000, int(snap["on_order"]))
    item = sid_i.rsplit("_", 2)[0]
    lead_default = int(fc.master.set_index("item_id").loc[item, "lead_time_days"])
    lead = c[2].number_input("Supplier lead time (days)", 1, 90, lead_default)
    sl = c[3].slider("Service level", 0.80, 0.995, 0.95, 0.005)
    r = fc.recommend(sid_i, stock, on_order, lead, sl)
    color = RISK_COLORS[r["stockout_risk"]]
    st.markdown(f"<div style='padding:12px;border-left:6px solid {color};background:#f6f8fa'>"
                f"<b>Stockout risk: <span style='color:{color}'>{r['stockout_risk']}</span></b> "
                f"({r['stockout_probability']:.0%}) &nbsp;|&nbsp; <b>{r['action']}</b></div>", unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Predicted weekly demand", f"{r['predicted_weekly_demand']:.0f}")
    c[1].metric("Safety stock", f"{r['safety_stock']:.0f}")
    c[2].metric("Reorder point", f"{r['reorder_point']:.0f}")
    c[3].metric("Order quantity", f"{r['recommended_order_qty']}")
    c[4].metric("Est. weekly holding cost", f"${r['est_weekly_holding_cost']:.2f}")
    st.caption(f"Demand over lead time {r['demand_over_lead_time']:.0f} · order-up-to level {r['order_up_to_level']:.0f} · "
               f"inventory position {r['inventory_position']:.0f} · days of cover {r['days_of_cover']} · EOQ {r['eoq']:.0f}. "
               "Stock levels come from a simulated snapshot — connect your ERP/WMS feed in production.")
    if recs is not None:
        st.subheader("All SKU-locations")
        f1, f2_ = st.columns(2)
        risk = f1.multiselect("Risk", ["High", "Medium", "Low"], default=["High", "Medium"])
        only_orders = f2_.checkbox("Only rows with an order to place", value=False)
        t = recs[recs["stockout_risk"].isin(risk)]
        if only_orders:
            t = t[t["recommended_order_qty"] > 0]
        st.dataframe(t.sort_values("stockout_probability", ascending=False), hide_index=True, width="stretch")
        st.download_button("Download recommendations (CSV)", recs.to_csv(index=False), "recommendations.csv")

# ------------------------------------------------------------------ Model comparison
with tabs[3]:
    if mval is None or mtest is None:
        st.info("Run the pipeline to generate metrics.")
    else:
        st.caption("Time-based split: train → validation (model selection) → test (final, untouched). "
                   "WAPE/MAE/RMSE on daily demand; MASE < 1 beats the in-sample seasonal-naive forecast; "
                   "WAPE_weekly aggregates to weekly buckets (the ordering cadence).")
        a, b = st.columns(2)
        a.subheader("Validation")
        a.dataframe(mval.round(3), hide_index=True, width="stretch")
        b.subheader("Test")
        b.dataframe(mtest.round(3), hide_index=True, width="stretch")
        metric = st.radio("Metric", ["WAPE", "WAPE_weekly", "MASE", "MAE", "RMSE"], horizontal=True)
        both = pd.concat([mval.assign(split="validation"), mtest.assign(split="test")])
        st.plotly_chart(px.bar(both, x="model", y=metric, color="split", barmode="group").update_layout(height=380),
                        width="stretch")
        sim = load_csv("policy_simulation.csv")
        if sim is not None:
            st.subheader("Inventory back-test (same replenishment policy, different forecasts)")
            st.dataframe(sim.round(3), hide_index=True, width="stretch")
        imp = load_csv("feature_importance.csv")
        if imp is not None:
            st.subheader(f"{info['model_name']} feature importance")
            st.plotly_chart(px.bar(imp.head(15).iloc[::-1], x="importance", y="feature", orientation="h").update_layout(height=420),
                            width="stretch")

# ------------------------------------------------------------------ Price & what-if
with tabs[4]:
    el = load_csv("price_elasticity.csv")
    if el is not None:
        st.subheader("Price elasticity by department")
        st.caption("Poisson GLM with series fixed effects. −1.5 means a 1% price cut raises expected demand by ~1.5%.")
        st.dataframe(el.round(3), hide_index=True, width="stretch")
    st.subheader("What-if: price change or promotion")
    sid_w = st.selectbox("SKU-location", fc.series_ids, key="wi_sid")
    c = st.columns(4)
    pct = c[0].slider("Price change (%)", -40, 20, -10)
    promo = c[1].checkbox("Run as promotion", value=False)
    start = c[2].slider("Start (days from now)", 0, 27, 0)
    dur = c[3].slider("Duration (days)", 1, 28, 14)
    w = fc.whatif(sid_w, pct, promo, start, dur, 28)
    c = st.columns(3)
    c[0].metric("Units", f"{w['scenario']['units']:.0f}", f"{w['delta_pct']['units']:+.1f}%")
    c[1].metric("Revenue", f"${w['scenario']['revenue']:.0f}", f"{w['delta_pct']['revenue']:+.1f}%")
    c[2].metric("Gross margin", f"${w['scenario']['gross_margin']:.0f}", f"{w['delta_pct']['gross_margin']:+.1f}%")
    d = pd.DataFrame(w["daily"])
    figw = go.Figure()
    figw.add_scatter(x=d["date"], y=d["baseline"], name="Baseline", line=dict(color="#888"))
    figw.add_scatter(x=d["date"], y=d["scenario"], name="Scenario", line=dict(color="#1F4E79", width=3))
    figw.update_layout(height=360, yaxis_title="units / day")
    st.plotly_chart(figw, width="stretch")
    st.caption(w["note"])

# ------------------------------------------------------------------ Anomalies & monitoring
with tabs[5]:
    dec = load_json("monitoring_summary.json")
    if dec:
        (st.error if dec["needs_retrain"] else st.success)(
            ("Retraining recommended: " + "; ".join(dec["reasons"])) if dec["needs_retrain"]
            else f"Model healthy: recent WAPE {dec['recent_wape']:.1%} vs validation {dec['baseline_wape']:.1%}, "
                 f"{dec['n_drifted_features']} drifted feature(s).")
    drift = load_csv("drift_features.csv")
    if drift is not None:
        st.subheader("Input drift (PSI: <0.1 stable · 0.1–0.2 moderate · >0.2 significant)")
        st.plotly_chart(px.bar(drift, x="feature", y="psi", color="drifted",
                               color_discrete_map={True: "#C62828", False: "#1F4E79"}).update_layout(height=320),
                        width="stretch")
    wk = load_csv("weekly_error.csv", ["week"])
    if wk is not None:
        st.plotly_chart(px.line(wk, x="week", y="wape", markers=True, title="Weekly WAPE on the test window")
                        .update_layout(height=300), width="stretch")
    an = load_csv("anomalies.csv", ["date"])
    if an is not None:
        st.subheader("Detected demand anomalies")
        sc = load_json("anomaly_metrics.json")
        if sc:
            st.caption(f"Back-test vs injected anomalies: precision {sc['precision']:.0%}, recall {sc['recall']:.0%} "
                       f"(spikes {sc['recall_spikes']:.0%}, drops {sc['recall_drops']:.0%}).")
        sid_a = st.selectbox("SKU-location", ["(all)"] + fc.series_ids, key="an_sid")
        t = an if sid_a == "(all)" else an[an["id"] == sid_a]
        st.dataframe(t.sort_values("date", ascending=False).head(200), hide_index=True, width="stretch")
