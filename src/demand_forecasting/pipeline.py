"""End-to-end reproducible pipeline.

    python -m demand_forecasting.pipeline prepare-data   # build processed datasets + time-based split
    python -m demand_forecasting.pipeline train          # features, models, evaluation, inventory, artefacts
    python -m demand_forecasting.pipeline monitor        # drift / performance check + retrain decision
    python -m demand_forecasting.pipeline retrain        # retrain when monitoring says so (or --force)
    python -m demand_forecasting.pipeline report         # markdown model comparison report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .anomaly import anomaly_scores, detect_anomalies
from .config import load_config, resolve
from .data.prepare import prepare, read_sales, read_split
from .data.synthetic import TRUE_ELASTICITY
from .evaluation import metrics_table, wape
from .features import BASE_COLS, CategoryEncoder, feature_columns, make_features, recursive_forecast
from .inventory import params_from_config, recommend_all, simulate_policy
from .models import baseline, ml, statistical
from .monitoring import feature_drift, retrain_decision, weekly_error
from .price import estimate_elasticity
from .serving import build_future_frame

log = logging.getLogger("pipeline")
ML_MODELS = {"random_forest": "RandomForest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
MONITOR_FEATURES = ["lag_7", "roll_mean_7", "roll_mean_28", "roll_std_28", "sell_price", "price_rel", "promo"]


def _stat_models(cfg: dict) -> dict:
    mc = cfg["models"]
    fns = {"SeasonalNaive": lambda tr, fu: baseline.seasonal_naive(tr, fu)}
    if mc["moving_average"]["enabled"]:
        w = mc["moving_average"]["window"]
        fns[f"MovingAvg_{w}"] = lambda tr, fu: baseline.moving_average(tr, fu, w)
    if mc["seasonal_average"]["enabled"]:
        wk = mc["seasonal_average"]["weeks"]
        fns[f"SeasonalAvg_{wk}w"] = lambda tr, fu: baseline.seasonal_average(tr, fu, wk)
    if mc["holt_winters"]["enabled"]:
        fns["HoltWinters"] = lambda tr, fu: statistical.holt_winters(tr, fu, mc["holt_winters"]["train_days"])
    if mc["arima"]["enabled"]:
        a = mc["arima"]
        fns["SARIMA"] = lambda tr, fu: statistical.sarima(tr, fu, a["order"], a["seasonal_order"], a["train_days"])
    return fns


def _read_r_forecasts(cfg: dict, fold: str, start, end) -> pd.DataFrame | None:
    path = resolve(cfg, "outputs_dir") / "r" / "r_forecasts.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path, parse_dates=["date"])
    r = r[(r["fold"] == fold) & (r["date"] >= pd.Timestamp(start)) & (r["date"] <= pd.Timestamp(end))]
    return r[["id", "date", "y_pred", "model"]] if len(r) else None


def run_train(cfg: dict, use_mlflow: bool | None = None) -> dict:
    t0 = time.time()
    out, models_dir = resolve(cfg, "outputs_dir"), resolve(cfg, "models_dir")
    sales, split = read_sales(cfg), read_split(cfg)
    enc = CategoryEncoder().fit(sales)
    cols = feature_columns(cfg)
    feats = make_features(sales, enc, cfg)
    last_date = sales["date"].max()
    folds = {"val": (split["train_end"], split["val_start"], split["val_end"]),
             "test": (split["val_end"], split["test_start"], split["test_end"])}
    forecasts: dict[str, list[pd.DataFrame]] = {"val": [], "test": []}
    best_iters: dict[str, int | None] = {}
    timings: dict[str, float] = {}
    lstm_on = cfg["models"]["lstm"]["enabled"]
    if lstm_on:
        from .models.lstm import LSTMForecaster, torch_available
        lstm_on = torch_available()
        if not lstm_on:
            log.warning("LSTM enabled but PyTorch is not installed - skipping (pip install -r requirements-lstm.txt)")

    for fold, (origin, a, b) in folds.items():
        train = sales[sales["date"] <= pd.Timestamp(origin)]
        future = sales[(sales["date"] >= pd.Timestamp(a)) & (sales["date"] <= pd.Timestamp(b))]
        fut_in = future.drop(columns=["sales"])           # forecasters never see the target
        log.info("Fold %s: train <= %s, forecast %s..%s", fold, origin, a, b)
        for name, fn in _stat_models(cfg).items():
            t = time.time()
            forecasts[fold].append(fn(train, fut_in).assign(model=name))
            timings[f"{name}:{fold}"] = time.time() - t
        for key, label in ML_MODELS.items():
            if not cfg["models"][key]["enabled"]:
                continue
            t = time.time()
            if fold == "val":
                model, best_iters[key] = ml.fit_with_early_stopping(key, feats, cols, cfg, origin)
            else:
                model = ml.refit(key, feats, cols, cfg, origin, best_iters.get(key))
            pred = recursive_forecast(model.predict, train, fut_in, enc, cfg)
            forecasts[fold].append(pred.assign(model=label))
            timings[f"{label}:{fold}"] = time.time() - t
        if lstm_on:
            t = time.time()
            lm = LSTMForecaster(cfg).fit(train)
            forecasts[fold].append(lm.predict(train, fut_in).assign(model="LSTM"))
            timings[f"LSTM:{fold}"] = time.time() - t
        r = _read_r_forecasts(cfg, fold, a, b)
        if r is not None:
            forecasts[fold].append(r)
            log.info("Merged R forecasts: %s", sorted(r["model"].unique()))

    truth = sales[["id", "date", "sales"]].rename(columns={"sales": "y_true"})
    fc = {f: pd.concat(parts, ignore_index=True).merge(truth, on=["id", "date"], how="left") for f, parts in forecasts.items()}
    fc_ids = {f: set(df.groupby("model")["id"].nunique().pipe(lambda s: s[s == s.max()]).index) for f, df in fc.items()}
    for f in fc:                                           # drop partial models (e.g. R run on a subset of series)
        fc[f] = fc[f][fc[f]["model"].isin(fc_ids[f])]
    metrics = {}
    for fold, origin in (("val", split["train_end"]), ("test", split["val_end"])):
        metrics[fold] = metrics_table(fc[fold], sales[sales["date"] <= pd.Timestamp(origin)])
        fc[fold].to_csv(out / f"forecasts_{fold}.csv", index=False)
        metrics[fold].to_csv(out / f"metrics_{fold}.csv", index=False)
        log.info("\n%s metrics\n%s", fold.upper(), metrics[fold].round(3).to_string(index=False))

    # ---- choose the serving model (best validation WAPE among refittable global ML models)
    cand = [ML_MODELS[k] for k in ML_MODELS if cfg["models"][k]["enabled"]]
    mv = metrics["val"].set_index("model")
    serving_label = min(cand, key=lambda m: mv.loc[m, "WAPE"])
    serving_key = next(k for k, v in ML_MODELS.items() if v == serving_label)
    log.info("Serving model: %s", serving_label)
    final = ml.refit(serving_key, feats, cols, cfg, last_date, best_iters.get(serving_key))
    imp = ml.feature_importance(final, cols)
    imp.to_csv(out / "feature_importance.csv", index=False)

    # forecast error sigma per series from the validation fold (no look-ahead)
    sigma_by_model = {}
    for m, g in fc["val"].groupby("model"):
        sigma_by_model[m] = g.assign(e=(g["y_true"] - g["y_pred"]) ** 2).groupby("id")["e"].mean().pow(0.5)
    sigma = sigma_by_model[serving_label]

    # ---- true future forecast + inventory recommendations
    fut_frame = build_future_frame(sales, 28)
    future_fc = recursive_forecast(final.predict, sales, fut_frame, enc, cfg)
    future_fc = future_fc.merge(fut_frame[["id", "date", "sell_price", "promo"]], on=["id", "date"])
    future_fc.to_csv(out / "future_forecast.csv", index=False)
    pdir = resolve(cfg, "processed_dir")
    master, snapshot = pd.read_csv(pdir / "item_master.csv"), pd.read_csv(pdir / "inventory_snapshot.csv")
    prices = future_fc.groupby("id")["sell_price"].first()
    recs = recommend_all(future_fc, sigma, snapshot, master, prices, cfg)
    recs.to_csv(out / "recommendations.csv", index=False)

    # ---- inventory policy back-test on the test window
    sim = _policy_backtest(cfg, sales, split, fc["test"], sigma_by_model, master, prices)
    sim.to_csv(out / "policy_simulation.csv", index=False)

    # ---- price analysis, anomalies
    elas = estimate_elasticity(sales, "dept_id", cfg["features"]["price_ref_days"])
    if cfg["data"]["source"] == "synthetic":
        elas["true_elasticity"] = elas["dept_id"].map(TRUE_ELASTICITY)
    elas.to_csv(out / "price_elasticity.csv", index=False)
    an = detect_anomalies(sales, cfg["anomaly"]["window_weeks"], cfg["anomaly"]["threshold"], cfg["anomaly"]["overdispersion"])
    an[an["is_anomaly"]].drop(columns=["dev", "mad", "explained_by_event_or_promo"]).to_csv(out / "anomalies.csv", index=False)
    an_scores = anomaly_scores(an)
    (out / "anomaly_metrics.json").write_text(json.dumps(an_scores, indent=2))

    # ---- monitoring (drift) on the test window vs training period
    summary = _monitor_summary(cfg, feats, split, fc, metrics, serving_label, 0.0)
    (out / "monitoring_summary.json").write_text(json.dumps(summary["decision"], indent=2))
    summary["drift"].to_csv(out / "drift_features.csv", index=False)
    summary["weekly"].to_csv(out / "weekly_error.csv", index=False)

    # ---- artefacts for the API / dashboard
    hist_cols = BASE_COLS
    tail = sales[sales["date"] > last_date - pd.Timedelta(days=120)][hist_cols]
    artefacts = {
        "version": __version__, "model_name": serving_label, "model": final, "features": cols, "encoder": enc,
        "config": {k: cfg[k] for k in ("features", "inventory", "monitoring", "project")},
        "history": tail, "sigma": sigma.to_dict(), "item_master": master, "snapshot": snapshot,
        "trained_through": str(last_date.date()), "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics_val": mv.loc[serving_label].to_dict(),
        "metrics_test": metrics["test"].set_index("model").loc[serving_label].to_dict(),
    }
    joblib.dump(artefacts, models_dir / "artifacts.joblib")
    card = {k: artefacts[k] for k in ("version", "model_name", "trained_through", "created_at", "metrics_val", "metrics_test")}
    (models_dir / "model_card.json").write_text(json.dumps(card, indent=2, default=str))

    result = {"serving_model": serving_label, "metrics": metrics, "recommendations": recs, "policy": sim,
              "elasticity": elas, "anomaly_scores": an_scores, "decision": summary["decision"],
              "timings": timings, "best_iters": best_iters, "runtime_s": time.time() - t0}
    _track_mlflow(cfg, result, final, serving_key, use_mlflow)
    _maybe_sync_db(cfg, sales, fc, recs)
    log.info("Done in %.0fs", time.time() - t0)
    return result


def _policy_backtest(cfg, sales, split, fc_test, sigma_by_model, master, prices) -> pd.DataFrame:
    m = master.set_index("item_id")
    last_train = pd.Timestamp(split["val_end"])
    recent = sales[(sales["date"] <= last_train) & (sales["date"] > last_train - pd.Timedelta(days=28))]
    mean_daily = recent.groupby("id")["sales"].mean()
    ref_price = recent.groupby("id")["sell_price"].max()
    rows = []
    models = list(fc_test["model"].unique())
    for model in models:
        agg = {"demanded": 0.0, "filled": 0.0, "stockout_days": 0, "days": 0, "inv_value": 0.0, "avg_on_hand": 0.0,
               "hold": 0.0, "lost": 0.0}
        for sid, g in fc_test[fc_test["model"] == fc_test["model"].iloc[0]].groupby("id"):
            g = g.sort_values("date")
            item = sid.rsplit("_", 2)[0]
            lead = int(m.loc[item, "lead_time_days"]) if item in m.index else None
            pack = int(m.loc[item, "pack_size"]) if item in m.index else 1
            unit_cost = float(ref_price[sid]) * cfg["inventory"]["unit_cost_ratio"]
            prm = params_from_config(cfg, lead_time_days=lead, pack_size=pack, unit_cost=unit_cost)
            demand = g["y_true"].to_numpy(float)
            gm = fc_test[(fc_test["model"] == model) & (fc_test["id"] == sid)].sort_values("date")
            f, sg = gm["y_pred"].to_numpy(float), float(sigma_by_model[model].get(sid, 1.0))
            init = mean_daily[sid] * (prm.lead_time_days + prm.review_period_days)
            r = simulate_policy(demand, f, sg, prm, init)
            for k in ("demanded", "filled", "stockout_days", "days"):
                agg[k] += r[k]
            agg["avg_on_hand"] += r["avg_on_hand"]
            agg["inv_value"] += r["avg_on_hand"] * unit_cost
            agg["hold"] += r["avg_on_hand"] * unit_cost * cfg["inventory"]["holding_cost_rate"] * r["days"] / 365.0
            agg["lost"] += (r["demanded"] - r["filled"]) * (float(ref_price[sid]) - unit_cost)   # lost gross margin
        rows.append({"model": model, "fill_rate": agg["filled"] / max(agg["demanded"], 1e-9),
                     "stockout_day_rate": agg["stockout_days"] / max(agg["days"], 1),
                     "avg_units_on_hand": agg["avg_on_hand"], "avg_inventory_value": agg["inv_value"],
                     "holding_cost": agg["hold"], "lost_margin": agg["lost"], "total_cost": agg["hold"] + agg["lost"]})
    return pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)


def _monitor_summary(cfg, feats, split, fc, metrics, model_label, age_days) -> dict:
    drift = feature_drift(feats, split["train_end"], split["test_start"], MONITOR_FEATURES, cfg["monitoring"]["psi_threshold"])
    sub = fc["test"][fc["test"]["model"] == model_label]
    weekly = weekly_error(sub)
    decision = retrain_decision(drift, float(metrics["val"].set_index("model").loc[model_label, "WAPE"]),
                                float(metrics["test"].set_index("model").loc[model_label, "WAPE"]), age_days, cfg)
    return {"drift": drift, "weekly": weekly, "decision": decision}


def _track_mlflow(cfg, result, final_model, key, use_mlflow) -> None:
    mc = cfg["mlflow"]
    if not (mc["enabled"] if use_mlflow is None else use_mlflow):
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", mc["tracking_uri"]))
        mlflow.set_experiment(mc["experiment"])
        with mlflow.start_run(run_name="pipeline"):
            mlflow.log_params({"data_source": cfg["data"]["source"], "serving_model": result["serving_model"],
                               "val_days": cfg["split"]["val_days"], "test_days": cfg["split"]["test_days"]})
            for fold in ("val", "test"):
                for _, row in result["metrics"][fold].iterrows():
                    with mlflow.start_run(run_name=f"{row['model']}-{fold}", nested=True):
                        mlflow.log_param("model", row["model"])
                        mlflow.log_param("fold", fold)
                        mlflow.log_metrics({k: float(row[k]) for k in ("MAE", "RMSE", "WAPE", "MASE", "WAPE_weekly", "Bias")})
            mlflow.log_artifact(str(resolve(cfg, "outputs_dir") / "metrics_test.csv"))
            mlflow.log_artifact(str(resolve(cfg, "outputs_dir") / "metrics_val.csv"))
            info = mlflow.sklearn.log_model(final_model, "model", registered_model_name=mc["registered_model"],
                                            serialization_format="cloudpickle")
            log.info("MLflow: logged + registered %s (%s)", mc["registered_model"], getattr(info, "model_uri", ""))
    except Exception as exc:  # tracking must never break the pipeline
        log.warning("MLflow tracking skipped: %s", exc)


def _maybe_sync_db(cfg, sales, fc, recs) -> None:
    if not (cfg["database"]["enabled"] or os.environ.get("DATABASE_URL")):
        return
    try:
        from .db import sync_outputs
        out = resolve(cfg, "outputs_dir")
        sync_outputs({"sales": sales.drop(columns=["anomaly_injected"], errors="ignore"),
                      "forecasts": pd.concat([fc["test"].assign(fold="test"), pd.read_csv(out / "future_forecast.csv").assign(model="serving", fold="future")], ignore_index=True),
                      "recommendations": recs})
    except Exception as exc:
        log.warning("Database sync skipped: %s", exc)


def run_monitor(cfg: dict) -> dict:
    """Compare the production model against data that arrived after training; decide on retraining."""
    out, models_dir = resolve(cfg, "outputs_dir"), resolve(cfg, "models_dir")
    art = joblib.load(models_dir / "artifacts.joblib")
    sales, split = read_sales(cfg), read_split(cfg)
    through = pd.Timestamp(art["trained_through"])
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(art["created_at"])).total_seconds() / 86400
    new = sales[sales["date"] > through]
    enc, cols = art["encoder"], art["features"]
    feats = make_features(sales, enc, cfg)
    base_wape = float(art["metrics_val"]["WAPE"])
    if new["date"].nunique() >= 7:
        horizon_end = new["date"].min() + pd.Timedelta(days=27)
        fut = new[new["date"] <= horizon_end]
        pred = recursive_forecast(art["model"].predict, sales[sales["date"] <= through], fut.drop(columns=["sales"]), enc, cfg)
        cmp_ = pred.merge(fut[["id", "date", "sales"]], on=["id", "date"])
        recent = float(wape(cmp_["sales"], cmp_["y_pred"]))
        drift = feature_drift(feats, through, new["date"].min(), MONITOR_FEATURES, cfg["monitoring"]["psi_threshold"])
        src = f"{new['date'].nunique()} new days"
    else:
        recent = float(art["metrics_test"]["WAPE"])
        drift = feature_drift(feats, split["train_end"], split["test_start"], MONITOR_FEATURES, cfg["monitoring"]["psi_threshold"])
        src = "no new data since training - using the held-out test window"
    decision = retrain_decision(drift, base_wape, recent, age, cfg) | {"evaluated_on": src}
    (out / "monitoring_summary.json").write_text(json.dumps(decision, indent=2))
    drift.to_csv(out / "drift_features.csv", index=False)
    log.info("Monitoring: %s", decision)
    return decision


def run_retrain(cfg: dict, force: bool = False) -> dict | None:
    decision = run_monitor(cfg)
    if force or decision["needs_retrain"]:
        log.info("Retraining (%s)", "forced" if force else "; ".join(decision["reasons"]))
        return run_train(cfg)
    log.info("Model is healthy - no retraining needed.")
    return None


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare-data", "train", "monitor", "retrain", "report", "run-all"])
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true", help="retrain even if monitoring is healthy")
    p.add_argument("--no-mlflow", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.command == "prepare-data":
        prepare(cfg)
    elif a.command == "train":
        run_train(cfg, use_mlflow=False if a.no_mlflow else None)
    elif a.command == "monitor":
        run_monitor(cfg)
    elif a.command == "retrain":
        run_retrain(cfg, a.force)
    elif a.command == "report":
        from .report import write_report
        write_report(cfg)
    else:
        prepare(cfg)
        run_train(cfg, use_mlflow=False if a.no_mlflow else None)
        from .report import write_report
        write_report(cfg)


if __name__ == "__main__":
    main()
