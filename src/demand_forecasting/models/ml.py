"""Machine-learning forecasters: Random Forest, LightGBM, XGBoost (global models over all series).

Hyper-parameter / n_estimators selection uses a *time-ordered* early-stopping block (the most
recent `es_days` of the training period) - never a random split.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

from ..features import CAT_COLS, training_frame


def _build(name: str, cfg: dict, n_estimators: int | None = None, early_stopping: bool = False):
    mc = cfg["models"][name]
    params = dict(mc["params"])
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        if early_stopping:
            params["early_stopping_rounds"] = mc.get("early_stopping_rounds", 50)
        return XGBRegressor(**params)
    raise ValueError(name)


def _fit(model, name, X, y, X_es=None, y_es=None, cfg=None):
    if name == "lightgbm":
        import lightgbm as lgb
        kw = {"categorical_feature": [c for c in CAT_COLS if c in X.columns]}
        if X_es is not None:
            kw["eval_set"] = [(X_es, y_es)]
            kw["callbacks"] = [lgb.early_stopping(cfg["models"][name].get("early_stopping_rounds", 50), verbose=False)]
        model.fit(X, y, **kw)
    elif name == "xgboost":
        if X_es is not None:
            model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
        else:
            model.fit(X, y, verbose=False)
    else:
        model.fit(X, y)
    return model


def best_iteration(model, name: str) -> int | None:
    if name == "lightgbm":
        return int(model.best_iteration_ or model.n_estimators)
    if name == "xgboost":
        bi = getattr(model, "best_iteration", None)
        return int(bi) + 1 if bi is not None else int(model.n_estimators)
    return None


def fit_with_early_stopping(name: str, feats: pd.DataFrame, columns: list[str], cfg: dict,
                            train_end, es_days: int = 28):
    """Fit on dates <= train_end - es_days, early-stop on the following es_days (time ordered)."""
    train_end = pd.Timestamp(train_end)
    es_start = train_end - pd.Timedelta(days=es_days)
    X, y, _ = training_frame(feats, columns, date_to=es_start)
    X_es, y_es, _ = training_frame(feats, columns, date_from=es_start + pd.Timedelta(days=1), date_to=train_end)
    if name == "random_forest":
        X, y, _ = training_frame(feats, columns, date_to=train_end)
        return _fit(_build(name, cfg), name, X, y), None
    model = _build(name, cfg, early_stopping=True)
    model = _fit(model, name, X, y, X_es, y_es, cfg)
    return model, best_iteration(model, name)


def refit(name: str, feats: pd.DataFrame, columns: list[str], cfg: dict, date_to, n_estimators: int | None):
    """Refit on all rows up to `date_to` with a fixed number of trees."""
    X, y, _ = training_frame(feats, columns, date_to=date_to)
    return _fit(_build(name, cfg, n_estimators=n_estimators if name != "random_forest" else None), name, X, y)


def feature_importance(model, columns: list[str]) -> pd.DataFrame:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance"])
    imp = np.asarray(imp, float)
    return pd.DataFrame({"feature": columns, "importance": imp / max(imp.sum(), 1e-12)}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
