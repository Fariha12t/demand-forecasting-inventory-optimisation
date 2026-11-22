"""Price analysis: elasticity estimation (Poisson GLM with series fixed effects) and what-if helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_elasticity(df: pd.DataFrame, group: str = "dept_id", price_ref_days: int = 56) -> pd.DataFrame:
    """Own-price elasticity per group.

    Model: sales ~ log(price_rel) + promo + C(dow) + C(month) + C(series), Poisson GLM (log link), so the
    coefficient on log(price_rel) is the elasticity: a 1% price change -> beta % change in expected demand.
    price_rel = price / trailing max price (so only discounts move it, not slow list-price drift).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df.sort_values(["id", "date"]).copy()
    ref = d.groupby("id")["sell_price"].transform(lambda s: s.rolling(price_ref_days, min_periods=1).max())
    d["log_price_rel"] = np.log(d["sell_price"] / ref)
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    rows = []
    for name, g in d.groupby(group):
        try:
            res = smf.glm("sales ~ log_price_rel + promo + C(dow) + C(month) + C(id)", data=g,
                          family=sm.families.Poisson()).fit(cov_type="HC1")
            b, se = res.params["log_price_rel"], res.bse["log_price_rel"]
            rows.append({group: name, "elasticity": b, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
                         "promo_uplift_pct": 100 * (np.exp(res.params["promo"]) - 1), "n_obs": int(res.nobs)})
        except Exception:
            rows.append({group: name, "elasticity": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                         "promo_uplift_pct": np.nan, "n_obs": len(g)})
    return pd.DataFrame(rows)
