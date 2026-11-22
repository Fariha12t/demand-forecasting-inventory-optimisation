import numpy as np
import pandas as pd

from demand_forecasting.data.synthetic import generate
from demand_forecasting.features import CategoryEncoder, feature_columns, make_features, recursive_forecast


def _data(cfg):
    sales, _ = generate(n_items=3, n_stores=2, n_days=200, seed=1)
    enc = CategoryEncoder().fit(sales)
    return sales, enc


def test_lags_and_rolling_use_only_the_past(tiny_cfg):
    sales, enc = _data(tiny_cfg)
    f = make_features(sales, enc, tiny_cfg)
    one = f[f["id"] == f["id"].iloc[0]].reset_index(drop=True)
    assert one.loc[50, "lag_7"] == one.loc[43, "sales"]
    assert one.loc[60, "lag_28"] == one.loc[32, "sales"]
    assert one.loc[60, "roll_mean_7"] == np.mean(one.loc[53:59, "sales"])   # excludes today (index 60)


def test_changing_future_sales_does_not_change_past_features(tiny_cfg):
    sales, enc = _data(tiny_cfg)
    a = make_features(sales, enc, tiny_cfg)
    s2 = sales.copy()
    cutoff = sales["date"].max() - pd.Timedelta(days=20)
    s2.loc[s2["date"] > cutoff, "sales"] = 9999
    b = make_features(s2, enc, tiny_cfg)
    cols = feature_columns(tiny_cfg)
    past = a["date"] <= cutoff
    pd.testing.assert_frame_equal(a.loc[past, cols].reset_index(drop=True), b.loc[past, cols].reset_index(drop=True))


def test_recursive_forecast_never_sees_future_sales(tiny_cfg):
    sales, enc = _data(tiny_cfg)
    cutoff = sales["date"].max() - pd.Timedelta(days=14)
    hist, fut = sales[sales["date"] <= cutoff], sales[sales["date"] > cutoff]
    calls = []

    def predict(X):
        calls.append(len(X))
        return np.full(len(X), 5.0)

    out = recursive_forecast(predict, hist, fut.drop(columns=["sales"]), enc, tiny_cfg)
    assert len(out) == len(fut) and len(calls) == 14          # one batch per forecast day
    assert (out["y_pred"] == 5.0).all()


def test_unbalanced_panel_is_rejected(tiny_cfg):
    sales, enc = _data(tiny_cfg)
    import pytest
    with pytest.raises(ValueError):
        make_features(sales.iloc[:-1], enc, tiny_cfg)
