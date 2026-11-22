import numpy as np
import pytest

from demand_forecasting.evaluation import bias, mae, mase, mase_scale, rmse, wape


def test_basic_metrics():
    y, p = [10, 20, 30], [12, 18, 33]
    assert mae(y, p) == pytest.approx(7 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert wape(y, p) == pytest.approx(7 / 60)
    assert bias(y, p) == pytest.approx(3 / 60)


def test_mase_uses_seasonal_naive_scale():
    train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], float)  # weekly diff = 7 everywhere
    scale = mase_scale(train, season=7)
    assert scale == 7.0
    assert mase([10, 10], [17, 3], scale) == pytest.approx(1.0)
    assert mase([5, 5], [5, 5], scale) == 0.0


def test_wape_handles_zero_demand():
    assert wape([0, 0], [1, 1]) > 0
