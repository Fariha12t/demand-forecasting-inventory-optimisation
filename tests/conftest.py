import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demand_forecasting.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """Small, fast configuration writing into a temp directory."""
    base = tmp_path_factory.mktemp("proj")
    cfg = copy.deepcopy(load_config())
    cfg["paths"] = {k: str(base / k) for k in ("raw_dir", "processed_dir", "outputs_dir", "models_dir")}
    cfg["data"]["synthetic"].update(n_items=4, n_stores=2, n_days=420)
    cfg["models"]["holt_winters"]["train_days"] = 200
    cfg["models"]["random_forest"]["params"].update(n_estimators=20, max_depth=8)
    cfg["models"]["lightgbm"]["params"].update(n_estimators=150)
    cfg["models"]["xgboost"]["params"].update(n_estimators=150)
    cfg["mlflow"]["enabled"] = False
    cfg["database"]["enabled"] = False
    return cfg


@pytest.fixture(scope="session")
def trained(tiny_cfg):
    from demand_forecasting.data.prepare import prepare
    from demand_forecasting.pipeline import run_train
    prepare(tiny_cfg)
    return run_train(tiny_cfg, use_mlflow=False)
