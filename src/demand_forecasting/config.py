"""Configuration loading and project paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("DF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load YAML config (default: configs/config.yaml, or $DF_CONFIG)."""
    path = Path(path or os.environ.get("DF_CONFIG", PROJECT_ROOT / "configs" / "config.yaml"))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Return an absolute, existing directory for paths.<key>."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
