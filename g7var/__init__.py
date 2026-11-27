"""G7-VaR: one-day FX spot VaR from option-implied volatility (the code of the thesis).

`CFG` is config.yaml; set G7VAR_CONFIG to a second YAML file to override some of its keys.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["HS", "RiskMetrics", "GJR-GARCH", "FHS-GARCH", "FHS-IV", "QR-IM",
          "RandomForest", "XGBoost", "LightGBM", "CatBoost", "FFNN", "RNN", "LSTM"]
BASELINES = MODELS[:4]          # left out of a run with `baselines: false`


def _load() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    extra = os.environ.get("G7VAR_CONFIG")
    if extra:
        for key, value in yaml.safe_load((ROOT / extra).read_text(encoding="utf-8")).items():
            cfg[key] = {**cfg[key], **value} if isinstance(value, dict) and key in cfg else value
    return cfg


CFG = _load()


def run_settings(name: str) -> dict:
    """The settings of one run of config.yaml, with every default filled in."""
    run = CFG["runs"][name] or {}
    test = run.get("test", [CFG["sample"]["test_start"], CFG["sample"]["test_end"]])
    models = [m for m in MODELS if run.get("baselines", True) or m not in BASELINES]
    return {"name": name, "tenor": str(run.get("tenor", "1W")), "features": list(run.get("features", ["ATM", "RR25"])),
            "extra": run.get("extra", {}), "train_end": run.get("train_end", CFG["sample"]["train_end"]),
            "test_start": test[0], "test_end": test[1], "models": models}
