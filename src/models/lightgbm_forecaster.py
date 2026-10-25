"""Direct multi-day-ahead LightGBM forecaster (30 d, 90 d). Additional model; XGBoost is unchanged.

It deliberately REUSES :class:`XGBForecaster` for everything that is not the booster itself, so LightGBM sees exactly the same
inputs, targets, leakage rules and splits (``split_rows``) as XGBoost:
  * target = change in depth over h days (depth[t+h] - depth[t]); persistence == predicting 0;
  * features = ``FEATURE_COLUMNS`` (rows with NaN features are dropped by ``split_rows``, as for XGBoost, so no
    NaN-handling differs between the models; LightGBM itself would accept NaN);
  * early stopping + the split-conformal-style interval use the validation block only; evaluation / forecast_latest are inherited.

Hyper-parameters come from ``config/lightgbm_config.json`` (fixed, no search). Artifacts use the existing naming pattern:
``lgbm_{h}d.txt`` (LightGBM text model) + ``lgbm_{h}d_meta.json``. There is no scikit-learn stand-in: if ``lightgbm`` is not
installed an ImportError is raised (``pip install lightgbm``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, SEED, TARGET, get_logger
from src.models.xgboost_forecaster import XGBForecaster
from src.preprocessing.features import FEATURE_COLUMNS

log = get_logger("lgbm")
CONFIG_PATH = REPO_ROOT / "config" / "lightgbm_config.json"
BACKEND = "lightgbm"


def load_lgbm_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the hyper-parameter file -> {'params': {...}, 'early_stopping_rounds': int}."""
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())
    return {"params": dict(cfg["params"]), "early_stopping_rounds": int(cfg.get("early_stopping_rounds", 30))}


def _import_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required: pip install lightgbm") from None
    return lgb


class LGBMForecaster(XGBForecaster):
    """Direct h-day-ahead depth forecaster backed by ``lightgbm.LGBMRegressor``."""

    def __init__(self, horizon: int, seed: int = SEED, config: dict[str, Any] | None = None) -> None:
        _import_lightgbm()
        self.horizon, self.seed = horizon, seed
        self.model: Any = None
        self.backend = BACKEND
        self.config = config or load_lgbm_config()
        self.residual_std = float("nan")
        self.conformal_alpha = 0.10
        self.conformal_halfwidth = float("nan")
        self.n_train = 0
        self.best_iteration: int | None = None

    def _new_model(self) -> Any:
        return _import_lightgbm().LGBMRegressor(**self.config["params"], random_state=self.seed)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "LGBMForecaster":
        """Fit on train; early-stop on val; calibrate residual std / conformal half-width on val (as XGBForecaster)."""
        lgb = _import_lightgbm()
        if len(train) < 50 or len(val) < 5:
            raise ValueError(f"Too little data for {self.horizon}-day model: train={len(train)}, val={len(val)}")
        self.model = self._new_model()
        self.model.fit(train[FEATURE_COLUMNS], self._delta(train),
                       eval_set=[(val[FEATURE_COLUMNS], self._delta(val))], eval_metric="l2",
                       callbacks=[lgb.early_stopping(self.config["early_stopping_rounds"], verbose=False),
                                  lgb.log_evaluation(period=0)])
        self.best_iteration = int(self.model.best_iteration_ or self.model.n_estimators_)
        self.n_train = len(train)
        resid = val[f"target_{self.horizon}d"].to_numpy() - self.predict_level(val)
        self.residual_std = float(np.std(resid, ddof=1))
        self.conformal_halfwidth = self._conformal_q(np.abs(resid), self.conformal_alpha)
        log.info("%dd model (lightgbm): train=%d val=%d best_iter=%d residual_std=%.3f m conformal_halfwidth(%.0f%%)=%.3f m",
                 self.horizon, len(train), len(val), self.best_iteration, self.residual_std, 100 * (1 - self.conformal_alpha),
                 self.conformal_halfwidth)
        return self

    # -- persistence of the model ------------------------------------------------
    def _paths(self, models_dir: Path) -> tuple[Path, Path]:
        return models_dir / f"lgbm_{self.horizon}d.txt", models_dir / f"lgbm_{self.horizon}d_meta.json"

    def save(self, models_dir: Path) -> Path:
        import lightgbm as lgb
        models_dir.mkdir(parents=True, exist_ok=True)
        mp, metap = self._paths(models_dir)
        booster = self.model.booster_ if hasattr(self.model, "booster_") else self.model
        booster.save_model(str(mp))          # saves up to best_iteration when early stopping was used
        metap.write_text(json.dumps({"horizon": self.horizon, "backend": BACKEND, "lightgbm_version": lgb.__version__,
                                     "seed": self.seed, "residual_std": self.residual_std, "n_train": self.n_train,
                                     "best_iteration": self.best_iteration, "hyperparameters": self.config,
                                     "conformal_alpha": self.conformal_alpha, "conformal_halfwidth": self.conformal_halfwidth,
                                     "features": FEATURE_COLUMNS}, indent=2))
        return mp

    @classmethod
    def load(cls, models_dir: Path, horizon: int) -> "LGBMForecaster":
        lgb = _import_lightgbm()
        metap, mp = models_dir / f"lgbm_{horizon}d_meta.json", models_dir / f"lgbm_{horizon}d.txt"
        if not metap.exists() or not mp.exists():
            raise FileNotFoundError(f"{metap} / {mp} not found - run scripts/train_lightgbm.py first")
        meta = json.loads(metap.read_text())
        if meta.get("features") != FEATURE_COLUMNS:
            raise ValueError("feature list in the saved LightGBM metadata differs from FEATURE_COLUMNS - retrain")
        obj = cls.__new__(cls)
        obj.horizon, obj.seed, obj.backend = horizon, meta["seed"], BACKEND
        obj.config = meta.get("hyperparameters") or load_lgbm_config()
        obj.residual_std, obj.n_train = meta["residual_std"], meta["n_train"]
        obj.best_iteration = meta.get("best_iteration")
        obj.conformal_alpha = meta.get("conformal_alpha", 0.10)
        obj.conformal_halfwidth = meta.get("conformal_halfwidth", 1.96 * meta["residual_std"])
        obj.model = lgb.Booster(model_file=str(mp))   # Booster.predict(DataFrame) -> ndarray, same call as the sklearn wrapper
        return obj
