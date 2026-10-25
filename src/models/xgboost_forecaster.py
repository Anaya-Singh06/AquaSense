"""Direct multi-day-ahead gradient-boosting forecaster (XGBoost) with approximate prediction intervals.

Design (honest description):
  * One model per horizon (30 d, 90 d). The model predicts the **change** in depth over h days
    (depth[t+h] - depth[t]); persistence therefore corresponds to predicting 0.
  * It forecasts an END POINT only. The daily curve shown in the dashboard is a LINEAR INTERPOLATION
    between today's depth and the predicted end point - a PROTOTYPE SIMPLIFICATION, not a
    multi-horizon probabilistic model.
  * Prediction interval (v2): EMPIRICAL, split-conformal-style. The half-width is the ceil((n+1)(1-alpha))-th smallest
    ABSOLUTE validation residual (alpha = 0.10 -> nominal 90 %), pooled over stations. It needs no normality
    assumption, but the finite-sample coverage guarantee of conformal prediction assumes exchangeable data, which
    daily groundwater series are not (autocorrelation, seasonality, drift). Treat it as an approximate empirical
    interval and read the measured coverage (reported in metrics.json / rolling-origin evaluation).
    Early stopping also looks at the validation set, so calibration is slightly optimistic.
    The old +/-1.96 * residual-std band is still reported (``residual_std``) for comparison only.
  * The band along the interpolated curve is widened with sqrt(day/h) - a heuristic.

If ``xgboost`` is not importable the module raises, unless the environment variable
``AQUASENSE_ALLOW_SKLEARN_FALLBACK=1`` is set (developer-only stand-in using sklearn
HistGradientBoostingRegressor, recorded in the metadata as ``backend``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import SEED, TARGET, get_logger
from src.evaluation.metrics import evaluate_against_persistence, persistence_forecast
from src.preprocessing.features import FEATURE_COLUMNS, INPUT_OK_COL
from src.preprocessing.splits import SplitDates

log = get_logger("xgb")
FALLBACK_BACKEND = "sklearn_HistGradientBoosting_STANDIN"


def _backend() -> str:
    try:
        import xgboost  # noqa: F401
        return "xgboost"
    except ImportError:
        if os.environ.get("AQUASENSE_ALLOW_SKLEARN_FALLBACK") == "1":
            log.warning("xgboost NOT installed - using sklearn stand-in (developer mode only)")
            return FALLBACK_BACKEND
        raise ImportError("xgboost is required: pip install xgboost") from None


def split_rows(df: pd.DataFrame, horizon: int, splits: SplitDates, stats: dict | None = None) -> dict[str, pd.DataFrame]:
    """Chronological split by TARGET date (= date + horizon).

    Leakage rule (identical for train / val / test): a row is kept only if its groundwater input history
    (value + 30-day look-back) AND its target day are real observations - no interpolated value is ever used.
    If ``stats`` is given it is filled with candidate / retained / dropped counts per split.
    """
    tcol = f"target_{horizon}d"
    cand = df.dropna(subset=[*FEATURE_COLUMNS, tcol]).copy()
    if INPUT_OK_COL not in cand or f"{tcol}_observed" not in cand:
        raise ValueError("frame lacks model_input_ok / target observed flags: run add_input_mask() and add_targets()")
    ok = (cand[INPUT_OK_COL] == 1) & (cand[f"{tcol}_observed"] == 1)
    tdate = cand["date"] + pd.Timedelta(days=horizon)
    masks = {"train": tdate <= splits.train_end,
             "val": (tdate > splits.train_end) & (tdate <= splits.val_end),
             "test": tdate > splits.val_end}
    if stats is not None:
        stats.update({k: {"candidate": int(m.sum()), "retained": int((m & ok).sum()), "dropped_imputed": int((m & ~ok).sum())}
                      for k, m in masks.items()})
    return {k: cand[m & ok] for k, m in masks.items()}


class XGBForecaster:
    """Direct h-day-ahead depth forecaster."""

    def __init__(self, horizon: int, seed: int = SEED) -> None:
        self.horizon, self.seed = horizon, seed
        self.model: Any = None
        self.backend = _backend()
        self.residual_std: float = float("nan")
        self.conformal_alpha: float = 0.10
        self.conformal_halfwidth: float = float("nan")
        self.n_train = 0

    # -- training ---------------------------------------------------------------
    def _new_model(self) -> Any:
        if self.backend == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                                    tree_method="hist", random_state=self.seed, n_jobs=-1,
                                    early_stopping_rounds=30)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                             min_samples_leaf=20, random_state=self.seed)

    def _delta(self, d: pd.DataFrame) -> np.ndarray:
        return (d[f"target_{self.horizon}d"] - d[TARGET]).to_numpy()

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "XGBForecaster":
        """Fit on train; early-stop on val (xgboost); set residual std from validation."""
        if len(train) < 50 or len(val) < 5:
            raise ValueError(f"Too little data for {self.horizon}-day model: train={len(train)}, val={len(val)}")
        self.model = self._new_model()
        Xt, yt = train[FEATURE_COLUMNS], self._delta(train)
        if self.backend == "xgboost":
            self.model.fit(Xt, yt, eval_set=[(val[FEATURE_COLUMNS], self._delta(val))], verbose=False)
        else:
            self.model.fit(Xt, yt)
        self.n_train = len(train)
        resid = val[f"target_{self.horizon}d"].to_numpy() - self.predict_level(val)
        self.residual_std = float(np.std(resid, ddof=1))
        self.conformal_halfwidth = self._conformal_q(np.abs(resid), self.conformal_alpha)
        log.info("%dd model (%s): train=%d val=%d residual_std=%.3f m conformal_halfwidth(%.0f%%)=%.3f m", self.horizon, self.backend,
                 len(train), len(val), self.residual_std, 100 * (1 - self.conformal_alpha), self.conformal_halfwidth)
        return self

    @staticmethod
    def _conformal_q(abs_resid: np.ndarray, alpha: float) -> float:
        """Finite-sample conformal quantile of absolute calibration residuals."""
        r = np.sort(np.asarray(abs_resid, float))
        r = r[np.isfinite(r)]
        if len(r) == 0:
            return float("nan")
        k = int(np.ceil((len(r) + 1) * (1 - alpha)))
        return float(r[min(k, len(r)) - 1])

    # -- inference --------------------------------------------------------------
    def predict_interval(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(prediction, lower, upper) of the depth at t+h with the empirical conformal-style half-width."""
        p = self.predict_level(df)
        return p, p - self.conformal_halfwidth, p + self.conformal_halfwidth

    def predict_level(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted depth (m bgl) at t+h for each row."""
        return df[TARGET].to_numpy() + self.model.predict(df[FEATURE_COLUMNS])

    def evaluate(self, test: pd.DataFrame) -> dict[str, Any]:
        """Metrics vs persistence on test rows whose target day was truly observed."""
        t = test[test[f"target_{self.horizon}d_observed"] == 1]
        y = t[f"target_{self.horizon}d"].to_numpy()
        res = evaluate_against_persistence(y, self.predict_level(t), persistence_forecast(t[TARGET].to_numpy()),
                                           t["sensor_id"].to_numpy())
        res["horizon_days"] = self.horizon
        err = np.abs(y - self.predict_level(t))
        res["approx_interval_halfwidth_m"] = round(self.conformal_halfwidth, 3)
        res["interval_nominal_coverage"] = 1 - self.conformal_alpha
        res["interval_coverage_on_test"] = float(np.mean(err <= self.conformal_halfwidth)) if len(t) else None
        res["legacy_gaussian_halfwidth_m"] = round(1.96 * self.residual_std, 3)
        res["legacy_gaussian_coverage_on_test"] = float(np.mean(err <= 1.96 * self.residual_std)) if len(t) else None
        res["backend"] = self.backend
        res["n_origin_days_per_sensor"] = int(t.groupby("sensor_id").size().median()) if len(t) else 0
        return res

    def forecast_latest(self, latest_row: pd.DataFrame) -> dict[str, Any]:
        """Forecast from a single-row frame (latest date of one sensor)."""
        if latest_row[FEATURE_COLUMNS].isna().any(axis=None):
            raise ValueError("latest row has NaN features; cannot forecast")
        now = float(latest_row[TARGET].iloc[0])
        end = float(self.predict_level(latest_row)[0])
        d0 = pd.Timestamp(latest_row["date"].iloc[0])
        days = np.arange(1, self.horizon + 1)
        mid = now + (end - now) * days / self.horizon          # linear interpolation (prototype)
        hw = self.conformal_halfwidth if np.isfinite(self.conformal_halfwidth) else 1.96 * self.residual_std
        half = hw * np.sqrt(days / self.horizon)  # heuristic widening
        return {"horizon_days": self.horizon, "method": "direct end-point forecast + linear interpolation (prototype)",
                "interval_type": f"approximate empirical {100 * (1 - self.conformal_alpha):.0f}% interval (split-conformal-style, validation absolute residuals)",
                "endpoint_depth_m_bgl": round(end, 3), "endpoint_lower": round(end - hw, 3),
                "endpoint_upper": round(end + hw, 3),
                "trajectory": [{"date": str((d0 + pd.Timedelta(days=int(k))).date()), "depth_m_bgl": round(float(m), 3),
                                "lower": round(float(m - h), 3), "upper": round(float(m + h), 3)}
                               for k, m, h in zip(days, mid, half)]}

    # -- persistence of the model ------------------------------------------------
    def _paths(self, models_dir: Path) -> tuple[Path, Path]:
        ext = "json" if self.backend == "xgboost" else "joblib"
        return models_dir / f"xgb_{self.horizon}d.{ext}", models_dir / f"xgb_{self.horizon}d_meta.json"

    def save(self, models_dir: Path) -> Path:
        models_dir.mkdir(parents=True, exist_ok=True)
        mp, metap = self._paths(models_dir)
        if self.backend == "xgboost":
            self.model.save_model(str(mp))
        else:
            joblib.dump(self.model, mp)
        metap.write_text(json.dumps({"horizon": self.horizon, "backend": self.backend, "seed": self.seed,
                                     "residual_std": self.residual_std, "n_train": self.n_train,
                                     "conformal_alpha": self.conformal_alpha, "conformal_halfwidth": self.conformal_halfwidth,
                                     "features": FEATURE_COLUMNS}, indent=2))
        return mp

    @classmethod
    def load(cls, models_dir: Path, horizon: int) -> "XGBForecaster":
        metap = models_dir / f"xgb_{horizon}d_meta.json"
        if not metap.exists():
            raise FileNotFoundError(f"{metap} not found - run scripts/run_pipeline.py first")
        meta = json.loads(metap.read_text())
        obj = cls.__new__(cls)
        obj.horizon, obj.seed, obj.backend = horizon, meta["seed"], meta["backend"]
        obj.residual_std, obj.n_train = meta["residual_std"], meta["n_train"]
        obj.conformal_alpha = meta.get("conformal_alpha", 0.10)
        obj.conformal_halfwidth = meta.get("conformal_halfwidth", 1.96 * meta["residual_std"])  # older artifacts
        if obj.backend == "xgboost":
            import xgboost as xgb
            obj.model = xgb.XGBRegressor()
            obj.model.load_model(str(models_dir / f"xgb_{horizon}d.json"))
        else:
            obj.model = joblib.load(models_dir / f"xgb_{horizon}d.joblib")
        return obj
