"""Unsupervised detection of *potential abnormal groundwater drawdown* (Isolation Forest + hydrological rules).

IMPORTANT: detections are statistical outliers. They are NOT proof of illegal extraction. Real data has no
verified anomaly labels, so precision/recall can only be computed on synthetic data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import LABEL_COLUMN, SEED, TARGET, get_logger
from src.preprocessing.features import FEATURE_COLUMNS

log = get_logger("anomaly")
ANOMALY_FEATURES = ["depth_z", "groundwater_change_1d", "groundwater_change_7d", "rainfall_mm", "rainfall_7d",
                    "rainfall_30d", "rolling_mean_7", "rolling_std_7", "rolling_std_30",
                    "seasonal_dev", "dev_from_30d_mean"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
OUTPUT_COLUMNS = ["sensor_id", "date", "anomaly_score", "anomaly_flag", "severity", "reason"]


class AnomalyDetector:
    """Isolation Forest on drawdown-related features + simple rule layer + severity grading."""

    def __init__(self, contamination: float = 0.03, n_estimators: int = 200, seed: int = SEED) -> None:
        self.contamination, self.n_estimators, self.seed = contamination, n_estimators, seed
        self.model: IsolationForest | None = None
        self.sensor_stats: dict[str, tuple[float, float]] = {}
        self.month_means: dict[tuple[str, int], float] = {}
        self.train_scores_sorted: np.ndarray = np.array([])
        self.thr7 = self.thr1 = 0.0

    # ---- feature construction (uses TRAINING statistics only) ----
    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.dropna(subset=FEATURE_COLUMNS).copy()
        mu = d["sensor_id"].map(lambda s: self.sensor_stats.get(s, (np.nan, np.nan))[0])
        sd = d["sensor_id"].map(lambda s: self.sensor_stats.get(s, (np.nan, np.nan))[1])
        d["depth_z"] = (d[TARGET] - mu) / sd
        mm = [self.month_means.get((s, m), np.nan) for s, m in zip(d["sensor_id"], d["month"])]
        d["seasonal_dev"] = d[TARGET] - np.asarray(mm)
        d["dev_from_30d_mean"] = d[TARGET] - d["rolling_mean_30"]
        return d.dropna(subset=ANOMALY_FEATURES)

    @staticmethod
    def _robust_upper(x: pd.Series, k: float) -> float:
        med = float(x.median())
        return med + k * 1.4826 * float((x - med).abs().median())

    def fit(self, train: pd.DataFrame) -> "AnomalyDetector":
        """Fit on training-period rows only (output of ``build_features``)."""
        tr = train.dropna(subset=FEATURE_COLUMNS)
        self.sensor_stats = {s: (g[TARGET].mean(), max(g[TARGET].std(), 1e-6)) for s, g in tr.groupby("sensor_id")}
        self.month_means = {(s, int(m)): v for (s, m), v in tr.groupby(["sensor_id", "month"])[TARGET].mean().items()}
        d = self._prep(tr)
        if len(d) < 100:
            raise ValueError("Not enough training rows for anomaly detector")
        # Robust thresholds (median + k * scaled MAD). Percentiles would be inflated because the unlabelled
        # training data itself contains abnormal events.
        self.thr7 = max(self._robust_upper(d["groundwater_change_7d"], 4.0), 0.10)
        self.thr1 = max(self._robust_upper(d["groundwater_change_1d"], 5.0), 0.05)
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination="auto", random_state=self.seed, n_jobs=-1)
        self.model.fit(d[ANOMALY_FEATURES])
        self.train_scores_sorted = np.sort(-self.model.score_samples(d[ANOMALY_FEATURES]))
        log.info("Isolation Forest fitted on %d rows; thr7=%.3f m, thr1=%.3f m", len(d), self.thr7, self.thr1)
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every row with complete features. Returns ``OUTPUT_COLUMNS`` (all rows, flagged or not).

        anomaly_score in [0, 1] = percentile of the raw isolation score within the training distribution.
        Flag = score >= 1 - contamination  OR  (>=1 rule hit AND score >= 0.90).
        """
        if self.model is None:
            raise RuntimeError("detector not fitted")
        d = self._prep(df)
        raw = -self.model.score_samples(d[ANOMALY_FEATURES])
        score = np.searchsorted(self.train_scores_sorted, raw, side="right") / len(self.train_scores_sorted)
        c7, c1 = d["groundwater_change_7d"].to_numpy(), d["groundwater_change_1d"].to_numpy()
        r7, r30 = d["rainfall_7d"].to_numpy(), d["rainfall_30d"].to_numpy()
        rule_dry = (c7 >= self.thr7) & (r7 < 10.0)
        rule_rain = (c7 >= self.thr7) & (r30 >= 50.0)
        rule_jump = c1 >= self.thr1
        n_rules = rule_dry.astype(int) + rule_rain.astype(int) + rule_jump.astype(int)
        if_flag = score >= 1.0 - self.contamination
        flag = if_flag | ((n_rules >= 1) & (score >= 0.90))
        level = (score >= 0.995).astype(int) + (n_rules >= 1) + ((n_rules >= 2) | (c7 >= 2 * self.thr7))
        sev = np.where(flag, np.array(SEVERITIES)[np.clip(level, 0, 3)], "NONE")
        reasons = []
        for i in range(len(d)):
            if not flag[i]:
                reasons.append("")
                continue
            parts = [f"unusual combination of drawdown features (isolation percentile {score[i]:.3f})"]
            if rule_dry[i]:
                parts.append(f"depth increased {c7[i]:+.2f} m in 7 d with only {r7[i]:.1f} mm rain")
            if rule_rain[i]:
                parts.append(f"groundwater deepened {c7[i]:+.2f} m in 7 d despite {r30[i]:.0f} mm rain in 30 d")
            if rule_jump[i]:
                parts.append(f"single-day depth change {c1[i]:+.2f} m")
            reasons.append("potential abnormal groundwater drawdown: " + "; ".join(parts))
        out = pd.DataFrame({"sensor_id": d["sensor_id"].to_numpy(), "date": d["date"].to_numpy(),
                            "anomaly_score": np.round(score, 4), "anomaly_flag": flag, "severity": sev, "reason": reasons})
        return out[OUTPUT_COLUMNS].reset_index(drop=True)

    # ---- evaluation ----
    @staticmethod
    def evaluate_synthetic(det: pd.DataFrame, labelled: pd.DataFrame, mask_after: pd.Timestamp | None = None) -> dict[str, Any]:
        """Precision/recall/F1/ROC-AUC against SYNTHETIC labels (only valid when LABEL_COLUMN exists)."""
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        m = det.merge(labelled[["sensor_id", "date", LABEL_COLUMN]], on=["sensor_id", "date"], how="left")
        if mask_after is not None:
            m = m[m["date"] > mask_after]
        y, f, s = m[LABEL_COLUMN].fillna(0).astype(int), m["anomaly_flag"].astype(int), m["anomaly_score"]
        if y.nunique() < 2:
            return {"note": "only one class present; metrics undefined", "n": int(len(m))}
        return {"n": int(len(m)), "n_positive": int(y.sum()), "precision": float(precision_score(y, f, zero_division=0)),
                "recall": float(recall_score(y, f, zero_division=0)), "f1": float(f1_score(y, f, zero_division=0)),
                "roc_auc": float(roc_auc_score(y, s)), "flagged": int(f.sum())}

    @staticmethod
    def summarize_unlabelled(det: pd.DataFrame) -> dict[str, Any]:
        """Descriptive summary for REAL data (no ground truth)."""
        fl = det[det["anomaly_flag"]]
        return {"note": "unsupervised detections without verified ground truth; not evidence of illegal extraction",
                "n_scored": int(len(det)), "n_flagged": int(len(fl)),
                "by_severity": {k: int(v) for k, v in fl["severity"].value_counts().items()},
                "by_sensor": {k: int(v) for k, v in fl["sensor_id"].value_counts().items()},
                "flagged_dates_sample": [str(pd.Timestamp(x).date()) for x in fl["date"].head(20)]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "AnomalyDetector":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/run_pipeline.py first")
        return joblib.load(path)
