"""Controlled anomaly injection into a (real) groundwater series.

Label for any result produced here:  "Injected synthetic anomaly evaluation on real background data".
It is NOT real-world ground truth: the background series is real, the drawdown event is artificial and documented below.

Event shape (per station, seeded): extra depth ramps linearly from 0 to +magnitude over ``RAMP_DAYS`` days, then recovers
linearly to 0 over ``RECOVERY_DAYS`` days. magnitude = max(``MIN_MAGNITUDE_M``, ``K_STD`` x station std of depth in the TRAINING
period). Larger depth = deeper groundwater, so an injected drawdown is a positive depth anomaly.
Events are placed only after the validation period (never in data the detector was fitted on).
Natural anomalies already present in real data are unlabelled, so day-level precision is NOT reported - only event recall,
ROC-AUC against injected days, and the flag rate on days without injection (which contains natural events + false positives).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, SEED, TARGET
from src.models.anomaly_detector import AnomalyDetector
from src.preprocessing.features import build_features
from src.preprocessing.splits import SplitDates

RAMP_DAYS, RECOVERY_DAYS, MIN_MAGNITUDE_M, K_STD, EVENTS_PER_STATION, DETECT_WINDOW_DAYS = 10, 20, 0.2, 2.0, 3, 14


def inject_drawdowns(clean: pd.DataFrame, splits: SplitDates, seed: int = SEED) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return (modified copy with LABEL_COLUMN marking ramp days, event list)."""
    rng = np.random.default_rng(seed)
    out = clean.sort_values(["sensor_id", "date"]).reset_index(drop=True).copy()
    out[LABEL_COLUMN] = 0
    events: list[dict[str, Any]] = []
    span = RAMP_DAYS + RECOVERY_DAYS
    for sid, g in out.groupby("sensor_id", sort=True):
        train = g[g["date"] <= splits.train_end][TARGET].dropna()
        mag = max(MIN_MAGNITUDE_M, K_STD * float(train.std())) if len(train) > 30 else MIN_MAGNITUDE_M
        cand = g[(g["date"] > splits.val_end + pd.Timedelta(days=5)) & (g["date"] <= splits.end - pd.Timedelta(days=span + 5))]
        if len(cand) < span + 5:
            continue
        starts: list[int] = []
        for i in rng.permutation(len(cand)):
            pos = cand.index[i]
            if all(abs(pos - s) > span + 5 for s in starts):
                starts.append(pos)
            if len(starts) == EVENTS_PER_STATION:
                break
        for pos in starts:
            ramp = np.concatenate([np.linspace(0, 1, RAMP_DAYS + 1)[1:], np.linspace(1, 0, RECOVERY_DAYS + 1)[1:]]) * mag
            idx = np.arange(pos, pos + span)
            out.loc[idx, TARGET] = out.loc[idx, TARGET].to_numpy() + ramp
            out.loc[idx[:RAMP_DAYS], LABEL_COLUMN] = 1
            events.append({"sensor_id": sid, "start": out.loc[pos, "date"], "magnitude_m": round(mag, 3)})
    return out, events


def evaluate_injection(clean: pd.DataFrame, detector: AnomalyDetector, splits: SplitDates, seed: int = SEED) -> dict[str, Any]:
    """Run the detector on real series with injected drawdowns and measure detection."""
    from src.preprocessing.features import add_targets  # noqa: PLC0415
    inj, events = inject_drawdowns(clean, splits, seed)
    if not events:
        return {"label": "Injected synthetic anomaly evaluation on real background data", "note": "no station long enough for injection"}
    feats = build_features(inj)
    scored = detector.detect(feats)
    metrics = AnomalyDetector.evaluate_synthetic(scored, inj, splits.val_end)
    flagged = scored[scored["anomaly_flag"]]
    hit = 0
    for e in events:
        w = flagged[(flagged["sensor_id"] == e["sensor_id"]) & (flagged["date"] >= e["start"]) &
                    (flagged["date"] <= e["start"] + pd.Timedelta(days=DETECT_WINDOW_DAYS))]
        hit += int(len(w) > 0)
    after = scored[scored["date"] > splits.val_end].merge(inj[["sensor_id", "date", LABEL_COLUMN]], on=["sensor_id", "date"], how="left")
    neg = after[after[LABEL_COLUMN] == 0]
    return {"label": "Injected synthetic anomaly evaluation on real background data",
            "design": {"events_per_station": EVENTS_PER_STATION, "ramp_days": RAMP_DAYS, "recovery_days": RECOVERY_DAYS,
                       "magnitude": f"max({MIN_MAGNITUDE_M} m, {K_STD} x training-period std)", "detect_window_days": DETECT_WINDOW_DAYS},
            "n_events": len(events), "event_recall": hit / len(events), "events_detected": hit,
            "roc_auc_vs_injected_days": metrics.get("roc_auc"), "recall_injected_days": metrics.get("recall"),
            "flag_rate_on_non_injected_days": float(neg["anomaly_flag"].mean()) if len(neg) else None,
            "caveat": "non-injected days contain natural (unlabelled) events, so their flag rate is not a false-positive rate"}
