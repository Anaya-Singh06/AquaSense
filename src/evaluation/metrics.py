"""Forecast metrics (MAE, RMSE, NSE) and persistence-baseline comparison."""
from __future__ import annotations

from typing import Any

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (same unit as the data, metres)."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (metres)."""
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency: 1 - SSE / variance-around-mean. 1 is perfect, 0 equals 'predict the mean'."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def persistence_forecast(last_observed: np.ndarray, horizon: int | None = None) -> np.ndarray:
    """Persistence baseline: y_pred(t+h) = y(t) for every h.

    ``last_observed`` has shape (N,). If ``horizon`` is given the result has shape (N, horizon).
    """
    last = np.asarray(last_observed, float)
    return last if horizon is None else np.repeat(last[:, None], horizon, axis=1)


def _block(y: np.ndarray, p: np.ndarray, sensors: np.ndarray | None) -> dict[str, float]:
    out = {"mae": mae(y, p), "rmse": rmse(y, p), "nse_pooled": nse(y, p)}
    if sensors is not None:
        per = []
        for s in np.unique(sensors):
            m = sensors == s
            if m.sum() > 2:
                v = nse(y[m], p[m])
                if not np.isnan(v):
                    per.append(v)
        out["nse_per_sensor_mean"] = float(np.mean(per)) if per else float("nan")
    return out


def evaluate_against_persistence(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray,
                                 sensors: np.ndarray | None = None) -> dict[str, Any]:
    """Compare a model and the persistence baseline on identical samples.

    ``nse_pooled`` mixes all sensors and is inflated by between-sensor differences;
    ``nse_per_sensor_mean`` (NSE computed per sensor, then averaged) is stricter but unstable when the test window
    has little within-sensor variance. ``rmse_skill_vs_persistence`` = 1 - RMSE_model / RMSE_persistence
    (> 0 means the model has lower error than persistence).
    """
    y, p, b = (np.asarray(a, float).ravel() for a in (y_true, y_pred, y_persist))
    s = None if sensors is None else np.asarray(sensors).ravel()
    res = {"n": int(y.size), "model": _block(y, p, s), "persistence": _block(y, b, s)}
    key = "nse_per_sensor_mean" if s is not None else "nse_pooled"
    res["rmse_skill_vs_persistence"] = float(1.0 - res["model"]["rmse"] / res["persistence"]["rmse"]) if res["persistence"]["rmse"] > 0 else float("nan")
    res["model_beats_persistence"] = bool(res["model"][key] > res["persistence"][key])
    return res
