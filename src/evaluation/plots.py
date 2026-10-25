"""Matplotlib figures written to ``outputs/`` (headless backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import TARGET  # noqa: E402


def plot_lstm_actual_vs_pred(arrays: dict[str, Any], out: Path, n_sensors: int = 3) -> None:
    """Day-7-ahead actual vs LSTM vs persistence for a few sensors (test period)."""
    meta: pd.DataFrame = arrays["meta"]
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, 3 * n_sensors), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, meta["sensor_id"].unique()[:n_sensors]):
        m = (meta["sensor_id"] == sid).to_numpy()
        dates = meta.loc[m, "origin_date"] + pd.Timedelta(days=7)
        ax.plot(dates, arrays["true"][m, -1], label="actual", color="black")
        ax.plot(dates, arrays["pred"][m, -1], label="LSTM", color="tab:blue")
        ax.plot(dates, arrays["base"][m, -1], label="persistence", color="tab:gray", ls="--")
        ax.invert_yaxis(); ax.set_ylabel("depth (m bgl)\n(deeper = down)"); ax.set_title(f"{sid} - 7-day-ahead, test period")
        ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_sequence_actual_vs_pred(arrays: dict[str, Any], out: Path, horizon: int, label: str, n_sensors: int = 3) -> None:
    """Day-``horizon``-ahead actual vs a sequence model vs persistence for a few sensors (test period)."""
    meta: pd.DataFrame = arrays["meta"]
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, 3 * n_sensors), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, meta["sensor_id"].unique()[:n_sensors]):
        m = (meta["sensor_id"] == sid).to_numpy()
        dates = meta.loc[m, "origin_date"] + pd.Timedelta(days=horizon)
        ax.plot(dates, arrays["true"][m, -1], label="actual", color="black")
        ax.plot(dates, arrays["pred"][m, -1], label=label, color="tab:green")
        ax.plot(dates, arrays["base"][m, -1], label="persistence", color="tab:gray", ls="--")
        ax.invert_yaxis(); ax.set_ylabel("depth (m bgl)\n(deeper = down)"); ax.set_title(f"{sid} - {horizon}-day-ahead, test period")
        ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_xgb_test(test: pd.DataFrame, pred: np.ndarray, horizon: int, out: Path, n_sensors: int = 3) -> None:
    """Actual vs predicted depth at t+h for the test period."""
    t = test.assign(pred=pred, tdate=test["date"] + pd.Timedelta(days=horizon))
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, 3 * n_sensors))
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, t["sensor_id"].unique()[:n_sensors]):
        g = t[t["sensor_id"] == sid]
        ax.plot(g["tdate"], g[f"target_{horizon}d"], color="black", label="actual")
        ax.plot(g["tdate"], g["pred"], color="tab:orange", label=f"gradient boosting {horizon}d")
        ax.plot(g["tdate"], g[TARGET], color="tab:gray", ls="--", label="persistence")
        ax.invert_yaxis(); ax.set_ylabel("depth (m bgl)"); ax.set_title(f"{sid} - {horizon}-day-ahead, test period"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_anomalies(clean: pd.DataFrame, det: pd.DataFrame, out: Path, n_sensors: int = 3) -> None:
    """Depth series with flagged days highlighted (context: daily rainfall bars)."""
    flagged = det[det["anomaly_flag"]]
    sensors = (flagged["sensor_id"].value_counts().index.tolist() or clean["sensor_id"].unique().tolist())[:n_sensors]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 3 * len(sensors)))
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, sensors):
        g = clean[clean["sensor_id"] == sid]
        f = flagged[flagged["sensor_id"] == sid].merge(g[["date", TARGET]], on="date")
        ax.plot(g["date"], g[TARGET], color="tab:blue", lw=1)
        ax.scatter(f["date"], f[TARGET], color="tab:red", s=10, label="flagged (potential abnormal drawdown)")
        ax.invert_yaxis(); ax.set_ylabel("depth (m bgl)")
        ax2 = ax.twinx(); ax2.bar(g["date"], g["rainfall_mm"], color="tab:cyan", alpha=0.4, width=1.0)
        ax2.set_ylabel("rain (mm)"); ax.set_title(sid); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
