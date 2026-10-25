"""Forecast baselines: persistence and seasonal-naive.

persistence      y_hat(t+h) = y(t)                      (the most recent observation, for every horizon)
seasonal-naive   y_hat(t+h) = y_ref(t+h-365 d)          (the value one year before the TARGET date)
  * lag = 365 days for daily data (``SEASONAL_LAG_DAYS``); annual cycle is the dominant groundwater seasonality.
  * y_ref is the mean of REAL (non-imputed) observations in a centred 7-day window, so one missing day does not
    make the baseline undefined; if no real observation exists in that window the baseline is NaN and that sample is
    excluded for ALL models, so every model is always compared on identical samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET

SEASONAL_LAG_DAYS = 365
SEASONAL_WINDOW_DAYS = 7


def seasonal_reference(clean: pd.DataFrame, window: int = SEASONAL_WINDOW_DAYS) -> pd.Series:
    """Series indexed by (sensor_id, date): centred-window mean of real observations."""
    parts = []
    for sid, g in clean.groupby("sensor_id", sort=True):
        g = g.sort_values("date")
        y = g[TARGET].where(g["depth_imputed"] == 0) if "depth_imputed" in g else g[TARGET]
        r = y.rolling(window, center=True, min_periods=1).mean()
        parts.append(pd.Series(r.to_numpy(), index=pd.MultiIndex.from_arrays([np.repeat(sid, len(g)), g["date"].to_numpy()])))
    return pd.concat(parts)


def seasonal_naive(ref: pd.Series, sensor_ids: np.ndarray, target_dates: pd.Series | np.ndarray,
                   lag_days: int = SEASONAL_LAG_DAYS) -> np.ndarray:
    """Seasonal-naive forecast for each (sensor, target date); NaN when the year-earlier value is unavailable."""
    td = pd.to_datetime(pd.Series(target_dates)).to_numpy() - np.timedelta64(lag_days, "D")
    idx = pd.MultiIndex.from_arrays([np.asarray(sensor_ids), td])
    return ref.reindex(idx).to_numpy()


def persistence(last_observed: np.ndarray) -> np.ndarray:
    return np.asarray(last_observed, float)
