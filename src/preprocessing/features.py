"""Leakage-safe feature engineering. Every feature at date t uses data up to and including t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET

GW_LAGS = (1, 3, 7, 14, 30)
FEATURE_COLUMNS: list[str] = [
    TARGET,
    *[f"lag_{k}" for k in GW_LAGS],
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30", "rolling_std_7", "rolling_std_30",
    "groundwater_change_1d", "groundwater_change_7d",
    "rainfall_1d", "rainfall_7d", "rainfall_14d", "rainfall_30d",
    "temperature_7d", "evapotranspiration_7d", "soil_moisture_7d",
    "month", "day_of_year", "sin_day_of_year", "cos_day_of_year",
]
HORIZONS = (7, 30, 90)
# Longest groundwater look-back used by any feature: lag_30 needs t-30 .. t  ->  31 consecutive observed days.
GW_LOOKBACK_DAYS = max(GW_LAGS) + 1
INPUT_OK_COL = "model_input_ok"   # NOT a model feature; row-eligibility mask (see add_input_mask)


def _features_one_sensor(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    y = g[TARGET]
    for k in GW_LAGS:
        g[f"lag_{k}"] = y.shift(k)
    for w in (7, 14, 30):
        g[f"rolling_mean_{w}"] = y.rolling(w, min_periods=w).mean()
    for w in (7, 30):
        g[f"rolling_std_{w}"] = y.rolling(w, min_periods=w).std()
    g["groundwater_change_1d"] = y - y.shift(1)
    g["groundwater_change_7d"] = y - y.shift(7)
    g["rainfall_1d"] = g["rainfall_mm"]
    for w in (7, 14, 30):
        g[f"rainfall_{w}d"] = g["rainfall_mm"].rolling(w, min_periods=w).sum()
    g["temperature_7d"] = g["temperature_c"].rolling(7, min_periods=7).mean()
    g["evapotranspiration_7d"] = g["evapotranspiration_mm"].rolling(7, min_periods=7).mean()
    g["soil_moisture_7d"] = g["soil_moisture"].rolling(7, min_periods=7).mean()
    g["month"] = g["date"].dt.month
    g["day_of_year"] = g["date"].dt.dayofyear
    ang = 2 * np.pi * g["day_of_year"] / 365.25
    g["sin_day_of_year"] = np.sin(ang)
    g["cos_day_of_year"] = np.cos(ang)
    return g


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all model input features (no target columns)."""
    out = pd.concat([_features_one_sensor(g) for _, g in df.groupby("sensor_id", sort=True)], ignore_index=True)
    return add_input_mask(out)


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add ``target_{h}d`` (depth h days ahead) and ``target_{h}d_observed`` (1 if that day was a real observation).

    Targets deliberately look into the future; they must never be used as inputs.
    Requires a gap-free daily calendar per sensor (see ``regularize_daily``).
    """
    parts = []
    for _, g in df.groupby("sensor_id", sort=True):
        g = g.sort_values("date").copy()
        observed = (g["depth_imputed"] == 0).astype(float) if "depth_imputed" in g else pd.Series(1.0, index=g.index)
        observed = observed.where(g[TARGET].notna())
        for h in horizons:
            g[f"target_{h}d"] = g[TARGET].shift(-h)
            g[f"target_{h}d_observed"] = observed.shift(-h)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_input_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``model_input_ok`` = 1 iff every groundwater value in [t-30, t] is a REAL observation.

    Interpolated (``depth_imputed`` != 0) or missing values are looked at from both sides of a gap, so they would
    carry look-ahead information. Any row whose groundwater features (value, lags up to 30 d, rolling stats, changes)
    touch one is flagged 0 and must be excluded from every train/val/test split and every rolling-origin fold.
    """
    parts = []
    for _, g in df.groupby("sensor_id", sort=True):
        g = g.sort_values("date").copy()
        real = (g["depth_imputed"] == 0) if "depth_imputed" in g else pd.Series(True, index=g.index)
        bad = (~real | g[TARGET].isna()).astype(float)
        g[INPUT_OK_COL] = (bad.rolling(GW_LOOKBACK_DAYS, min_periods=GW_LOOKBACK_DAYS).sum() == 0).astype(int)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
