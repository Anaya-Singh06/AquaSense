"""Loading, validation, regularisation and gap-filling of AquaSense daily data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLIMATE_COLUMNS, COMMON_COLUMNS, LABEL_COLUMN, TARGET, VALID_RANGES, get_logger

log = get_logger("preprocess")

SHORT_GAP_DAYS = 7    # gaps <= this: linear interpolation (imputed flag = 1)
LONG_GAP_DAYS = 30    # gaps in (7, 30]: linear interpolation, flagged as long-gap (flag = 2)
                      # gaps > 30: left missing; such rows are excluded from modelling


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Read a common-schema CSV and check required columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in COMMON_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def _gap_lengths(is_nan: np.ndarray) -> np.ndarray:
    """For every NaN position return the length of the NaN run it belongs to (0 elsewhere)."""
    out = np.zeros(len(is_nan), dtype=int)
    i, n = 0, len(is_nan)
    while i < n:
        if is_nan[i]:
            j = i
            while j < n and is_nan[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse dates, coerce numerics, drop duplicates, null-out out-of-range values, sort."""
    report: dict[str, Any] = {"rows_in": int(len(df))}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date", "sensor_id"])
    df["sensor_id"] = df["sensor_id"].astype(str)
    for c in ["latitude", "longitude", TARGET, *CLIMATE_COLUMNS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not df["latitude"].between(-90, 90).all() or not df["longitude"].between(-180, 180).all():
        raise ValueError("Invalid latitude/longitude values in input data")
    n_dup = int(df.duplicated(["sensor_id", "date"], keep="last").sum())
    df = df.drop_duplicates(["sensor_id", "date"], keep="last")
    out_of_range: dict[str, int] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        out_of_range[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    df = df.sort_values(["sensor_id", "date"]).reset_index(drop=True)
    report.update(bad_dates=bad_dates, duplicates_removed=n_dup, out_of_range_nullified=out_of_range)
    return df, report


def regularize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each sensor to a gap-free daily calendar (new rows are NaN)."""
    parts = []
    for sid, g in df.groupby("sensor_id", sort=True):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D", name="date")
        g = g.set_index("date").reindex(idx)
        g["sensor_id"] = sid
        g["latitude"] = g["latitude"].ffill().bfill()
        g["longitude"] = g["longitude"].ffill().bfill()
        if LABEL_COLUMN in g.columns:
            g[LABEL_COLUMN] = g[LABEL_COLUMN].fillna(0).astype(int)
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def missing_report(df: pd.DataFrame) -> dict[str, Any]:
    """Missing-data analysis: per-column fraction and per-sensor longest depth gap."""
    cols = [TARGET, *CLIMATE_COLUMNS]
    longest = {}
    for sid, g in df.groupby("sensor_id"):
        longest[sid] = int(_gap_lengths(g[TARGET].isna().to_numpy()).max(initial=0))
    return {"missing_fraction": {c: round(float(df[c].isna().mean()), 4) for c in cols},
            "longest_depth_gap_days": longest}


def fill_missing(df: pd.DataFrame, short_gap: int = SHORT_GAP_DAYS, long_gap: int = LONG_GAP_DAYS) -> pd.DataFrame:
    """Fill gaps and add ``depth_imputed`` (0 observed, 1 short-gap, 2 long-gap fill).

    * depth: linear interpolation *inside* gaps of length <= ``long_gap``; longer gaps and
      leading/trailing gaps stay NaN.
    * rainfall: missing -> 0 mm (conservative; no rain invented).
    * temperature / ET / soil moisture: linear interpolation then forward/back fill.
    NOTE: interpolation looks at both sides of a gap, which is acceptable for cleaning
    historical records but must not be used to fill the *forecast* period.
    """
    out = []
    for _, g in df.groupby("sensor_id", sort=True):
        g = g.sort_values("date").copy()
        isnan = g[TARGET].isna().to_numpy()
        gl = _gap_lengths(isnan)
        interp = g[TARGET].interpolate(method="linear", limit_area="inside")
        fill_ok = isnan & (gl <= long_gap) & interp.notna().to_numpy()
        flag = np.zeros(len(g), dtype=int)
        flag[fill_ok & (gl <= short_gap)] = 1
        flag[fill_ok & (gl > short_gap)] = 2
        g[TARGET] = np.where(fill_ok, interp, g[TARGET])
        g["depth_imputed"] = flag
        g["rainfall_mm"] = g["rainfall_mm"].fillna(0.0)
        for c in ("temperature_c", "evapotranspiration_mm", "soil_moisture"):
            g[c] = g[c].interpolate(method="linear", limit_area="inside").ffill().bfill()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full preprocessing pipeline. Returns (clean daily frame, report)."""
    df, report = validate_and_clean(df)
    df = regularize_daily(df)
    report["missing_before_fill"] = missing_report(df)
    df = fill_missing(df)
    report["depth_still_missing_fraction"] = round(float(df[TARGET].isna().mean()), 4)
    report["depth_imputed_counts"] = {str(k): int(v) for k, v in df["depth_imputed"].value_counts().sort_index().items()}
    report["rows_out"] = int(len(df))
    report["n_sensors"] = int(df["sensor_id"].nunique())
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    log.info("Preprocessed: %s", {k: report[k] for k in ("rows_in", "rows_out", "n_sensors", "date_range")})
    return df, report
