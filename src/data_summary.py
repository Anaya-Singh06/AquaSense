"""Dataset-level summary (data source, station count, date range, observation count, missingness) for the API/dashboard."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import SOURCE_BADGES, SOURCE_LABELS, TARGET


def compute_data_summary(clean: pd.DataFrame, source: str) -> dict[str, Any]:
    """Summarise a preprocessed frame. 'Observed' = a real measurement (not interpolated, not missing)."""
    observed = clean[TARGET].notna() & (clean["depth_imputed"] == 0 if "depth_imputed" in clean else True)
    per = clean.assign(_obs=observed).groupby("sensor_id").agg(rows=("date", "size"), observed=("_obs", "sum"),
                                                              first=("date", "min"), last=("date", "max"))
    per["missing_fraction"] = 1 - per["observed"] / per["rows"]
    n_total = int(len(clean))
    return {
        "source": source, "source_label": SOURCE_LABELS.get(source, source), "source_badge": SOURCE_BADGES.get(source, source),
        "station_count": int(clean["sensor_id"].nunique()),
        "date_range": [str(clean["date"].min().date()), str(clean["date"].max().date())],
        "observation_count": int(observed.sum()), "calendar_rows": n_total,
        "missing_fraction": round(float(1 - observed.sum() / max(n_total, 1)), 4),
        "missing_fraction_by_station": {"median": round(float(per["missing_fraction"].median()), 4),
                                        "max": round(float(per["missing_fraction"].max()), 4)},
        "imputed_fraction": round(float((clean["depth_imputed"] > 0).mean()), 4) if "depth_imputed" in clean else None,
        "has_synthetic_labels": "is_synthetic_anomalous_extraction" in clean.columns,
        "units": "m below ground level (higher = deeper groundwater)",
        "note": ("Simulated stations - only evidence that the pipeline runs." if source == "synthetic" else
                 "Groundwater: USGS monitoring wells. Climate: NASA POWER gridded values attached by date and location (not measured at the well)."),
    }
