"""Data-sufficiency analysis: does a station's record contain enough temporal information for forecasting?

All thresholds come from ``config/real_data_config.json`` (``sufficiency`` block) and were FIXED BEFORE any model
was trained. They must not be tuned to make stations pass. Verdicts:
  FAIL = data-quality problem (too short / too sparse / too many or too long gaps / too few test observations)
  WARN = usable but with a stated weakness (e.g. near-constant well, persistence trivially strong, few cycles)
  PASS = no flagged weakness
Stable wells are NOT excluded for hurting model skill - they are labelled ``stable`` (vs ``dynamic``) and reported.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET
from src.preprocessing.splits import make_split_dates

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_years_fail": 3.0, "min_years_warn": 6.0, "min_obs_fail": 730, "max_missing_fraction_fail": 0.50,
    "max_missing_fraction_warn": 0.25, "max_gap_days_fail": 365, "max_gap_days_warn": 120,
    "min_seasonal_cycles_warn": 5, "min_test_obs_fail": 60, "min_test_obs_warn": 150, "min_std_m_warn": 0.05,
    "persistence_trivial_nse_30d": 0.95, "stable_amplitude_m": 0.30, "stable_std_30d_change_m": 0.10,
}
PERSIST_HORIZON = 30
# Dataset-level conclusion rule (also fixed in advance): YES if >= 60 % PASS; PARTIALLY if >= 50 % PASS or WARN; else NO.
YES_PASS_SHARE, PARTIAL_USABLE_SHARE = 0.60, 0.50


def _longest_gap_days(dates: pd.DatetimeIndex) -> int:
    if len(dates) < 2:
        return 0
    return int(max(0, (np.diff(dates.values).astype("timedelta64[D]").astype(int) - 1).max()))


def _seasonal_amplitude(s: pd.Series) -> float:
    """Range of the monthly climatology of year-demeaned depth (removes slow trend). NaN if < 24 months of data."""
    if s.index.to_period("M").nunique() < 24:
        return float("nan")
    anom = s - s.groupby(s.index.year).transform("mean")
    clim = anom.groupby(anom.index.month).mean()
    return float(clim.max() - clim.min()) if len(clim) >= 6 else float("nan")


def _lag_pairs(s: pd.Series, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (y_t, y_{t+lag}) where both days were actually observed."""
    shifted = s.reindex(s.index + pd.Timedelta(days=lag))
    y1 = shifted.to_numpy()
    y0 = s.to_numpy()
    ok = ~np.isnan(y1)
    return y0[ok], y1[ok]


def station_sufficiency(sensor_id: str, obs: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp,
                        th: dict[str, float] | None = None) -> dict[str, Any]:
    """Sufficiency metrics + verdict for one station.

    ``obs`` has columns ``date`` and ``groundwater_depth_m_bgl`` containing ONLY real observations (NaN rows are ignored).
    ``start``/``end`` define the calendar used for the chronological 70/15/15 split (the test period is the last 15 %).
    """
    t = {**DEFAULT_THRESHOLDS, **(th or {})}
    d = obs.dropna(subset=[TARGET]).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    s = d.drop_duplicates("date").set_index("date")[TARGET].sort_index()
    res: dict[str, Any] = {"sensor_id": sensor_id, "n_obs": int(len(s))}
    if len(s) < 2:
        res.update(years=0.0, earliest=None, latest=None, expected_obs=0, missing_fraction=1.0, longest_gap_days=0,
                   mean_m=np.nan, std_m=np.nan, min_m=np.nan, max_m=np.nan, seasonal_amplitude_m=np.nan, n_seasonal_cycles=0,
                   test_obs=0, test_variance=np.nan, test_std=np.nan, std_30d_change_m=np.nan, persistence_nse_30d_test=np.nan,
                   persistence_trivial=False, dynamics="unknown", verdict="FAIL", reasons=["fewer than 2 observations"])
        return res
    years = (s.index[-1] - s.index[0]).days / 365.25
    expected = (s.index[-1] - s.index[0]).days + 1
    cal = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    test = s.iloc[0:0] if len(cal) < 20 else s[s.index > make_split_dates(cal).val_end]
    y0, y1 = _lag_pairs(s, PERSIST_HORIZON)
    std30 = float(np.std(y1 - y0)) if len(y0) > 20 else float("nan")
    ty0, ty1 = _lag_pairs(test, PERSIST_HORIZON) if len(test) else (np.array([]), np.array([]))
    denom = float(np.sum((ty1 - ty1.mean()) ** 2)) if len(ty1) > 20 else 0.0
    p_nse = float(1.0 - np.sum((ty1 - ty0) ** 2) / denom) if denom > 0 else float("nan")
    amp = _seasonal_amplitude(s)
    res.update(years=round(years, 2), earliest=str(s.index[0].date()), latest=str(s.index[-1].date()), expected_obs=int(expected),
               missing_fraction=round(1 - len(s) / expected, 4), longest_gap_days=_longest_gap_days(s.index),
               mean_m=float(s.mean()), std_m=float(s.std()), min_m=float(s.min()), max_m=float(s.max()),
               seasonal_amplitude_m=amp, n_seasonal_cycles=int(np.floor(years)), test_obs=int(len(test)),
               test_variance=float(test.var()) if len(test) > 1 else float("nan"),
               test_std=float(test.std()) if len(test) > 1 else float("nan"), std_30d_change_m=std30,
               persistence_nse_30d_test=p_nse, persistence_trivial=bool(np.isfinite(p_nse) and p_nse >= t["persistence_trivial_nse_30d"]))
    stable = (np.isfinite(amp) and amp < t["stable_amplitude_m"]) and (np.isfinite(std30) and std30 < t["stable_std_30d_change_m"])
    res["dynamics"] = "stable" if stable else "dynamic"
    fail, warn = [], []
    if years < t["min_years_fail"]: fail.append(f"record {years:.1f} y < {t['min_years_fail']}")
    if len(s) < t["min_obs_fail"]: fail.append(f"{len(s)} obs < {int(t['min_obs_fail'])}")
    if res["missing_fraction"] > t["max_missing_fraction_fail"]: fail.append(f"missing {res['missing_fraction']:.0%} > {t['max_missing_fraction_fail']:.0%}")
    if res["longest_gap_days"] > t["max_gap_days_fail"]: fail.append(f"gap {res['longest_gap_days']} d > {int(t['max_gap_days_fail'])}")
    if res["test_obs"] < t["min_test_obs_fail"]: fail.append(f"test obs {res['test_obs']} < {int(t['min_test_obs_fail'])}")
    if not fail:
        if years < t["min_years_warn"]: warn.append(f"record {years:.1f} y < {t['min_years_warn']}")
        if res["missing_fraction"] > t["max_missing_fraction_warn"]: warn.append(f"missing {res['missing_fraction']:.0%}")
        if res["longest_gap_days"] > t["max_gap_days_warn"]: warn.append(f"gap {res['longest_gap_days']} d")
        if res["n_seasonal_cycles"] < t["min_seasonal_cycles_warn"]: warn.append(f"{res['n_seasonal_cycles']} seasonal cycles")
        if res["test_obs"] < t["min_test_obs_warn"]: warn.append(f"test obs {res['test_obs']}")
        if res["std_m"] < t["min_std_m_warn"]: warn.append(f"near-constant (std {res['std_m']:.3f} m)")
        if res["persistence_trivial"]: warn.append(f"persistence trivially strong (30 d NSE {p_nse:.2f})")
    res["verdict"] = "FAIL" if fail else ("WARN" if warn else "PASS")
    res["reasons"] = fail or warn
    return res


def sufficiency_table(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp,
                      th: dict[str, float] | None = None) -> pd.DataFrame:
    """Per-station table for a common-schema frame (depth NaN = not observed)."""
    rows = [station_sufficiency(str(sid), g[["date", TARGET]], start, end, th) for sid, g in df.groupby("sensor_id", sort=True)]
    return pd.DataFrame(rows)


def overall_conclusion(table: pd.DataFrame) -> dict[str, Any]:
    """Dataset-level answer to 'is there enough temporal information?' using the fixed rule above."""
    n = len(table)
    if n == 0:
        return {"answer": "NO", "n": 0, "detail": "no stations"}
    c = table["verdict"].value_counts().to_dict()
    p, w, f = c.get("PASS", 0), c.get("WARN", 0), c.get("FAIL", 0)
    ans = "YES" if p / n >= YES_PASS_SHARE else ("PARTIALLY" if (p + w) / n >= PARTIAL_USABLE_SHARE else "NO")
    return {"answer": ans, "n": n, "PASS": p, "WARN": w, "FAIL": f,
            "stable": int((table["dynamics"] == "stable").sum()), "dynamic": int((table["dynamics"] == "dynamic").sum()),
            "persistence_trivial": int(table["persistence_trivial"].sum()),
            "rule": f"YES if PASS share >= {YES_PASS_SHARE:.0%}; PARTIALLY if (PASS+WARN) share >= {PARTIAL_USABLE_SHARE:.0%}; else NO"}
