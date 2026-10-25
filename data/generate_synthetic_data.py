"""Synthetic DWLR (digital water level recorder) data generator.

Produces ``data/dwlr_data.csv`` and ``data/stations.csv`` in the common AquaSense schema plus the
synthetic-only label ``is_synthetic_anomalous_extraction``.

Physical story (deliberately simple, all parameters seeded):
  * regional rainfall with a SW monsoon (Jun-Sep) and a weaker NE monsoon (Oct-Dec);
  * recharge = delayed (12 d) and smoothed (tau 18 d) rainfall above a threshold;
  * depletion = baseline pumping + evapotranspiration term + slow long-term trend;
  * AR(1) natural noise + sensor noise, random missing values, multi-day sensor dropouts;
  * "synthetic anomalous extraction events": extra drawdown for 6-14 days, then slow recovery.
These events are NOT claims about real illegal extraction; they only give the detector a
controlled ground truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LABEL_COLUMN, SEED, TARGET, Paths, get_logger  # noqa: E402

log = get_logger("synthetic")


def _rain_probability(doy: np.ndarray) -> np.ndarray:
    sw = 0.55 * np.exp(-0.5 * ((doy - 215) / 38.0) ** 2)
    ne = 0.25 * np.exp(-0.5 * ((doy - 310) / 25.0) ** 2)
    return 0.03 + sw + ne


def _delayed_smoothed(rain: np.ndarray, delay: int = 12, tau: float = 18.0) -> np.ndarray:
    """Delay rainfall by ``delay`` days and apply exponential smoothing (soil/aquifer lag)."""
    a = np.exp(-1.0 / tau)
    out = np.zeros_like(rain, dtype=float)
    for t in range(len(rain)):
        src = rain[t - delay] if t - delay >= 0 else 0.0
        out[t] = (out[t - 1] * a if t > 0 else 0.0) + src * (1 - a)
    return out


def generate(n_days: int = 730, n_stations: int = 10, seed: int = SEED,
             start: str = "2022-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the synthetic dataset. Returns (observations, stations)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()

    regional_event = rng.random(n_days) < _rain_probability(doy)
    regional_rain = rng.gamma(0.8, 14.0, n_days) * regional_event

    frames, stations = [], []
    for k in range(n_stations):
        sid = f"DWLR_{k + 1:03d}"
        lat = float(rng.uniform(10.6, 12.0))
        lon = float(rng.uniform(78.3, 79.6))
        stations.append({"sensor_id": sid, "latitude": round(lat, 5), "longitude": round(lon, 5),
                         "name": f"Synthetic station {k + 1}", "source": "synthetic"})

        # --- climate (station-level variation around the regional signal) ---
        local = rng.gamma(0.5, 4.0, n_days) * (rng.random(n_days) < 0.5 * _rain_probability(doy))
        rain = np.clip(regional_rain * rng.uniform(0.85, 1.15) * rng.uniform(0.7, 1.3, n_days) + local, 0, None)
        temp = 29 + 4 * np.sin(2 * np.pi * (doy - 40) / 365) - 2.5 * (rain > 5) + rng.normal(0, 0.8, n_days)
        et = np.clip(4.2 + 0.45 * (temp - 29) - 0.02 * rain + rng.normal(0, 0.3, n_days), 1.0, 9.0)
        sm = np.empty(n_days)
        s = 0.25
        for t in range(n_days):
            s = s + 0.004 * rain[t] * (1 - s / 0.6) - 0.0025 * et[t] * (s / 0.6)
            s = float(np.clip(s, 0.05, 0.6))
            sm[t] = s

        # --- natural groundwater dynamics ---
        base = rng.uniform(7.0, 19.0)
        dep_rate = rng.uniform(0.010, 0.014)
        gain = rng.uniform(0.007, 0.010)
        trend = rng.uniform(0.1, 0.4) / 365.0
        recharge_src = _delayed_smoothed(rain)
        d = np.empty(n_days)
        cur = base
        ar = 0.0
        for t in range(n_days):
            recharge = gain * max(recharge_src[t] - 1.5, 0.0)
            depletion = dep_rate * (1 + 0.08 * (et[t] - 4.5)) + trend
            cur = max(1.0, cur + depletion - recharge)
            ar = 0.7 * ar + rng.normal(0, 0.02)
            d[t] = cur + ar

        # --- synthetic anomalous extraction events (stratified over time) ---
        label = np.zeros(n_days, dtype=int)
        extra = np.zeros(n_days)
        n_events = int(rng.integers(2, 4))
        chunk = n_days // 6
        for c in rng.choice(6, size=n_events, replace=False):
            length = int(rng.integers(6, 15))
            lo = c * chunk + 5
            hi = max(lo + 1, (c + 1) * chunk - length - 20)
            st = int(rng.integers(lo, hi))
            rate = rng.uniform(0.12, 0.30)
            label[st:st + length] = 1
            for t in range(st, min(n_days, st + length)):
                extra[t] = extra[t - 1] + rate * rng.uniform(0.8, 1.2)
            for t in range(st + length, n_days):
                if extra[t - 1] < 1e-3:
                    break
                extra[t] = extra[t - 1] * np.exp(-1 / 25.0)
        depth = d + extra + rng.normal(0, 0.015, n_days)

        # --- missingness ---
        depth = depth.copy()
        depth[rng.random(n_days) < 0.015] = np.nan
        for _ in range(int(rng.integers(1, 3))):  # sensor dropouts
            length = int(rng.integers(5, 21))
            st = int(rng.integers(0, n_days - length))
            depth[st:st + length] = np.nan
        climate = {"rainfall_mm": rain, "temperature_c": temp, "evapotranspiration_mm": et, "soil_moisture": sm}
        for name, arr in climate.items():
            arr = arr.copy()
            arr[rng.random(n_days) < 0.005] = np.nan
            climate[name] = arr

        frames.append(pd.DataFrame({
            "sensor_id": sid, "latitude": round(lat, 5), "longitude": round(lon, 5),
            "date": dates, TARGET: depth, **climate, LABEL_COLUMN: label}))

    df = pd.concat(frames, ignore_index=True)
    for c in (TARGET, "rainfall_mm", "temperature_c", "evapotranspiration_mm"):
        df[c] = df[c].round(3)
    df["soil_moisture"] = df["soil_moisture"].round(4)
    return df, pd.DataFrame(stations)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--stations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    paths = Paths("synthetic")
    df, st = generate(args.days, args.stations, args.seed)
    df.to_csv(paths.data_csv, index=False, date_format="%Y-%m-%d")
    st.to_csv(paths.stations_csv, index=False)
    log.info("Wrote %s (%d rows) and %s", paths.data_csv, len(df), paths.stations_csv)


if __name__ == "__main__":
    main()
