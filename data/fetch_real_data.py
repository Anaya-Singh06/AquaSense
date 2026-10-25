"""Real-data adapter: USGS groundwater (modern Water Data OGC APIs) + NASA POWER climate -> AquaSense common schema.

Outputs
  data/real_data.csv, data/real_stations.csv         (common schema; see src/config.py)
  outputs/real/station_selection.json                (discovered / rejected / selected + reasons)
  data/raw/                                          (cached raw API responses; reused unless --refresh)

Pipeline: discover candidates (time-series-metadata + monitoring-locations) -> cheap metadata filter ->
download daily values (cached) -> data-sufficiency verdict per station (thresholds from the config, fixed in advance) ->
download NASA POWER (cached) -> unit-normalise -> merge on date -> validate -> save.

Usage
  python data/fetch_real_data.py                 # use cache where present, download what is missing
  python data/fetch_real_data.py --refresh       # ignore the cache and re-download everything
  python data/fetch_real_data.py --offline       # cache only; never touch the network (errors if something is missing)
  python data/fetch_real_data.py --max-stations 30

Credentials: optional env var USGS_API_KEY (higher rate limits). Never commit keys.

NASA POWER is a gridded model/satellite product. It is merged to each well by DATE and LOCATION; it is not measured
at the well. Units are read from the response metadata and converted/validated explicitly (see ``parse_nasa_power``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import usgs_ogc as usgs  # noqa: E402
from data.usgs_ogc import FetchError, FT_TO_M  # noqa: E402,F401  (re-exported for callers/tests)
from src.config import COMMON_COLUMNS, REPO_ROOT, TARGET, Paths, get_logger  # noqa: E402

log = get_logger("fetch_real")
NASA_FILL = -999.0
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "config" / "real_data_config.json"

# Accepted POWER unit strings -> multiplier into AquaSense units.
# We never assume: if the response declares a unit that is not listed here we raise.
_NASA_UNITS: dict[str, dict[str, float]] = {
    "rainfall_mm": {"mm/day": 1.0, "mm day-1": 1.0, "mm/d": 1.0, "kg m-2 s-1": 86400.0},   # 1 kg/m2 of water = 1 mm
    "temperature_c": {"c": 1.0, "degc": 1.0, "deg c": 1.0, "°c": 1.0},
    "evapotranspiration_mm": {"mm/day": 1.0, "mm day-1": 1.0, "mm/d": 1.0, "kg m-2 s-1": 86400.0},
    "soil_moisture": {"1": 1.0, "fraction": 1.0, "dimensionless": 1.0, "0-1": 1.0, "-": 1.0},
}


# ----------------------------------------------------------------------------- config / caching
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def resolve_end_date(cfg: dict[str, Any], today: pd.Timestamp | None = None) -> str:
    """'auto' -> today minus ``end_date_lag_days`` (POWER/USGS latest days may be incomplete)."""
    if cfg.get("end_date", "auto") != "auto":
        return str(cfg["end_date"])
    t = (today or pd.Timestamp.today()).normalize()
    return str((t - pd.Timedelta(days=int(cfg.get("end_date_lag_days", 10)))).date())


class RawCache:
    """Tiny JSON file cache under data/raw/. ``refresh`` ignores existing files; ``offline`` forbids downloads."""

    def __init__(self, root: Path = RAW_DIR, refresh: bool = False, offline: bool = False) -> None:
        self.root, self.refresh, self.offline = Path(root), refresh, offline
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.downloads = 0

    def get_json(self, name: str, producer) -> Any:
        f = self.root / name
        if f.exists() and not self.refresh:
            self.hits += 1
            return json.loads(f.read_text())
        if self.offline:
            raise FetchError(f"--offline given but {f.name} is not in the cache ({self.root})")
        data = producer()
        f.write_text(json.dumps(data))
        self.downloads += 1
        return data


# ----------------------------------------------------------------------------- NASA POWER
def parse_nasa_power(payload: dict[str, Any], mapping: dict[str, str]) -> pd.DataFrame:
    """Parse a POWER daily point response into AquaSense columns with EXPLICIT unit handling.

    * fill value -999 -> NaN
    * units are read from ``payload['parameters'][NAME]['units']``; known units are converted
      (e.g. kg m-2 s-1 -> mm/day), unknown units raise ``FetchError``; a missing unit entry is logged (cannot verify).
    * soil moisture is clipped to the 0-1 wetness fraction.
    """
    try:
        params = payload["properties"]["parameter"]
    except KeyError as exc:
        raise FetchError(f"Unexpected NASA POWER response (no properties.parameter): {str(payload)[:200]}") from exc
    declared = payload.get("parameters", {}) or {}
    frames: dict[str, pd.Series] = {}
    for nasa_name, our_name in mapping.items():
        if nasa_name not in params:
            raise FetchError(f"NASA POWER response lacks parameter {nasa_name}; got {list(params)}")
        s = pd.Series(params[nasa_name], dtype=float)
        s[s <= NASA_FILL] = np.nan
        unit = (declared.get(nasa_name, {}) or {}).get("units")
        if unit is None:
            log.warning("NASA POWER %s: no units in response metadata - cannot verify, assuming %s", nasa_name, our_name)
        else:
            table = _NASA_UNITS.get(our_name, {})
            factor = table.get(str(unit).strip().lower())
            if factor is None:
                raise FetchError(f"NASA POWER {nasa_name} has unexpected units '{unit}' for {our_name}; refusing to guess.")
            s = s * factor
        frames[our_name] = s
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()].rename_axis("date").reset_index()
    if "soil_moisture" in df:
        df["soil_moisture"] = df["soil_moisture"].clip(0, 1)
    return df


def fetch_nasa_power(lat: float, lon: float, start: str, end: str, cfg: dict[str, Any], cache: RawCache) -> pd.DataFrame:
    """NASA POWER daily point data for [start, end], in ``chunk_years``-sized requests (each cached)."""
    import requests
    n = cfg["nasa_power"]
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    cur = s
    while cur <= e:
        nxt = min(cur + pd.DateOffset(years=int(n.get("chunk_years", 10))) - pd.Timedelta(days=1), e)
        name = f"nasa_power_{lat:.3f}_{lon:.3f}_{cur:%Y%m%d}_{nxt:%Y%m%d}.json"

        def producer(cur=cur, nxt=nxt):
            r = requests.get(n["url"], params={"parameters": ",".join(n["parameters"]), "community": n["community"],
                                               "longitude": round(lon, 4), "latitude": round(lat, 4),
                                               "start": f"{cur:%Y%m%d}", "end": f"{nxt:%Y%m%d}", "format": "JSON"}, timeout=120,
                             headers={"User-Agent": "AquaSense-college-project/2.0"})
            if r.status_code != 200:
                raise FetchError(f"NASA POWER HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        parts.append(parse_nasa_power(cache.get_json(name, producer), n["parameters"]))
        cur = nxt + pd.Timedelta(days=1)
    return pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------- merge
def build_common_schema(sensor_id: str, lat: float, lon: float, depth: pd.DataFrame, climate: pd.DataFrame,
                        start: str, end: str) -> pd.DataFrame:
    """Merge groundwater + climate on a continuous daily calendar into the AquaSense common schema.

    Depth is NaN where no observation exists (preprocessing decides what to fill; nothing is invented here).
    Climate is location-based gridded data attached by date - it was not recorded by the well.
    """
    cal = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    out = cal.merge(depth[["date", TARGET]], on="date", how="left").merge(climate, on="date", how="left")
    out.insert(0, "sensor_id", sensor_id)
    out.insert(1, "latitude", round(float(lat), 5))
    out.insert(2, "longitude", round(float(lon), 5))
    for c in COMMON_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[COMMON_COLUMNS]


# ----------------------------------------------------------------------------- USGS discovery + download
def discover_candidates(cfg: dict[str, Any], cache: RawCache) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Programmatic station discovery. Returns (kept candidates, rejected with reasons, n_discovered)."""
    u, d = cfg["usgs"], cfg["discovery"]
    hdr = usgs.api_headers(u.get("api_key_env", "USGS_API_KEY"))
    ts_url = usgs.collection_url(u["base_url"], u["api_version"], "time-series-metadata")
    ml_url = usgs.collection_url(u["base_url"], u["api_version"], "monitoring-locations")
    key = hashlib.md5(json.dumps([u["parameter_code"], d["bbox"]]).encode()).hexdigest()[:8]
    ts_feats = cache.get_json(f"usgs_ts_metadata_{key}.json", lambda: usgs.fetch_all_items(
        ts_url, {"parameter_code": u["parameter_code"], "bbox": ",".join(map(str, d["bbox"])), "limit": u["page_limit"]},
        headers=hdr, timeout=u["timeout_s"]))
    meta = usgs.parse_time_series_metadata(ts_feats, u["parameter_code"], u.get("statistic_id"))
    n_discovered = int(meta["monitoring_location_id"].nunique())
    log.info("USGS time-series-metadata: %d locations with parameter %s", n_discovered, u["parameter_code"])
    # coordinates for the longest-record locations only (keeps the number of requests small)
    meta["record_years"] = (meta["end"] - meta["begin"]).dt.days / 365.25
    top = meta.sort_values("record_years", ascending=False).drop_duplicates("monitoring_location_id").head(d["max_candidates"] * 3)
    ids = top["monitoring_location_id"].tolist()
    locs = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        feats = cache.get_json(f"usgs_locations_{hashlib.md5(','.join(chunk).encode()).hexdigest()[:10]}.json", lambda c=chunk: usgs.fetch_all_items(
            ml_url, {**usgs.monitoring_location_filter(c), "limit": u["page_limit"]}, headers=hdr, timeout=u["timeout_s"]))
        locs.append(usgs.parse_monitoring_locations(feats))
    loc_df = pd.concat(locs, ignore_index=True) if locs else pd.DataFrame(columns=["monitoring_location_id", "latitude", "longitude", "name", "site_type_code", "state"])
    merged = meta.merge(loc_df, on="monitoring_location_id", how="left")
    # only groundwater sites: site_type_code 'GW' (or one of the GW sub-types, which start with 'GW')
    is_gw = merged["site_type_code"].fillna("").str.upper().str.startswith("GW") | merged["site_type_code"].isna()
    not_gw = merged[~is_gw].assign(reason="not a groundwater monitoring location")
    kept, rejected = usgs.filter_candidates(merged[is_gw], d["min_record_years"], d["must_be_active_since"], d["max_candidates"])
    below_cap = meta.sort_values("record_years", ascending=False).drop_duplicates("monitoring_location_id").iloc[len(top):]
    rejected = pd.concat([rejected, not_gw, below_cap.assign(reason="beyond candidate pre-selection cap")], ignore_index=True)
    return kept, rejected, n_discovered


def download_daily(site_id: str, start: str, end: str, cfg: dict[str, Any], cache: RawCache) -> pd.DataFrame:
    u = cfg["usgs"]
    url = usgs.collection_url(u["base_url"], u["api_version"], "daily")
    feats = cache.get_json(f"usgs_daily_{site_id}_{u['parameter_code']}.json", lambda: usgs.fetch_all_items(
        url, {**usgs.daily_query_params(site_id, u["parameter_code"], u.get("statistic_id", "00003")),
              "time": f"{start}/{end}", "limit": u["page_limit"]}, headers=usgs.api_headers(u.get("api_key_env", "USGS_API_KEY")),
        timeout=u["timeout_s"]))
    df = usgs.parse_daily_values(feats, u["parameter_code"], u.get("statistic_id"))
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)


def sensor_id_for(monitoring_location_id: str) -> str:
    return f"USGS_{usgs.bare_site_number(monitoring_location_id)}"


def build_real_dataset(cfg: dict[str, Any], cache: RawCache, max_stations: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Full real-data build. Returns (common-schema observations, stations table, selection report)."""
    from src.evaluation.sufficiency import station_sufficiency  # local import: avoids a cycle at module load
    start, end = cfg["start_date"], resolve_end_date(cfg)
    kept, rejected, n_discovered = discover_candidates(cfg, cache)
    rej = [{"id": r.monitoring_location_id, "reason": r.reason} for r in rejected.itertuples()]
    limit = max_stations or cfg["discovery"]["target_stations"]
    rows, stations, suff_rows = [], [], []
    for r in kept.itertuples():
        if len(stations) >= min(limit, cfg["discovery"]["max_stations"]):
            rej.append({"id": r.monitoring_location_id, "reason": "target station count already reached"})
            continue
        try:
            depth = download_daily(r.monitoring_location_id, start, end, cfg, cache)
        except FetchError as exc:
            rej.append({"id": r.monitoring_location_id, "reason": f"download failed: {exc}"[:200]})
            continue
        sid = sensor_id_for(r.monitoring_location_id)
        sres = station_sufficiency(sid, depth.rename(columns={TARGET: TARGET}), start, end, cfg["sufficiency"])
        suff_rows.append(sres)
        if sres["verdict"] == "FAIL":
            rej.append({"id": r.monitoring_location_id, "reason": "data-sufficiency FAIL: " + "; ".join(sres["reasons"])})
            continue
        climate = fetch_nasa_power(float(r.latitude), float(r.longitude), start, end, cfg, cache)
        rows.append(build_common_schema(sid, float(r.latitude), float(r.longitude), depth, climate, start, end))
        stations.append({"sensor_id": sid, "latitude": float(r.latitude), "longitude": float(r.longitude), "name": r.name,
                         "state": r.state, "usgs_monitoring_location_id": r.monitoring_location_id, "source": "USGS"})
    report = {"api": {"base_url": cfg["usgs"]["base_url"], "version": cfg["usgs"]["api_version"],
                      "parameter_code": cfg["usgs"]["parameter_code"], "statistic_id": cfg["usgs"].get("statistic_id")},
              "period": [start, end], "stations_discovered": n_discovered, "stations_rejected": n_discovered - len(stations),
              "stations_selected": len(stations), "rejected": rej, "sufficiency_at_selection": suff_rows,
              "criteria": {"discovery": cfg["discovery"], "sufficiency": cfg["sufficiency"]},
              "cache": {"hits": cache.hits, "downloads": cache.downloads}}
    if not rows:
        raise FetchError("No station passed discovery + data-sufficiency; see the selection report.")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(stations), report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--refresh", action="store_true", help="ignore data/raw/ and download everything again")
    ap.add_argument("--offline", action="store_true", help="use only cached raw files; never access the network")
    ap.add_argument("--max-stations", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths = Paths("real")
    paths.ensure_dirs()
    try:
        df, st, report = build_real_dataset(cfg, RawCache(refresh=a.refresh, offline=a.offline), a.max_stations)
    except FetchError as exc:
        log.error("%s\nCheck the connection, `python scripts/check_usgs_api.py`, and the USGS block in %s.", exc, a.config)
        return 1
    df.to_csv(paths.data_csv, index=False, date_format="%Y-%m-%d")
    st.to_csv(paths.stations_csv, index=False)
    (paths.outputs_dir / "station_selection.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("discovered=%d rejected=%d selected=%d -> %s", report["stations_discovered"], report["stations_rejected"],
             report["stations_selected"], paths.data_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
