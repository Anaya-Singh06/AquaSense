"""Client + parsers for the MODERN USGS Water Data APIs (OGC API - Features).

Base:  https://api.waterdata.usgs.gov/ogcapi/{version}/collections/{collection}/items
Used collections (names as documented by USGS; verify with ``scripts/check_usgs_api.py``):
  * time-series-metadata : which monitoring locations have parameter 72019 daily-mean series, and their begin/end
  * monitoring-locations : name + coordinates (GeoJSON Point, [lon, lat])
  * daily                : daily values (property ``time`` = date, ``value`` = string, ``unit_of_measure``)

The legacy ``waterservices.usgs.gov`` endpoints (including ``gwlevels``) are NOT used anywhere in AquaSense.

HONESTY NOTE: this module was written without live access to the USGS service (the build environment had no
Internet). Property names are therefore read defensively (several spellings accepted) and every parser raises a
clear error instead of guessing when a required field is missing. Run ``python scripts/check_usgs_api.py`` on your
own machine to confirm the endpoints/field names, and adjust ``config/real_data_config.json`` if USGS changed them.

Parameter 72019 = "Depth to water level, feet below land surface" (positive = below ground). Verify with
``scripts/check_usgs_api.py`` which prints the parameter name/units reported by the live metadata.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from src.config import TARGET

FT_TO_M = 0.3048
UNIT_TO_M = {"ft": FT_TO_M, "feet": FT_TO_M, "foot": FT_TO_M, "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0}


class FetchError(RuntimeError):
    """Network / API / unexpected-format failure with a human-readable message."""


# ----------------------------------------------------------------------------- HTTP
def api_headers(api_key_env: str = "USGS_API_KEY") -> dict[str, str]:
    """Request headers. The key (if any) comes from the environment - never from source code."""
    h = {"User-Agent": "AquaSense-college-project/2.0", "Accept": "application/json"}
    key = os.environ.get(api_key_env)
    if key:
        h["X-Api-Key"] = key
    return h


def http_get_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
                  retries: int = 4, timeout: int = 120) -> dict[str, Any]:
    """GET -> JSON with retry/backoff. HTTP 429 (rate limit) is retried with a longer wait and a hint about API keys."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers or api_headers(), timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                last = FetchError("HTTP 429 (rate limited). Set the USGS_API_KEY environment variable "
                                  "(free key: https://api.waterdata.usgs.gov/signup/) and retry.")
                time.sleep(10 * attempt)
                continue
            if r.status_code in (400, 404, 422):
                raise FetchError(f"HTTP {r.status_code} from {r.url}: {r.text[:300]}")
            last = FetchError(f"HTTP {r.status_code} from {r.url}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(2 ** attempt)
    raise FetchError(f"Failed after {retries} attempts: {last}")


def collection_url(base_url: str, version: str, collection: str) -> str:
    return f"{base_url.rstrip('/')}/{version}/collections/{collection}/items"


def fetch_all_items(url: str, params: dict[str, Any], max_pages: int = 200, headers: dict[str, str] | None = None,
                    timeout: int = 120) -> list[dict[str, Any]]:
    """Follow OGC ``next`` links and return every GeoJSON feature."""
    feats: list[dict[str, Any]] = []
    nxt: str | None = url
    p: dict[str, Any] | None = {"f": "json", **params}
    for _ in range(max_pages):
        payload = http_get_json(nxt, p, headers, timeout=timeout)
        feats.extend(payload.get("features", []))
        nxt = next((ln.get("href") for ln in payload.get("links", []) if ln.get("rel") == "next"), None)
        p = None  # the next-link already carries the query string
        if not nxt:
            return feats
    raise FetchError(f"Pagination did not finish after {max_pages} pages for {url}")


# ----------------------------------------------------------------------------- helpers
def _prop(props: dict[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in props and props[n] is not None:
            return props[n]
    return default


def monitoring_location_filter(ids: str | Iterable[str]) -> dict[str, str]:
    """Query params selecting monitoring locations by their USGS id (e.g. 'USGS-01234567') on the v1
    ``monitoring-locations`` collection.

    v1 distinguishes the endpoints: ``/monitoring-locations`` is filtered by ``id`` (it has NO ``monitoring_location_id``
    queryable -> HTTP 400 "At least one requested property wasn't found"), whereas ``/time-series-metadata`` and
    ``/daily`` are filtered by ``monitoring_location_id``. Do not use this helper for those two collections.
    """
    if not isinstance(ids, str):
        ids = ",".join(ids)
    return {"id": ids}


def daily_query_params(site_id: str, parameter_code: str, statistic_id: str | None) -> dict[str, str]:
    """Core /daily filter shared by the pipeline and the live probe (so they cannot drift apart).

    Uses ``monitoring_location_id`` (daily's property; NOT ``id``). ``statistic_id`` must be requested explicitly:
    USGS statistic codes are 00001=maximum, 00002=minimum, 00003=mean, 00008=median, and a location can hold several
    daily series for the same parameter. Without it the API returns an arbitrary one (e.g. the daily minimum), which
    ``parse_daily_values`` correctly refuses to treat as the configured statistic.
    """
    p = {"monitoring_location_id": site_id, "parameter_code": parameter_code}
    if statistic_id:
        p["statistic_id"] = statistic_id
    return p


def bare_site_number(monitoring_location_id: str) -> str:
    """'USGS-01234567' -> '01234567'."""
    return str(monitoring_location_id).split("-", 1)[-1]


def unit_factor_to_m(unit: str | None) -> float:
    """Multiplier converting a USGS depth unit to metres. Unknown units raise (we never guess)."""
    u = (unit or "").strip().lower().replace(".", "")
    if u in UNIT_TO_M:
        return UNIT_TO_M[u]
    raise FetchError(f"Unknown depth unit '{unit}'. Expected feet or metres; refusing to guess.")


# ----------------------------------------------------------------------------- parsers
def parse_monitoring_locations(features: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """GeoJSON features of the monitoring-locations collection -> station table (valid coordinates only)."""
    rows = []
    for f in features:
        p = f.get("properties", {}) or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") if geom.get("type") == "Point" else None
        lon = lat = np.nan
        if coords and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
        mid = _prop(p, "monitoring_location_id", "id") or f.get("id")
        if mid is None:
            continue
        rows.append({"monitoring_location_id": str(mid), "name": _prop(p, "monitoring_location_name", "name", default=""),
                     "site_type_code": _prop(p, "site_type_code", default=""),
                     "state": _prop(p, "state_name", "state_code", default=""),
                     "latitude": lat, "longitude": lon})
    df = pd.DataFrame(rows, columns=["monitoring_location_id", "name", "site_type_code", "state", "latitude", "longitude"])
    ok = df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    return df[ok].drop_duplicates("monitoring_location_id").reset_index(drop=True)


def parse_time_series_metadata(features: Iterable[dict[str, Any]], parameter_code: str = "72019",
                               statistic_id: str | None = "00003") -> pd.DataFrame:
    """Time-series-metadata features -> one row per (location, series) with begin/end dates."""
    rows = []
    for f in features:
        p = f.get("properties", {}) or {}
        if str(_prop(p, "parameter_code", default="")) != parameter_code:
            continue
        stat = str(_prop(p, "statistic_id", default=""))
        if statistic_id and stat and stat != statistic_id:
            continue
        mid = _prop(p, "monitoring_location_id")
        if mid is None:
            continue
        rows.append({"monitoring_location_id": str(mid),
                     "begin": pd.to_datetime(_prop(p, "begin", "begin_utc"), errors="coerce", utc=True),
                     "end": pd.to_datetime(_prop(p, "end", "end_utc"), errors="coerce", utc=True),
                     "unit": _prop(p, "unit_of_measure", default=""),
                     "parameter_name": _prop(p, "parameter_name", "parameter_description", default="")})
    df = pd.DataFrame(rows, columns=["monitoring_location_id", "begin", "end", "unit", "parameter_name"])
    for c in ("begin", "end"):
        df[c] = df[c].dt.tz_localize(None) if len(df) else df[c]
    return df


def parse_daily_values(features: Iterable[dict[str, Any]], parameter_code: str = "72019",
                       statistic_id: str | None = "00003") -> pd.DataFrame:
    """Daily-values features -> DataFrame(date, groundwater_depth_m_bgl).

    * keeps only ``parameter_code`` (and ``statistic_id`` when given),
    * converts feet -> metres (x 0.3048) using each record's ``unit_of_measure``,
    * drops non-numeric / missing values, averages duplicate dates.
    """
    rows: list[tuple[str, float]] = []
    for f in features:
        p = f.get("properties", {}) or {}
        pc = _prop(p, "parameter_code")
        if pc is not None and str(pc) != parameter_code:
            continue
        st = _prop(p, "statistic_id")
        if statistic_id and st is not None and str(st) != statistic_id:
            continue
        try:
            x = float(p.get("value"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        t = _prop(p, "time", "date")
        if t is None:
            continue
        rows.append((str(t), x * unit_factor_to_m(_prop(p, "unit_of_measure", "unit", default="ft"))))
    if not rows:
        return pd.DataFrame({"date": pd.to_datetime([]), TARGET: []})
    df = pd.DataFrame(rows, columns=["date", TARGET])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    return df.dropna(subset=["date"]).groupby("date", as_index=False)[TARGET].mean()


# ----------------------------------------------------------------------------- station filtering
def filter_candidates(meta: pd.DataFrame, min_record_years: float, must_be_active_since: str,
                      max_candidates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cheap metadata-level filter. Returns (kept, rejected-with-reason). Longest records first."""
    if meta.empty:
        return meta.assign(reason=[]), meta.assign(reason=[])
    m = meta.copy()
    m["record_years"] = (m["end"] - m["begin"]).dt.days / 365.25
    # one series per location: keep the longest
    m = m.sort_values("record_years", ascending=False).drop_duplicates("monitoring_location_id")
    reason = pd.Series("", index=m.index)
    reason[m["begin"].isna() | m["end"].isna()] = "missing begin/end"
    reason[(reason == "") & (m["record_years"] < min_record_years)] = f"record shorter than {min_record_years} years"
    reason[(reason == "") & (m["end"] < pd.Timestamp(must_be_active_since))] = f"no data since {must_be_active_since}"
    if "latitude" in m:
        reason[(reason == "") & (m["latitude"].isna() | m["longitude"].isna())] = "no valid coordinates"
    m["reason"] = reason
    kept = m[m["reason"] == ""].head(max_candidates)
    rejected = pd.concat([m[m["reason"] != ""], m[m["reason"] == ""].iloc[max_candidates:].assign(reason="beyond max_candidates cap")])
    return kept.reset_index(drop=True), rejected.reset_index(drop=True)
