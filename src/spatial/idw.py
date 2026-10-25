"""Inverse Distance Weighting (IDW) interpolation of groundwater depth + GeoJSON export.

w_i = 1 / d_i^p,   z(x) = sum(w_i z_i) / sum(w_i)   (d_i = great-circle distance in km).
Status classes use PROTOTYPE thresholds from ``src.config`` - NOT official regulatory categories.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.config import STATUS_CRITICAL_M, STATUS_SEMI_CRITICAL_M, classify_depth

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km); arguments broadcast."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def idw_interpolate(st_lat: np.ndarray, st_lon: np.ndarray, st_val: np.ndarray,
                    q_lat: np.ndarray, q_lon: np.ndarray, power: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate at query points. Returns (values, distance_to_nearest_station_km)."""
    st_lat, st_lon, st_val = (np.asarray(a, float) for a in (st_lat, st_lon, st_val))
    q_lat, q_lon = np.asarray(q_lat, float), np.asarray(q_lon, float)
    if power <= 0:
        raise ValueError("power must be > 0")
    if not (len(st_lat) == len(st_lon) == len(st_val)) or len(st_lat) == 0:
        raise ValueError("station arrays must be non-empty and equal length")
    ok = ~np.isnan(st_val)
    st_lat, st_lon, st_val = st_lat[ok], st_lon[ok], st_val[ok]
    if len(st_val) == 0:
        raise ValueError("no valid station values")
    d = haversine_km(q_lat[:, None], q_lon[:, None], st_lat[None, :], st_lon[None, :])
    nearest = d.min(axis=1)
    w = 1.0 / np.power(np.maximum(d, 1e-9), power)  # clamp: avoids 0-division at station locations
    out = (w * st_val[None, :]).sum(axis=1) / w.sum(axis=1)
    exact = nearest < 1e-9  # query on top of a station: return that station's value
    if exact.any():
        out[exact] = st_val[d[exact].argmin(axis=1)]
    return out, nearest


def geographic_bounds(st_lat: np.ndarray, st_lon: np.ndarray, margin_frac: float = 0.10,
                      min_margin_deg: float = 0.05) -> dict[str, float]:
    """Bounding box of the ACTIVE stations plus a margin (fraction of the span). Nothing region-specific is assumed."""
    st_lat, st_lon = np.asarray(st_lat, float), np.asarray(st_lon, float)
    dlat, dlon = np.ptp(st_lat), np.ptp(st_lon)
    m_lat, m_lon = max(min_margin_deg, margin_frac * dlat), max(min_margin_deg, margin_frac * dlon)
    return {"south": float(st_lat.min() - m_lat), "north": float(st_lat.max() + m_lat),
            "west": float(st_lon.min() - m_lon), "east": float(st_lon.max() + m_lon)}


def auto_max_distance_km(st_lat: np.ndarray, st_lon: np.ndarray, floor_km: float = 100.0, factor: float = 2.0) -> float:
    """Mask distance derived from station spacing: max(floor, factor x median nearest-neighbour distance)."""
    st_lat, st_lon = np.asarray(st_lat, float), np.asarray(st_lon, float)
    if len(st_lat) < 2:
        return floor_km
    d = haversine_km(st_lat[:, None], st_lon[:, None], st_lat[None, :], st_lon[None, :])
    np.fill_diagonal(d, np.inf)
    return float(max(floor_km, factor * np.median(d.min(axis=1))))


def _merc_y(lat_deg: np.ndarray) -> np.ndarray:
    return np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat_deg, -85, 85)) / 2))


def _inv_merc_y(y: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def build_idw_raster(st_lat: np.ndarray, st_lon: np.ndarray, st_depth: np.ndarray, power: float = 2.0, nx: int = 120,
                     max_distance_km: float | None = None, bounds: dict[str, float] | None = None) -> dict[str, Any]:
    """Dense IDW surface for smooth rendering. Rows are uniform in Web-Mercator y so a Leaflet ImageOverlay is exact.

    The interpolation itself is the same ``idw_interpolate`` used everywhere; only the sampling grid is denser.
    Cells farther than ``max_distance_km`` from every station are null (not extrapolated).
    """
    st_lat, st_lon, st_depth = (np.asarray(a, float) for a in (st_lat, st_lon, st_depth))
    b = bounds or geographic_bounds(st_lat, st_lon)
    y0, y1 = _merc_y(np.array([b["south"]]))[0], _merc_y(np.array([b["north"]]))[0]
    ratio = (y1 - y0) / max(np.radians(b["east"] - b["west"]), 1e-9)
    ny = int(np.clip(round(nx * ratio), 24, 240))
    lat_rows = _inv_merc_y(np.linspace(y1, y0, ny))                    # north -> south (image row order)
    lon_cols = np.linspace(b["west"], b["east"], nx)
    glat, glon = np.meshgrid(lat_rows, lon_cols, indexing="ij")
    vals, near = idw_interpolate(st_lat, st_lon, st_depth, glat.ravel(), glon.ravel(), power)
    mdk = auto_max_distance_km(st_lat, st_lon) if max_distance_km is None else max_distance_km
    vals = np.where(near > mdk, np.nan, vals)
    ok = ~np.isnan(vals)
    return {"bounds": b, "nx": int(nx), "ny": ny, "max_distance_km": round(float(mdk), 1),
            "values": [None if np.isnan(v) else round(float(v), 2) for v in vals],
            "vmin": float(np.nanmin(vals)) if ok.any() else None, "vmax": float(np.nanmax(vals)) if ok.any() else None}


def build_spatial_grid(st_lat: np.ndarray, st_lon: np.ndarray, st_depth: np.ndarray, power: float = 2.0,
                       n: int = 25, margin_deg: float | None = None, max_distance_km: float | None | str = "auto",
                       as_of: str | None = None, with_raster: bool = True) -> dict[str, Any]:
    """Regular n x n grid over the ACTIVE-station bounding box -> GeoJSON cells (+ dense raster for smooth rendering).

    Bounds, map centre and mask distance are all derived from the stations (``margin_deg`` None = 10 % of the span).
    ``max_distance_km`` = 'auto' derives it from station spacing; None disables masking; a number sets it explicitly.
    """
    st_lat, st_lon, st_depth = (np.asarray(a, float) for a in (st_lat, st_lon, st_depth))
    ok_st = ~np.isnan(st_depth)
    st_lat, st_lon, st_depth = st_lat[ok_st], st_lon[ok_st], st_depth[ok_st]
    if max_distance_km == "auto":
        max_distance_km = auto_max_distance_km(st_lat, st_lon)
    if margin_deg is None:
        bnds = geographic_bounds(st_lat, st_lon)
    else:
        bnds = {"south": float(st_lat.min() - margin_deg), "north": float(st_lat.max() + margin_deg),
                "west": float(st_lon.min() - margin_deg), "east": float(st_lon.max() + margin_deg)}
    lat_e = np.linspace(bnds["south"], bnds["north"], n + 1)
    lon_e = np.linspace(bnds["west"], bnds["east"], n + 1)
    clat, clon = (lat_e[:-1] + lat_e[1:]) / 2, (lon_e[:-1] + lon_e[1:]) / 2
    glat, glon = np.meshgrid(clat, clon, indexing="ij")
    vals, near = idw_interpolate(st_lat, st_lon, st_depth, glat.ravel(), glon.ravel(), power)
    feats = []
    for k, (v, dn) in enumerate(zip(vals, near)):
        if max_distance_km is not None and dn > max_distance_km:
            continue
        i, j = divmod(k, n)
        ring = [[lon_e[j], lat_e[i]], [lon_e[j + 1], lat_e[i]], [lon_e[j + 1], lat_e[i + 1]],
                [lon_e[j], lat_e[i + 1]], [lon_e[j], lat_e[i]]]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[round(x, 5), round(y, 5)] for x, y in ring]]},
                      "properties": {"depth_m_bgl": round(float(v), 2), "status": classify_depth(float(v))}})
    props_extra = {"bounds": bnds, "center": [round((bnds["south"] + bnds["north"]) / 2, 5), round((bnds["west"] + bnds["east"]) / 2, 5)],
                   "max_distance_km": None if max_distance_km is None else round(float(max_distance_km), 1)}
    out = {"type": "FeatureCollection", "features": feats,
            "properties": {"method": "IDW", "power": power, "units": "m below ground level (higher = deeper)",
                           **props_extra,
                           "as_of": as_of, "n_stations": int(len(st_depth)),
                           "thresholds_m": {"SEMI_CRITICAL": STATUS_SEMI_CRITICAL_M, "CRITICAL": STATUS_CRITICAL_M},
                           "disclaimer": "Prototype thresholds - not official regulatory groundwater categories."}}
    if with_raster:
        out["raster"] = build_idw_raster(st_lat, st_lon, st_depth, power, max_distance_km=max_distance_km, bounds=bnds)
    return out
