"""LIVE probe of the modern USGS Water Data API (run this on a machine with Internet access).

  python scripts/check_usgs_api.py

It does NOT assume anything about the API version or field names: it prints what the live service reports so you can
confirm (or fix) the settings in config/real_data_config.json. Checks, in order:
  1. landing page + collections list          -> are 'monitoring-locations', 'time-series-metadata', 'daily' present?
  2. one time-series-metadata page for parameter 72019 -> property names, parameter name/units
  3. one monitoring-locations feature          -> property names and Point geometry
  4. one daily-values page                      -> property names, units
Prints PASS/FAIL per check and a final verdict. Exit code 0 only if every check passed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import usgs_ogc as usgs  # noqa: E402
from data.fetch_real_data import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    u = cfg["usgs"]
    base = f"{u['base_url'].rstrip('/')}/{u['api_version']}"
    hdr = usgs.api_headers(u.get("api_key_env", "USGS_API_KEY"))
    ok, reached = True, True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    try:
        cols = usgs.http_get_json(f"{base}/collections", {"f": "json"}, hdr)
        ids = sorted(c.get("id") for c in cols.get("collections", []))
        print("collections:", ids)
        for need in ("monitoring-locations", "time-series-metadata", "daily"):
            check(f"collection '{need}' exists", need in ids)
        ts = usgs.http_get_json(f"{base}/collections/time-series-metadata/items",
                                {"f": "json", "parameter_code": u["parameter_code"],
                                 **({"statistic_id": u["statistic_id"]} if u.get("statistic_id") else {}), "limit": 3}, hdr)
        feats = ts.get("features", [])
        check("time-series-metadata returns features for parameter " + u["parameter_code"], len(feats) > 0)
        if feats:
            p = feats[0]["properties"]
            print("  metadata properties:", sorted(p))
            print("  parameter_name:", p.get("parameter_name") or p.get("parameter_description"), "| unit:", p.get("unit_of_measure"))
            mid = p.get("monitoring_location_id")
            # v1: /monitoring-locations is filtered by `id` (not `monitoring_location_id`, which only exists on
            # /time-series-metadata and /daily)
            loc = usgs.http_get_json(f"{base}/collections/monitoring-locations/items",
                                     {"f": "json", **usgs.monitoring_location_filter(mid), "limit": 1}, hdr)
            lf = loc.get("features", [])
            check("monitoring-locations returns a Point", bool(lf) and (lf[0].get("geometry") or {}).get("type") == "Point")
            if lf:
                print("  location properties:", sorted(lf[0]["properties"]))
            # discovery sends several ids at once (comma-separated), so probe that form too
            ids = list(dict.fromkeys(f["properties"].get("monitoring_location_id") for f in feats))
            ids = [i for i in ids if i]
            if len(ids) > 1:
                multi = usgs.http_get_json(f"{base}/collections/monitoring-locations/items",
                                           {"f": "json", **usgs.monitoring_location_filter(ids), "limit": len(ids)}, hdr)
                got = {f.get("id") or f["properties"].get("id") for f in multi.get("features", [])}
                check("monitoring-locations accepts comma-separated ids", set(ids) <= got, f"(asked {len(ids)}, got {len(got)})")
            # same core filter as the pipeline's download_daily (incl. the configured statistic_id)
            d = usgs.http_get_json(f"{base}/collections/daily/items",
                                   {"f": "json", **usgs.daily_query_params(mid, u["parameter_code"], u.get("statistic_id")), "limit": 3}, hdr)
            df = d.get("features", [])
            check("daily returns values", len(df) > 0)
            if df:
                stats = sorted({str(f["properties"].get("statistic_id")) for f in df})
                print("  daily statistic_id(s) returned:", stats, "| configured:", u.get("statistic_id"))
                print("  daily properties:", sorted(df[0]["properties"]), "->", json.dumps(df[0]["properties"])[:200])
                parsed = usgs.parse_daily_values(df, u["parameter_code"], u.get("statistic_id"))
                check("parser converts values to metres", len(parsed) > 0, f"({parsed.to_dict('records')[:1]})")
    except usgs.FetchError as exc:
        # an "HTTP 4xx from <url>" error means the service answered (our query was rejected); only a failure
        # without an HTTP status (timeout / DNS / connection) means it was not reached
        reached = str(exc).startswith("HTTP ")
        check("API request failed" if reached else "API reachable", False, str(exc)[:300])
    if not reached:
        print("\nUSGS live API: NOT TESTED (could not reach the service from this machine/network)")
    else:
        print("\nUSGS live API:", "TESTED - all checks passed" if ok else "TESTED - ONE OR MORE CHECKS FAILED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
