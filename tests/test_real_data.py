"""Tests for the MODERN-USGS + NASA POWER adapter.

Fixtures in tests/fixtures are HAND-AUTHORED in the documented OGC/GeoJSON shape; they are not captured live responses.
Live behaviour must be checked on a machine with Internet access: python scripts/check_usgs_api.py
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tests._helpers import FIXTURES, ROOT  # noqa: F401
import data.fetch_real_data as fr
import data.usgs_ogc as usgs
from src.config import COMMON_COLUMNS, TARGET

MAPPING = {"PRECTOTCORR": "rainfall_mm", "T2M": "temperature_c", "EVLAND": "evapotranspiration_mm", "GWETROOT": "soil_moisture"}


def _j(name):
    return json.loads((FIXTURES / name).read_text())


class TestUsgsParsing(unittest.TestCase):
    def test_daily_values_units_and_filtering(self):
        df = usgs.parse_daily_values(_j("usgs_daily_sample.json")["features"])
        self.assertEqual(len(df), 8)                              # null, 'Ssn' and parameter 62610 dropped
        self.assertAlmostEqual(df[TARGET].iloc[0], 41.20 * 0.3048, places=4)   # feet -> metres bgl
        self.assertTrue(df["date"].is_monotonic_increasing)

    def test_unit_conversion_and_unknown_unit(self):
        self.assertAlmostEqual(usgs.unit_factor_to_m("ft"), 0.3048)
        self.assertEqual(usgs.unit_factor_to_m("m"), 1.0)
        with self.assertRaises(usgs.FetchError):
            usgs.unit_factor_to_m("furlongs")
        f = _j("usgs_daily_sample.json")["features"][:2]
        for x in f:
            x["properties"]["unit_of_measure"] = "m"
        self.assertAlmostEqual(usgs.parse_daily_values(f)[TARGET].iloc[0], 41.20)

    def test_empty_daily(self):
        self.assertTrue(usgs.parse_daily_values([]).empty)

    def test_monitoring_locations_drop_invalid_coordinates(self):
        df = usgs.parse_monitoring_locations(_j("usgs_locations_sample.json")["features"])
        self.assertNotIn("USGS-0000004", set(df["monitoring_location_id"]))     # geometry None
        row = df[df["monitoring_location_id"] == "USGS-0000001"].iloc[0]
        self.assertAlmostEqual(row["latitude"], 31.5); self.assertAlmostEqual(row["longitude"], -97.25)

    def test_metadata_parameter_filter(self):
        m = usgs.parse_time_series_metadata(_j("usgs_ts_metadata_sample.json")["features"])
        self.assertNotIn("USGS-0000005", set(m["monitoring_location_id"]))      # parameter 62610 ignored
        self.assertEqual(len(m), 5)


class TestStationFiltering(unittest.TestCase):
    def test_candidate_filter(self):
        meta = usgs.parse_time_series_metadata(_j("usgs_ts_metadata_sample.json")["features"])
        loc = usgs.parse_monitoring_locations(_j("usgs_locations_sample.json")["features"])
        m = meta.merge(loc, on="monitoring_location_id", how="left")
        kept, rej = usgs.filter_candidates(m, min_record_years=6, must_be_active_since="2024-01-01", max_candidates=10)
        self.assertEqual(sorted(kept["monitoring_location_id"]), ["USGS-0000001", "USGS-0000006"])
        reasons = dict(zip(rej["monitoring_location_id"], rej["reason"]))
        self.assertIn("shorter", reasons["USGS-0000002"]); self.assertIn("no data since", reasons["USGS-0000003"])
        self.assertIn("coordinates", reasons["USGS-0000004"])

    def test_candidate_cap(self):
        meta = usgs.parse_time_series_metadata(_j("usgs_ts_metadata_sample.json")["features"]).assign(latitude=30.0, longitude=-97.0)
        kept, rej = usgs.filter_candidates(meta, 6, "2024-01-01", max_candidates=1)
        self.assertEqual(len(kept), 1); self.assertTrue((rej["reason"] == "beyond max_candidates cap").any())

    def test_pagination_follows_next_link(self):
        pages = [{"features": [{"id": 1}], "links": [{"rel": "next", "href": "http://x/p2"}]}, {"features": [{"id": 2}], "links": []}]
        with mock.patch.object(usgs, "http_get_json", side_effect=pages):
            self.assertEqual(len(usgs.fetch_all_items("http://x/p1", {})), 2)

    def test_api_key_comes_from_environment_only(self):
        with mock.patch.dict("os.environ", {"USGS_API_KEY": "abc"}):
            self.assertEqual(usgs.api_headers()["X-Api-Key"], "abc")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertNotIn("X-Api-Key", usgs.api_headers())

    def test_no_legacy_endpoints_in_source(self):
        for f in ("data/fetch_real_data.py", "data/usgs_ogc.py", "config/real_data_config.json"):
            body = (ROOT / f).read_text()
            self.assertNotIn("https://waterservices.usgs.gov", body, f)
            self.assertIn("api.waterdata.usgs.gov", (ROOT / "config/real_data_config.json").read_text())


class TestV1EndpointProperties(unittest.TestCase):
    """USGS v1: /monitoring-locations is filtered by `id`; /time-series-metadata and /daily by `monitoring_location_id`."""

    def test_filter_helper_uses_id_not_monitoring_location_id(self):
        self.assertEqual(usgs.monitoring_location_filter("USGS-351132079275301"), {"id": "USGS-351132079275301"})
        self.assertEqual(usgs.monitoring_location_filter(["USGS-1", "USGS-2"]), {"id": "USGS-1,USGS-2"})
        self.assertNotIn("monitoring_location_id", usgs.monitoring_location_filter("USGS-1"))

    def test_config_uses_v1(self):
        cfg = json.loads((ROOT / "config" / "real_data_config.json").read_text())
        self.assertEqual(cfg["usgs"]["api_version"], "v1")
        self.assertEqual(cfg["usgs"]["parameter_code"], "72019")

    def test_discovery_and_download_use_the_right_property_per_collection(self):
        cfg = json.loads((ROOT / "config" / "real_data_config.json").read_text())
        meta, loc, daily = (_j("usgs_ts_metadata_sample.json"), _j("usgs_locations_sample.json"), _j("usgs_daily_sample.json"))
        seen = {}

        def fake_items(url, params, **k):
            name = url.split("/collections/")[1].split("/")[0]
            seen[name] = params
            return {"time-series-metadata": meta, "monitoring-locations": loc, "daily": daily}[name]["features"]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(usgs, "fetch_all_items", side_effect=fake_items):
            cache = fr.RawCache(Path(d))
            fr.discover_candidates(cfg, cache)
            fr.download_daily("USGS-0000001", "2021-03-01", "2021-03-12", cfg, cache)
        self.assertIn("/v1/", usgs.collection_url(cfg["usgs"]["base_url"], cfg["usgs"]["api_version"], "daily"))
        # monitoring-locations: `id` only
        self.assertIn("id", seen["monitoring-locations"]); self.assertNotIn("monitoring_location_id", seen["monitoring-locations"])
        self.assertIn("USGS-0000001", seen["monitoring-locations"]["id"])
        # time-series-metadata: no location property in discovery; daily: `monitoring_location_id`, never `id`
        self.assertNotIn("id", seen["time-series-metadata"])
        self.assertEqual(seen["daily"]["monitoring_location_id"], "USGS-0000001"); self.assertNotIn("id", seen["daily"])

    def test_locations_parser_reads_id_when_no_monitoring_location_id_property(self):
        feat = {"type": "Feature", "id": "USGS-351132079275301", "geometry": {"type": "Point", "coordinates": [-79.46, 35.19]},
                "properties": {"monitoring_location_number": "351132079275301", "monitoring_location_name": "X",
                               "site_type_code": "GW", "state_name": "North Carolina"}}
        df = usgs.parse_monitoring_locations([feat])
        self.assertEqual(df["monitoring_location_id"].tolist(), ["USGS-351132079275301"])


LIVE_DAILY_MIN = {  # shape of the live /daily record reported by scripts/check_usgs_api.py (statistic 00002 = daily MINIMUM)
    "type": "Feature", "id": "live-1", "geometry": None,
    "properties": {"time_series_id": "00024da3e7be4be4bc1a0e74e65f9c66", "monitoring_location_id": "USGS-351132079275301",
                   "parameter_code": "72019", "statistic_id": "00002", "time": "2015-08-06", "value": "42.13",
                   "unit_of_measure": "ft"}}


class TestLiveDailyStatistic(unittest.TestCase):
    def test_minimum_series_is_not_accepted_as_configured_mean(self):
        # configured statistic is 00003 (mean); a 00002 (minimum) record must NOT be silently used as a mean
        self.assertTrue(usgs.parse_daily_values([LIVE_DAILY_MIN], "72019", "00003").empty)

    def test_live_record_parses_when_its_statistic_is_requested(self):
        df = usgs.parse_daily_values([LIVE_DAILY_MIN], "72019", "00002")
        self.assertEqual(len(df), 1)
        self.assertEqual(df["date"].iloc[0], pd.Timestamp("2015-08-06"))
        self.assertAlmostEqual(df[TARGET].iloc[0], 42.13 * 0.3048, places=4)

    def test_mixed_statistics_keep_only_the_configured_one(self):
        mean = json.loads(json.dumps(LIVE_DAILY_MIN))
        mean["properties"].update(statistic_id="00003", value="43.00")
        df = usgs.parse_daily_values([LIVE_DAILY_MIN, mean], "72019", "00003")
        self.assertEqual(len(df), 1); self.assertAlmostEqual(df[TARGET].iloc[0], 43.00 * 0.3048, places=4)

    def test_daily_query_params_request_the_statistic_with_monitoring_location_id(self):
        p = usgs.daily_query_params("USGS-351132079275301", "72019", "00003")
        self.assertEqual(p, {"monitoring_location_id": "USGS-351132079275301", "parameter_code": "72019", "statistic_id": "00003"})
        self.assertNotIn("id", p)

    def test_pipeline_daily_request_uses_shared_params(self):
        cfg = json.loads((ROOT / "config" / "real_data_config.json").read_text())
        seen = {}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
                usgs, "fetch_all_items", side_effect=lambda url, params, **k: seen.update(params) or []):
            fr.download_daily("USGS-1", "2021-03-01", "2021-03-12", cfg, fr.RawCache(Path(d)))
        for k, v in usgs.daily_query_params("USGS-1", cfg["usgs"]["parameter_code"], cfg["usgs"]["statistic_id"]).items():
            self.assertEqual(seen[k], v)


class TestNasaPower(unittest.TestCase):
    def test_parse_fill_and_columns(self):
        df = fr.parse_nasa_power(_j("nasa_power_sample.json"), MAPPING)
        self.assertEqual(len(df), 10); self.assertTrue(np.isnan(df["rainfall_mm"].iloc[5]))   # -999 -> NaN
        self.assertEqual(set(MAPPING.values()) - set(df.columns), set())
        with self.assertRaises(fr.FetchError):
            fr.parse_nasa_power({"bad": 1}, MAPPING)

    def test_unit_conversion_flux_to_mm_per_day(self):
        p = _j("nasa_power_sample.json")
        p["parameters"]["EVLAND"]["units"] = "kg m-2 s-1"
        base = fr.parse_nasa_power(_j("nasa_power_sample.json"), MAPPING)["evapotranspiration_mm"].iloc[0]
        self.assertAlmostEqual(fr.parse_nasa_power(p, MAPPING)["evapotranspiration_mm"].iloc[0], base * 86400.0)

    def test_unknown_units_are_rejected(self):
        p = _j("nasa_power_sample.json")
        p["parameters"]["T2M"]["units"] = "K"
        with self.assertRaises(fr.FetchError):
            fr.parse_nasa_power(p, MAPPING)


class TestMergeAndCache(unittest.TestCase):
    def test_common_schema_merge(self):
        dv = usgs.parse_daily_values(_j("usgs_daily_sample.json")["features"])
        cl = fr.parse_nasa_power(_j("nasa_power_sample.json"), MAPPING)
        out = fr.build_common_schema("USGS_01234567", 31.5, -97.25, dv, cl, "2021-03-01", "2021-03-12")
        self.assertEqual(list(out.columns), COMMON_COLUMNS); self.assertEqual(len(out), 12)
        self.assertEqual(int(out[TARGET].notna().sum()), 8)       # depth only where USGS observed; nothing invented
        self.assertNotIn("is_synthetic_anomalous_extraction", out.columns)   # never fabricate real labels

    def test_cache_hit_refresh_and_offline(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            prod = lambda: calls.append(1) or {"v": 1}
            c = fr.RawCache(Path(d))
            c.get_json("a.json", prod); c.get_json("a.json", prod)
            self.assertEqual(len(calls), 1)                                   # second call served from cache
            fr.RawCache(Path(d), refresh=True).get_json("a.json", prod)
            self.assertEqual(len(calls), 2)                                   # --refresh forces a download
            with self.assertRaises(fr.FetchError):
                fr.RawCache(Path(d), offline=True).get_json("missing.json", prod)

    def test_end_date_auto(self):
        self.assertEqual(fr.resolve_end_date({"end_date": "auto", "end_date_lag_days": 10}, pd.Timestamp("2026-10-04")), "2026-09-24")
        self.assertEqual(fr.resolve_end_date({"end_date": "2025-12-31"}), "2025-12-31")

    def test_http_error_message(self):
        bad = mock.Mock(status_code=422, text="bad parameter", url="http://x")
        with mock.patch.object(usgs.requests, "get", return_value=bad):
            with self.assertRaises(usgs.FetchError) as cm:
                usgs.http_get_json("http://x", {})
        self.assertIn("422", str(cm.exception))

    def test_discover_and_build_with_mocked_services(self):
        """End-to-end adapter logic with mocked HTTP: discovery -> filter -> download -> sufficiency -> merge."""
        cfg = json.loads((ROOT / "config" / "real_data_config.json").read_text())
        cfg.update(start_date="2021-03-01", end_date="2021-03-12")
        cfg["discovery"].update(target_stations=5)
        meta, loc, daily, nasa = (_j("usgs_ts_metadata_sample.json"), _j("usgs_locations_sample.json"),
                                  _j("usgs_daily_sample.json"), _j("nasa_power_sample.json"))

        def fake_items(url, params, **k):
            if "time-series-metadata" in url: return meta["features"]
            if "monitoring-locations" in url: return loc["features"]
            return daily["features"]
        resp = mock.Mock(status_code=200); resp.json.return_value = nasa
        with tempfile.TemporaryDirectory() as d, mock.patch.object(usgs, "fetch_all_items", side_effect=fake_items), \
                mock.patch("requests.get", return_value=resp):
            kept, rej, n = fr.discover_candidates(cfg, fr.RawCache(Path(d)))
            self.assertEqual(n, 5)
            self.assertEqual(sorted(kept["monitoring_location_id"]), ["USGS-0000001"])    # 0000006 is a spring (not GW)
            self.assertTrue((rej["reason"] == "not a groundwater monitoring location").any())
            # the 8-day fixture series is far too short -> data-sufficiency FAIL -> no station selected (honest rejection)
            with self.assertRaises(fr.FetchError):
                fr.build_real_dataset(cfg, fr.RawCache(Path(d)))


if __name__ == "__main__":
    unittest.main()
