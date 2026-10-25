"""Tests for data sufficiency, baselines, rolling-origin evaluation, conformal-style intervals, geographic bounds,
IDW, the data-summary API/service and real/synthetic source selection."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tests._helpers import ROOT, allow_xgb_standin  # noqa: F401
allow_xgb_standin()
import src.pipeline as pipeline  # noqa: E402
from backend.service import AquaSenseService  # noqa: E402
from src.config import SOURCE_LABELS, TARGET, Paths  # noqa: E402
from src.data_summary import compute_data_summary  # noqa: E402
from src.evaluation import baselines as bl  # noqa: E402
from src.evaluation import rolling_origin as ro  # noqa: E402
from src.evaluation.sufficiency import overall_conclusion, station_sufficiency, sufficiency_table  # noqa: E402
from src.models.xgboost_forecaster import XGBForecaster  # noqa: E402
from src.spatial.idw import auto_max_distance_km, build_idw_raster, build_spatial_grid, geographic_bounds, idw_interpolate  # noqa: E402

TH = json.loads((ROOT / "config" / "real_data_config.json").read_text())["sufficiency"]


def _series(years=8.0, amp=1.0, noise=0.02, missing=0.05, seed=0, trend=0.0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2015-01-01", periods=int(years * 365.25), freq="D")
    y = 10 + trend * np.arange(len(d)) / 365.25 + amp * np.sin(2 * np.pi * d.dayofyear / 365.25) + rng.normal(0, noise, len(d))
    df = pd.DataFrame({"date": d, TARGET: y})
    df.loc[rng.random(len(d)) < missing, TARGET] = np.nan
    return df


class TestSufficiency(unittest.TestCase):
    def test_good_station_passes(self):
        df = _series()
        r = station_sufficiency("A", df, df["date"].min(), df["date"].max(), TH)
        self.assertEqual(r["verdict"], "PASS", r["reasons"])
        self.assertGreater(r["seasonal_amplitude_m"], 1.5)           # sin amplitude 1 -> range ~2
        self.assertEqual(r["dynamics"], "dynamic")
        self.assertGreater(r["test_obs"], 150)

    def test_short_record_fails(self):
        df = _series(years=1.5)
        self.assertEqual(station_sufficiency("S", df, df["date"].min(), df["date"].max(), TH)["verdict"], "FAIL")

    def test_long_gap_fails_and_is_measured(self):
        df = _series()
        df.loc[500:900, TARGET] = np.nan                              # 401-day gap
        r = station_sufficiency("G", df, df["date"].min(), df["date"].max(), TH)
        self.assertGreaterEqual(r["longest_gap_days"], 401); self.assertEqual(r["verdict"], "FAIL")

    def test_stable_well_is_warned_not_excluded(self):
        df = _series(amp=0.0, noise=0.005, missing=0.02)
        r = station_sufficiency("Z", df, df["date"].min(), df["date"].max(), TH)
        self.assertEqual(r["dynamics"], "stable"); self.assertEqual(r["verdict"], "WARN")   # kept, flagged
        self.assertTrue(any("near-constant" in x for x in r["reasons"]))

    def test_persistence_trivial_flag(self):
        df = _series(years=20, amp=0.0, noise=0.001, trend=0.5, missing=0.0)   # 20 y slow drift: 30 d persistence is almost perfect
        r = station_sufficiency("T", df, df["date"].min(), df["date"].max(), TH)
        self.assertTrue(r["persistence_trivial"], r["persistence_nse_30d_test"])

    def test_table_and_overall(self):
        parts = [_series(seed=i).assign(sensor_id=f"S{i}") for i in range(3)] + [_series(years=1, seed=9).assign(sensor_id="short")]
        df = pd.concat(parts)
        t = sufficiency_table(df, df["date"].min(), df["date"].max(), TH)
        self.assertEqual(set(t["sensor_id"]), {"S0", "S1", "S2", "short"})
        self.assertEqual(overall_conclusion(t)["answer"], "YES")      # 3/4 = 75 % PASS
        self.assertEqual(overall_conclusion(t.assign(verdict="FAIL"))["answer"], "NO")

    def test_thresholds_are_fixed_in_config(self):
        self.assertEqual(TH["min_years_fail"], 3.0); self.assertEqual(TH["persistence_trivial_nse_30d"], 0.95)


class TestBaselinesAndRolling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.preprocessing.features import add_targets, build_features
        from src.preprocessing.preprocess import load_dataset, preprocess
        cls.clean, _ = preprocess(load_dataset(ROOT / "data" / "dwlr_data.csv"))
        cls.feats = add_targets(build_features(cls.clean))

    def test_persistence_and_seasonal_naive_definitions(self):
        c = self.clean[self.clean["sensor_id"] == "DWLR_001"]
        ref = bl.seasonal_reference(c)
        t = pd.Timestamp("2023-06-15")
        got = bl.seasonal_naive(ref, np.array(["DWLR_001"]), [t])[0]
        want = c[(c["date"] >= t - pd.Timedelta(days=368)) & (c["date"] <= t - pd.Timedelta(days=362)) & (c["depth_imputed"] == 0)][TARGET].mean()
        self.assertAlmostEqual(got, want, places=9)                   # +/-3 d mean of REAL observations one year earlier
        self.assertTrue(np.isnan(bl.seasonal_naive(ref, np.array(["DWLR_001"]), [pd.Timestamp("2022-06-15")])[0]))   # no year earlier
        np.testing.assert_allclose(bl.persistence(np.array([1.0, 2.0])), [1.0, 2.0])

    def test_folds_expand_and_do_not_overlap(self):
        folds = ro.make_rolling_folds(self.clean["date"], **ro.default_rolling_params(self.clean["date"].nunique()))
        self.assertGreaterEqual(len(folds), 3)
        for a, b in zip(folds, folds[1:]):
            self.assertGreater(b.splits.train_end, a.splits.train_end)            # training window EXPANDS
            self.assertEqual(a.test_end, b.splits.val_end)                         # origin advances to the end of the previous test block
        for f in folds:
            self.assertLess(f.splits.train_end, f.splits.val_end)
        with self.assertRaises(ValueError):
            ro.make_rolling_folds(self.clean["date"].iloc[:300], 4, 60, 90, 365)

    def test_rolling_xgb_never_leaks_future_targets(self):
        folds = ro.make_rolling_folds(self.clean["date"], **ro.default_rolling_params(self.clean["date"].nunique()))
        long = ro.rolling_xgb(self.feats, self.clean, folds[:2], horizons=(30,))
        self.assertFalse(long.empty)
        for k, f in enumerate(folds[:2]):
            g = long[long["fold"] == k]
            self.assertTrue((g["target_date"] > f.splits.val_end).all() and (g["target_date"] <= f.test_end).all())
        s = ro.summarize(long)["xgboost_30d"]
        for key in ("model", "persistence", "seasonal_naive"):
            self.assertIn("median_station", s[key]); self.assertIn("pooled", s[key])
        self.assertEqual(s["n_samples_compared"] <= s["n_samples_total"], True)
        self.assertIn("empirical_coverage", s["interval"])
        t = ro.metrics_table({"xgboost_30d": s}, "Synthetic")
        self.assertEqual(set(t["Model"]), {"XGBoost", "Persistence", "Seasonal naive"})


class TestConformalInterval(unittest.TestCase):
    def test_conformal_quantile(self):
        r = np.arange(1, 101, dtype=float)
        self.assertEqual(XGBForecaster._conformal_q(r, 0.10), 91.0)   # ceil(101*0.9)=91 -> 91st smallest
        self.assertEqual(XGBForecaster._conformal_q(np.array([1.0, 2.0]), 0.01), 2.0)   # clipped to the max
        self.assertTrue(np.isnan(XGBForecaster._conformal_q(np.array([]), 0.1)))

    def test_coverage_on_exchangeable_noise(self):
        rng = np.random.default_rng(1)
        calib, test = np.abs(rng.normal(0, 1, 2000)), np.abs(rng.normal(0, 1, 20000))
        q = XGBForecaster._conformal_q(calib, 0.10)
        self.assertAlmostEqual(float((test <= q).mean()), 0.90, delta=0.02)


class TestGeography(unittest.TestCase):
    def test_bounds_follow_the_stations_wherever_they_are(self):
        for lat, lon in [(np.array([11.0, 11.5]), np.array([78.0, 78.6])), (np.array([30.0, 44.0]), np.array([-105.0, -80.0]))]:
            b = geographic_bounds(lat, lon)
            self.assertLess(b["south"], lat.min()); self.assertGreater(b["north"], lat.max())
            self.assertLess(b["west"], lon.min()); self.assertGreater(b["east"], lon.max())
            g = build_spatial_grid(lat, lon, np.array([5.0, 9.0]), n=6)
            c = g["properties"]["center"]
            self.assertAlmostEqual(c[0], (b["south"] + b["north"]) / 2, places=3)
            self.assertAlmostEqual(c[1], (b["west"] + b["east"]) / 2, places=3)
            self.assertEqual(g["properties"]["bounds"], b)

    def test_mask_distance_adapts_to_spacing(self):
        near = auto_max_distance_km(np.array([30.0, 30.1]), np.array([-97.0, -97.1]))
        far = auto_max_distance_km(np.array([30.0, 40.0, 35.0]), np.array([-120.0, -80.0, -100.0]))
        self.assertEqual(near, 100.0); self.assertGreater(far, 1000.0)

    def test_raster_is_idw_and_masked(self):
        lat, lon, v = np.array([30.0, 31.0, 32.0]), np.array([-97.0, -96.0, -97.5]), np.array([4.0, 10.0, 25.0])
        r = build_idw_raster(lat, lon, v, power=2.0, nx=40, max_distance_km=None)
        vals = np.array([np.nan if x is None else x for x in r["values"]]).reshape(r["ny"], r["nx"])
        self.assertGreaterEqual(np.nanmin(vals), v.min() - 1e-6); self.assertLessEqual(np.nanmax(vals), v.max() + 1e-6)   # IDW never extrapolates
        r2 = build_idw_raster(lat, lon, v, nx=40, max_distance_km=1.0)
        self.assertGreater(sum(x is None for x in r2["values"]), len(r2["values"]) * 0.9)

    def test_idw_weights_and_power(self):
        val, _ = idw_interpolate(np.array([0.0, 0.0]), np.array([0.0, 2.0]), np.array([1.0, 3.0]), np.array([0.0]), np.array([0.5]), power=2)
        self.assertLess(val[0], 2.0)                                  # closer to the station with value 1
        v1, _ = idw_interpolate(np.array([0.0, 0.0]), np.array([0.0, 2.0]), np.array([1.0, 3.0]), np.array([0.0]), np.array([0.5]), power=4)
        self.assertLess(v1[0], val[0])                                # higher power -> more local


class TestSummaryAndSourceMode(unittest.TestCase):
    def test_summary_fields(self):
        from src.preprocessing.preprocess import load_dataset, preprocess
        clean, _ = preprocess(load_dataset(ROOT / "data" / "dwlr_data.csv"))
        s = compute_data_summary(clean, "synthetic")
        self.assertEqual(s["station_count"], 10); self.assertEqual(s["source_label"], "Synthetic demonstration")
        self.assertEqual(s["date_range"], ["2022-01-01", "2023-12-31"]); self.assertGreater(s["observation_count"], 6000)
        self.assertTrue(0 < s["missing_fraction"] < 0.2); self.assertTrue(s["has_synthetic_labels"])
        self.assertEqual(compute_data_summary(clean.drop(columns=["is_synthetic_anomalous_extraction"]), "real")["source_badge"], "USGS + NASA POWER")

    def test_service_source_selection_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "data").mkdir()
            fac = lambda source="synthetic": Paths(source, root=root)
            with mock.patch.object(pipeline, "Paths", fac):
                pipeline.run("synthetic", regenerate=True, skip_lstm=True, rolling=False)
                raw = pd.read_csv(root / "data" / "dwlr_data.csv").drop(columns=["is_synthetic_anomalous_extraction"])
                raw["sensor_id"] = raw["sensor_id"].str.replace("DWLR_", "USGS_", regex=False)
                raw.to_csv(root / "data" / "real_data.csv", index=False)
                pipeline.run("real", regenerate=False, skip_lstm=True, fetch=False, rolling=False)
            import backend.service as service
            with mock.patch.object(service, "Paths", fac):
                syn, real = service.AquaSenseService("synthetic"), service.AquaSenseService("real")
                with mock.patch.dict(os.environ, {"AQUASENSE_SOURCE": "real"}):
                    self.assertEqual(service.AquaSenseService().health()["source"], "real")
            self.assertEqual(syn.health()["source_label"], SOURCE_LABELS["synthetic"])
            self.assertEqual(real.health()["source_label"], "USGS + NASA POWER")
            self.assertEqual(syn.data_summary()["source"], "synthetic"); self.assertEqual(real.data_summary()["source"], "real")
            self.assertFalse(real.data_summary()["has_synthetic_labels"])     # real mode never carries fabricated labels
            self.assertEqual(real.data_summary()["station_count"], 10)
            sensors = real.sensors()
            self.assertIn("max_active_severity", sensors[0]); self.assertTrue(all(s["sensor_id"].startswith("USGS_") for s in sensors))
            self.assertIn("raster", real.spatial_grid()); self.assertIn("bounds", real.spatial_grid()["properties"])
            m = json.loads((root / "outputs" / "real" / "metrics.json").read_text())
            self.assertNotIn("precision", m["anomaly"])                       # no precision/recall on real anomalies
            self.assertIn("injected_evaluation", m["anomaly"])
            self.assertIn("Injected synthetic anomaly evaluation", m["anomaly"]["injected_evaluation"]["label"])
            self.assertIn("data_sufficiency", m)

    def test_pipeline_real_without_data_and_no_fetch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            fac = lambda source="synthetic": Paths(source, root=Path(d))
            with mock.patch.object(pipeline, "Paths", fac):
                with self.assertRaises(FileNotFoundError):
                    pipeline.run("real", regenerate=False, skip_lstm=True, fetch=False)


if __name__ == "__main__":
    unittest.main()
