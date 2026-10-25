"""Lightweight LightGBM tests. Tests that need the ``lightgbm`` package are SKIPPED when it is not installed;
config / plumbing tests always run (they use a stub forecaster, not LightGBM)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests._helpers import ROOT, allow_xgb_standin, has

allow_xgb_standin()
from src.config import TARGET  # noqa: E402
from src.evaluation import rolling_origin as ro  # noqa: E402
from src.inference import compute_all_forecasts  # noqa: E402
from src.models.lightgbm_forecaster import CONFIG_PATH, LGBMForecaster, load_lgbm_config  # noqa: E402
from src.models.xgboost_forecaster import split_rows  # noqa: E402
from src.preprocessing.features import FEATURE_COLUMNS  # noqa: E402
from src.preprocessing.splits import make_split_dates  # noqa: E402
from tests.test_leakage_masking import pipeline, toy  # noqa: E402

HAS_LGBM = has("lightgbm")
FAST = {"params": {**load_lgbm_config()["params"], "n_estimators": 40, "n_jobs": 1}, "early_stopping_rounds": 5}


class TestConfigAndPlumbing(unittest.TestCase):
    def test_config_file(self) -> None:
        cfg = load_lgbm_config()
        self.assertTrue(CONFIG_PATH.exists())
        for k in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples"):
            self.assertIn(k, cfg["params"])
        self.assertGreater(cfg["early_stopping_rounds"], 0)
        self.assertEqual(cfg["params"]["device_type"], "cpu")

    def test_custom_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"params": {"n_estimators": 7}, "early_stopping_rounds": 3}))
            self.assertEqual(load_lgbm_config(p), {"params": {"n_estimators": 7}, "early_stopping_rounds": 3})

    @unittest.skipIf(HAS_LGBM, "lightgbm installed")
    def test_clear_error_without_lightgbm(self) -> None:
        with self.assertRaises(ImportError) as cm:
            LGBMForecaster(30)
        self.assertIn("pip install lightgbm", str(cm.exception))

    def test_inference_extra_models_and_table_label(self) -> None:
        class Stub:
            def forecast_latest(self, row):   # same interface as XGBForecaster
                return {"horizon_days": 30, "stub": True}
        clean, feats = pipeline(toy(n=400))
        out = compute_all_forecasts(clean, feats, None, None, None, extra_models={"lgbm_30d": Stub()})
        self.assertEqual(set(out), {"S1", "S2"})
        self.assertTrue(all(v["lgbm_30d"] == {"horizon_days": 30, "stub": True} for v in out.values()))
        self.assertTrue(all(v["xgb_30d"] is None for v in out.values()))     # existing keys untouched
        long = pd.DataFrame({"fold": 0, "model": "lightgbm", "horizon": 30, "sensor_id": "A", "y_true": np.arange(20.0),
                             "y_model": np.arange(20.0) + 0.1, "y_persistence": np.arange(20.0) + 1, "y_seasonal_naive": np.arange(20.0) + 2,
                             "lo": np.nan, "hi": np.nan})
        t = ro.metrics_table(ro.summarize(long), "Test")
        self.assertEqual(set(t["Model"]), {"LightGBM", "Persistence", "Seasonal naive"})


@unittest.skipUnless(HAS_LGBM, "lightgbm not installed (pip install lightgbm)")
class TestLightGBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean, cls.feats = pipeline(toy())
        cls.splits = make_split_dates(cls.clean["date"])
        cls.parts = split_rows(cls.feats, 30, cls.splits)
        cls.model = LGBMForecaster(30, config=FAST).fit(cls.parts["train"], cls.parts["val"])

    def test_setup_and_training(self) -> None:
        self.assertEqual(self.model.backend, "lightgbm")
        self.assertEqual(self.model.n_train, len(self.parts["train"]))
        self.assertTrue(np.isfinite(self.model.conformal_halfwidth) and self.model.conformal_halfwidth > 0)
        self.assertGreaterEqual(self.model.best_iteration, 1)

    def test_prediction_shape_and_finite(self) -> None:
        test = self.parts["test"]
        p, lo, hi = self.model.predict_interval(test)
        self.assertEqual(p.shape, (len(test),))
        self.assertTrue(np.isfinite(p).all() and (lo < p).all() and (p < hi).all())

    def test_uses_same_rows_as_xgboost_split_and_no_imputed_targets(self) -> None:
        for name, d in self.parts.items():
            self.assertTrue((d["model_input_ok"] == 1).all() and (d["target_30d_observed"] == 1).all(), name)
        self.assertLess((self.parts["train"]["date"] + pd.Timedelta(days=30)).max(), self.splits.train_end + pd.Timedelta(days=1))
        self.assertGreater((self.parts["test"]["date"] + pd.Timedelta(days=30)).min(), self.splits.val_end)

    def test_evaluate_reports_station_metrics(self) -> None:
        r = self.model.evaluate(self.parts["test"])
        self.assertEqual(r["backend"], "lightgbm")
        for k in ("mae", "rmse", "nse_pooled", "nse_per_sensor_mean"):
            self.assertIn(k, r["model"])

    def test_save_load_roundtrip_keeps_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.model.save(d)
            self.assertEqual(sorted(x.name for x in d.iterdir()), ["lgbm_30d.txt", "lgbm_30d_meta.json"])   # never xgb_* names
            m2 = LGBMForecaster.load(d, 30)
            test = self.parts["test"]
            np.testing.assert_allclose(m2.predict_level(test), self.model.predict_level(test), rtol=1e-6, atol=1e-8)
            self.assertEqual(m2.conformal_halfwidth, self.model.conformal_halfwidth)
            self.assertEqual(json.loads((d / "lgbm_30d_meta.json").read_text())["features"], FEATURE_COLUMNS)
            with self.assertRaises(FileNotFoundError):
                LGBMForecaster.load(d, 90)

    def test_forecast_latest_matches_existing_output_shape(self) -> None:
        latest = self.feats.dropna(subset=FEATURE_COLUMNS).sort_values("date").groupby("sensor_id").tail(1).head(1)
        f = self.model.forecast_latest(latest)
        self.assertEqual(len(f["trajectory"]), 30)
        self.assertLessEqual(f["endpoint_lower"], f["endpoint_depth_m_bgl"])
        out = compute_all_forecasts(self.clean, self.feats, None, None, None, extra_models={"lgbm_30d": self.model})
        self.assertEqual(set(out), {"S1", "S2"})                              # station ids preserved
        self.assertEqual(len(out["S1"]["lgbm_30d"]["trajectory"]), 30)

    def test_too_little_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            LGBMForecaster(30, config=FAST).fit(self.parts["train"].head(10), self.parts["val"])

    def test_rolling_lightgbm_same_loop_no_future_targets(self) -> None:
        folds = ro.make_rolling_folds(self.clean["date"], n_folds=2, test_days=60, val_days=90, min_train_days=365)
        long = ro.rolling_lightgbm(self.feats, self.clean, folds, horizons=(30,), config=FAST)
        self.assertFalse(long.empty)
        self.assertEqual(set(long["model"]), {"lightgbm"})
        f0 = {f.index: f for f in folds}
        self.assertTrue(all(r.origin_date >= f0[r.fold].splits.val_end - pd.Timedelta(days=30) and r.target_date > f0[r.fold].splits.val_end
                            for r in long.itertuples()))
        self.assertTrue(all(r.target_date <= f0[r.fold].test_end for r in long.itertuples()))
        s = ro.summarize(long)["lightgbm_30d"]
        self.assertIn("rmse_skill_vs_persistence_pooled", s)
        self.assertEqual(set(long["sensor_id"]), {"S1", "S2"})


if __name__ == "__main__":
    unittest.main()
