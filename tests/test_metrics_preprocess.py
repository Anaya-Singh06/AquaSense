import unittest

import numpy as np
import pandas as pd

from tests._helpers import ROOT  # noqa: F401
from src.config import TARGET
from src.evaluation.metrics import evaluate_against_persistence, mae, nse, persistence_forecast, rmse
from src.preprocessing.features import FEATURE_COLUMNS, add_targets, build_features
from src.preprocessing.preprocess import fill_missing, preprocess, regularize_daily, validate_and_clean
from src.preprocessing.splits import make_split_dates
from src.preprocessing.windows import make_windows


def _frame(n=120, sid="S1", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sensor_id": sid, "latitude": 10.0, "longitude": 78.0,
                         "date": pd.date_range("2022-01-01", periods=n),
                         TARGET: 10 + np.cumsum(rng.normal(0.02, 0.05, n)),
                         "rainfall_mm": rng.gamma(0.5, 5, n), "temperature_c": 28 + rng.normal(0, 1, n),
                         "evapotranspiration_mm": 4 + rng.normal(0, .3, n), "soil_moisture": 0.3 + rng.normal(0, .01, n)})


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y, p = np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 6.])
        self.assertAlmostEqual(mae(y, p), 0.5)
        self.assertAlmostEqual(rmse(y, p), 1.0)
        self.assertAlmostEqual(nse(y, y), 1.0)
        self.assertAlmostEqual(nse(y, np.full(4, y.mean())), 0.0)
        self.assertAlmostEqual(nse(y, p), 1 - 4 / 5)

    def test_persistence_and_report(self):
        self.assertEqual(persistence_forecast(np.array([1., 2.]), 3).shape, (2, 3))
        y = np.array([1., 2., 3., 4., 5., 6.]); sens = np.array(list("aaabbb"))
        r = evaluate_against_persistence(y, y, y - 1, sens)
        self.assertTrue(r["model_beats_persistence"])
        self.assertAlmostEqual(r["rmse_skill_vs_persistence"], 1.0)


class TestPreprocess(unittest.TestCase):
    def test_clean_duplicates_ranges_gaps(self):
        df = _frame(60)
        df = pd.concat([df, df.iloc[[5]]])  # duplicate
        df.loc[10, TARGET] = -5.0           # out of range -> NaN
        df.loc[20:23, TARGET] = np.nan      # 4-day gap -> short fill
        df.loc[30:50, TARGET] = np.nan      # 21-day gap -> long-gap fill
        df.loc[0:1, "rainfall_mm"] = np.nan
        clean, rep = preprocess(df)
        self.assertEqual(rep["duplicates_removed"], 1)
        self.assertEqual(rep["out_of_range_nullified"][TARGET], 1)
        self.assertEqual(len(clean), 60)
        self.assertTrue(clean[TARGET].notna().all())
        flags = clean.set_index("date")["depth_imputed"]
        self.assertTrue((flags.iloc[20:24] == 1).all())
        self.assertTrue((flags.iloc[30:51] == 2).all())
        self.assertEqual(int(flags.iloc[15]), 0)
        self.assertEqual(int((clean["depth_imputed"] == 1).sum()), 5)   # idx10 (1 day) + 4-day gap
        self.assertEqual(int((clean["depth_imputed"] == 2).sum()), 21)
        self.assertTrue((clean["rainfall_mm"].iloc[:2] == 0).all())

    def test_very_long_gap_left_missing(self):
        df = _frame(100); df.loc[20:70, TARGET] = np.nan  # 51 days
        clean, _ = preprocess(df)
        self.assertEqual(int(clean[TARGET].isna().sum()), 51)

    def test_unsorted_input_and_missing_dates(self):
        df = _frame(40).sample(frac=1, random_state=1).iloc[:-5]
        reg = regularize_daily(validate_and_clean(df)[0])
        self.assertTrue((reg["date"].diff().dropna() == pd.Timedelta(days=1)).all())

    def test_missing_columns_raise(self):
        import tempfile, pathlib
        from src.preprocessing.preprocess import load_dataset
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.csv"; pd.DataFrame({"a": [1]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_dataset(p)


class TestFeaturesAndSplits(unittest.TestCase):
    def setUp(self):
        self.clean, _ = preprocess(_frame(200))

    def test_all_features_present(self):
        f = build_features(self.clean)
        for c in FEATURE_COLUMNS:
            self.assertIn(c, f.columns)

    def test_no_future_leakage(self):
        """Changing values AFTER day t must not change features at day t."""
        base = build_features(self.clean)
        mod = self.clean.copy()
        t = 120
        mod.loc[mod.index > t, [TARGET, "rainfall_mm", "temperature_c"]] += 50.0
        alt = build_features(mod)
        a = base.loc[:t, FEATURE_COLUMNS].to_numpy(float); b = alt.loc[:t, FEATURE_COLUMNS].to_numpy(float)
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_targets_are_shifted_future(self):
        f = add_targets(build_features(self.clean), (7,))
        g = f.sort_values("date")
        np.testing.assert_allclose(g["target_7d"].iloc[:10].to_numpy(), g[TARGET].iloc[7:17].to_numpy())

    def test_split_chronological_and_proportions(self):
        s = make_split_dates(self.clean["date"])
        self.assertLess(s.train_end, s.val_end); self.assertLess(s.val_end, s.end)
        n = self.clean["date"].nunique()
        self.assertAlmostEqual((self.clean["date"] <= s.train_end).sum() / len(self.clean), 0.70, delta=0.02)
        self.assertAlmostEqual(((self.clean["date"] > s.train_end) & (self.clean["date"] <= s.val_end)).sum() / len(self.clean), 0.15, delta=0.02)
        self.assertGreater(n, 0)

    def test_windows_alignment_and_no_split_straddle(self):
        s = make_split_dates(self.clean["date"])
        c = self.clean.copy(); c["obs"] = 1
        w = make_windows(c, [TARGET], TARGET, 10, 3, s, "obs")
        i = 5
        g = c.sort_values("date")[TARGET].to_numpy()
        np.testing.assert_allclose(w["X"][i, :, 0], g[i:i + 10])
        np.testing.assert_allclose(w["Y"][i], g[i + 10:i + 13])
        m = w["meta"]
        last_target = m["origin_date"] + pd.Timedelta(days=3)
        self.assertTrue((last_target[m.split == "train"] <= s.train_end).all())
        self.assertTrue((m.origin_date[m.split == "test"] >= s.val_end).all())


if __name__ == "__main__":
    unittest.main()
