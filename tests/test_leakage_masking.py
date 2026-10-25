"""Leakage regression tests: no interpolated groundwater value may reach any model input or target."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests._helpers import allow_xgb_standin, has

allow_xgb_standin()

from src.config import TARGET  # noqa: E402
from src.evaluation import rolling_origin as ro  # noqa: E402
from src.models.xgboost_forecaster import split_rows  # noqa: E402
from src.preprocessing.features import FEATURE_COLUMNS, GW_LOOKBACK_DAYS, INPUT_OK_COL, add_targets, build_features  # noqa: E402
from src.preprocessing.preprocess import fill_missing  # noqa: E402
from src.preprocessing.splits import make_split_dates  # noqa: E402
from src.preprocessing.windows import make_windows  # noqa: E402

GAPS = {"S1": [(100, 1), (300, 3), (500, 10), (800, 40), (1100, 7), (1300, 2)], "S2": [(150, 2), (420, 5), (900, 25)]}


def toy(n: int = 1500, seed: int = 0, perturb_after: int | None = None, gaps=GAPS) -> pd.DataFrame:
    """Raw (pre-fill) daily frame with NaN gaps. ``perturb_after``: add noise to observations with index > this."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    rows = []
    for k, (sid, gl) in enumerate(gaps.items()):
        y = 10 + np.cumsum(rng.normal(0, 0.05, n)) + 0.5 * np.sin(np.arange(n) / 58 + k)
        for start, ln in gl:
            y[start:start + ln] = np.nan
        if perturb_after is not None:
            noise = np.random.default_rng(99).normal(0, 3, n)
            m = (np.arange(n) > perturb_after) & ~np.isnan(y)
            y[m] += noise[m]
        rows.append(pd.DataFrame({"sensor_id": sid, "date": dates, "latitude": 10.0 + k, "longitude": 78.0, TARGET: y,
                                  "rainfall_mm": rng.gamma(0.3, 5, n), "temperature_c": 25 + 5 * np.sin(np.arange(n) / 58),
                                  "evapotranspiration_mm": 3 + np.cos(np.arange(n) / 58), "soil_moisture": 0.3 + 0.05 * np.sin(np.arange(n) / 30)}))
    return pd.concat(rows, ignore_index=True)


def pipeline(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = fill_missing(raw)
    return clean, add_targets(build_features(clean))


def imputed_lookup(clean: pd.DataFrame) -> pd.Series:
    return clean.set_index(["sensor_id", "date"])["depth_imputed"]


class TestLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean, cls.feats = pipeline(toy())
        cls.imp = imputed_lookup(cls.clean)
        cls.splits = make_split_dates(cls.clean["date"])

    def test_toy_has_imputation(self) -> None:
        self.assertGreater(int((self.clean["depth_imputed"] > 0).sum()), 20)

    # TEST 1 - future invariance --------------------------------------------------------------------
    def test_1_future_invariance(self) -> None:
        base = self.feats.set_index(["sensor_id", "date"])
        unmasked_changed_total = 0
        for origin_idx in (90, 301, 421, 502, 700, 1102, 1301):  # several origins fall INSIDE a gap
            origin = pd.Timestamp("2015-01-01") + pd.Timedelta(days=origin_idx)
            _, f2 = pipeline(toy(perturb_after=origin_idx))
            alt = f2.set_index(["sensor_id", "date"])
            past = base.index.get_level_values("date") <= origin
            cols = [*FEATURE_COLUMNS, TARGET]
            b, a = base[past][cols], alt.loc[base[past].index, cols]
            both = b.notna().all(axis=1) & a.notna().all(axis=1)
            kept = base[past][INPUT_OK_COL].to_numpy() == 1
            np.testing.assert_allclose(b[kept & both.to_numpy()].to_numpy(float), a[kept & both.to_numpy()].to_numpy(float))
            # detector sanity: WITHOUT the mask, interpolated rows near the origin DO change (look-ahead exists)
            diff = ~np.isclose(b[both].to_numpy(float), a[both].to_numpy(float)).all(axis=1)
            unmasked_changed_total += int(diff.sum())
        self.assertGreater(unmasked_changed_total, 0, "test cannot detect look-ahead: nothing changed without the mask")

    # TEST 2 - lookback boundary ------------------------------------------------------------------
    def test_2_lookback_boundary(self) -> None:
        g = self.feats[self.feats["sensor_id"] == "S1"].reset_index(drop=True)
        i = 100  # single imputed day
        self.assertEqual(int(self.clean[self.clean["sensor_id"] == "S1"]["depth_imputed"].iloc[i]), 1)
        ok = g[INPUT_OK_COL].to_numpy()
        self.assertEqual(ok[i - 1], 1)                                 # before the imputed day: eligible
        self.assertTrue((ok[i:i + GW_LOOKBACK_DAYS] == 0).all())        # i .. i+30 depend on it (lag_30) -> excluded
        self.assertEqual(ok[i + GW_LOOKBACK_DAYS], 1)                  # i+31: fully outside the look-back -> eligible again
        self.assertEqual(GW_LOOKBACK_DAYS, 31)

    # TEST 3 - XGBoost masks, all splits, both horizons ---------------------------------------------
    def test_3_xgb_masks(self) -> None:
        for h in (30, 90):
            parts = split_rows(self.feats, h, self.splits)
            self.assertGreater(len(parts["train"]), 100)
            for name, d in parts.items():
                self.assertGreater(len(d), 0, f"{name} empty at h={h}")
                self.assertTrue((d[INPUT_OK_COL] == 1).all())
                tgt = [(s, t + pd.Timedelta(days=h)) for s, t in zip(d["sensor_id"], d["date"])]
                self.assertTrue((self.imp.reindex(tgt).to_numpy() == 0).all(), f"imputed target in {name} h={h}")
                for back in (0, 1, 3, 7, 14, 30):
                    hist = [(s, t - pd.Timedelta(days=back)) for s, t in zip(d["sensor_id"], d["date"])]
                    self.assertTrue((self.imp.reindex(hist).to_numpy() == 0).all(), f"imputed history lag {back} in {name} h={h}")

    # TEST 4 - LSTM windows ----------------------------------------------------------------------
    def _windows(self, clean: pd.DataFrame, splits) -> dict:
        d = clean.sort_values(["sensor_id", "date"]).copy()
        d["x"] = d[TARGET]
        d["depth_observed"] = (d["depth_imputed"] == 0).astype(int)
        return make_windows(d, ["x", "rainfall_mm"], "x", 30, 7, splits, observed_col="depth_observed")

    def _assert_windows_clean(self, w: dict, imp: pd.Series) -> None:
        for sid, od in zip(w["meta"]["sensor_id"], w["meta"]["origin_date"]):
            span = [(sid, od + pd.Timedelta(days=k)) for k in range(-29, 8)]   # 30 input days + 7 target days
            self.assertTrue((imp.reindex(span).to_numpy() == 0).all())
        self.assertTrue(w["obs"].all())

    def test_4_lstm_windows(self) -> None:
        w = self._windows(self.clean, self.splits)
        self.assertEqual(set(w["meta"]["split"]), {"train", "val", "test", "discard"} & set(w["meta"]["split"]))
        self._assert_windows_clean(w, self.imp)
        self.assertGreater(len(w["meta"]), 500)

    # TEST 5 - rolling origin ----------------------------------------------------------------------
    def test_5_rolling_origin(self) -> None:
        folds = ro.make_rolling_folds(self.clean["date"], n_folds=3, test_days=60, val_days=90, min_train_days=365)
        long = ro.rolling_xgb(self.feats, self.clean, folds)
        self.assertGreater(len(long), 100)
        ok = self.feats.set_index(["sensor_id", "date"])[INPUT_OK_COL]
        o = list(zip(long["sensor_id"], pd.to_datetime(long["origin_date"])))
        t = list(zip(long["sensor_id"], pd.to_datetime(long["target_date"])))
        self.assertTrue((self.imp.reindex(o).to_numpy() == 0).all(), "imputed rolling origin")
        self.assertTrue((self.imp.reindex(t).to_numpy() == 0).all(), "imputed rolling target")
        self.assertTrue((ok.reindex(o).to_numpy() == 1).all(), "imputed value inside rolling input history")
        for f in folds:  # LSTM windows used by every fold
            self._assert_windows_clean(self._windows(self.clean, f.splits), self.imp)

    # TEST 6 - accounting --------------------------------------------------------------------------
    def test_6_accounting(self) -> None:
        for h in (30, 90):
            st: dict = {}
            parts = split_rows(self.feats, h, self.splits, stats=st)
            for k in ("train", "val", "test"):
                self.assertEqual(st[k]["retained"] + st[k]["dropped_imputed"], st[k]["candidate"])
                self.assertEqual(st[k]["retained"], len(parts[k]))
            self.assertGreater(sum(st[k]["dropped_imputed"] for k in st), 0)
        w = self._windows(self.clean, self.splits)
        tot_ret = 0
        for k, c in w["stats"].items():
            self.assertEqual(c["retained"] + c["dropped_imputed"], c["candidate"])
            tot_ret += c["retained"]
        self.assertEqual(tot_ret, len(w["meta"]))
        self.assertGreater(sum(c["dropped_imputed"] for c in w["stats"].values()), 0)


if __name__ == "__main__":
    unittest.main()
