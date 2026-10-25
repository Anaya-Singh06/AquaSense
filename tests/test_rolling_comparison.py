"""Lightweight tests for the all-model rolling-origin comparison (tiny synthetic data, seconds each, no real data, no GPU).

Run without PyTorch / LightGBM: the sequence-model scoring is exercised through a torch-free stand-in that uses the REAL ``make_windows``
and the REAL ``rolling_sequence`` / ``_score_sequence_windows``; XGBoost falls back to the labelled sklearn stand-in. Tests that need
``torch`` or ``lightgbm`` are SKIPPED when the package is missing (they run on the Windows machine that has both).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests._helpers import ROOT, allow_xgb_standin, has

allow_xgb_standin()
from src.config import TARGET  # noqa: E402
from src.evaluation import rolling_comparison as rc  # noqa: E402
from src.evaluation import rolling_origin as ro  # noqa: E402
from src.models.xgboost_forecaster import XGBForecaster, split_rows  # noqa: E402
from src.preprocessing.features import FEATURE_COLUMNS, INPUT_OK_COL  # noqa: E402
from src.preprocessing.windows import make_windows  # noqa: E402
from tests.test_leakage_masking import pipeline, toy  # noqa: E402


# ------------------------------------------------------------------------------------------------ helpers
def _long(model: str, keys: list[tuple], h: int = 7, noise: float = 0.0, seed: int = 0, y_model_shift: float = 0.0) -> pd.DataFrame:
    """Synthetic long frame; truth / baselines are deterministic functions of the key so models agree on shared keys."""
    rng = np.random.default_rng(seed)
    f, s, o = zip(*keys)
    f, s, o = np.array(f), np.array(s), pd.to_datetime(list(o))
    code = np.array([sum(ord(ch) for ch in str(x)) % 17 for x in s], float)                  # deterministic per station
    days = (o - pd.Timestamp("2020-01-01")).days.to_numpy().astype(float)
    base = 10.0 + 5.0 * code + 0.01 * days                                                   # a function of the KEY only
    y = base + np.sin(days / 3.0)                                                            # so every model agrees on shared keys
    return pd.DataFrame({"fold": f, "model": model, "horizon": h, "sensor_id": s, "origin_date": o, "target_date": o + pd.Timedelta(days=h),
                         "y_true": y + rng.normal(0, noise, len(f)), "y_model": y + y_model_shift, "lo": np.nan, "hi": np.nan,
                         "y_persistence": base, "y_seasonal_naive": base + 0.5})


def _keys(station_ids, days, fold=0):
    return [(fold, s, pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)) for s in station_ids for d in days]


class FakeSequenceModel:
    """Torch-free stand-in with the interface ``rolling_sequence`` needs; windows come from the REAL ``make_windows``."""

    def __init__(self, h: int, input_len: int = 30) -> None:
        self.h, self.input_len = h, input_len
        self.history = [1]
        self.best_epoch, self.best_val_loss, self.persistence_val_loss, self.device_info = 1, 0.1, 0.2, {"device": "fake"}
        self.fit_args: dict = {}

    def fit(self, clean: pd.DataFrame, splits) -> dict:
        c = clean.copy()
        c["depth_observed"] = (c["depth_imputed"] == 0).astype(int)
        w = make_windows(c, [TARGET], TARGET, self.input_len, self.h, splits, observed_col="depth_observed")
        self.window_stats = w["stats"]
        self.fit_args = {"train_end": splits.train_end, "val_end": splits.val_end}
        return w

    def _predict_scaled(self, X: np.ndarray) -> np.ndarray:                     # persistence + a constant offset
        return np.repeat(X[:, -1, :1], self.h, axis=1) + 0.05

    @staticmethod
    def _inv_depth(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float)


# ------------------------------------------------------------------------------------------------ sample alignment
class TestCommonSamples(unittest.TestCase):
    def test_intersection_identical_baselines_and_accounting(self) -> None:
        a = _long("lightgbm", _keys(["S1", "S2"], range(0, 100)))
        b = _long("patchtst", _keys(["S1", "S2"], range(20, 120)), y_model_shift=0.1).assign(  # float32-style round-off on shared columns
            y_true=lambda d: d["y_true"] + 2e-6, y_persistence=lambda d: d["y_persistence"] - 3e-6)
        c = _long("xgboost", _keys(["S1", "S2"], range(10, 110)))
        c.loc[c.index[:5], "y_model"] = np.nan                                       # non-finite forecast -> not eligible
        rows, acct = rc.common_samples(pd.concat([a, b, c], ignore_index=True), 7)
        # common days: 20..99 for both stations = 80 days, minus the 5 NaN rows of xgboost (days 10..14 -> outside the range) = 80*2
        self.assertEqual(acct["n_common"], 160)
        self.assertEqual(sorted(acct["models"]), ["lightgbm", "patchtst", "xgboost"])
        self.assertEqual(set(rows.groupby("model").size()), {160})
        keyed = {m: g.set_index(rc.KEY).sort_index() for m, g in rows.groupby("model")}
        for m in ("patchtst", "xgboost"):                                            # EXACTLY identical truth and baselines across models
            for col in ("y_true", "y_persistence", "y_seasonal_naive"):
                np.testing.assert_array_equal(keyed["lightgbm"][col].to_numpy(), keyed[m][col].to_numpy())
        self.assertAlmostEqual(acct["per_model"]["patchtst"]["fraction_of_own_kept"], 80 * 2 / 200)

    def test_rows_without_seasonal_naive_are_dropped_for_all_models(self) -> None:
        a, b = _long("lightgbm", _keys(["S1"], range(50))), _long("patchtst", _keys(["S1"], range(50)))
        a.loc[a.index[:7], "y_seasonal_naive"] = np.nan
        b.loc[b.index[:7], "y_seasonal_naive"] = np.nan                              # the baseline is a property of the KEY: NaN for every model
        rows, acct = rc.common_samples(pd.concat([a, b], ignore_index=True), 7)
        self.assertEqual(acct["n_common"], 43)
        self.assertEqual(set(rows.groupby("model").size()), {43})
        rows2, acct2 = rc.common_samples(pd.concat([a, b], ignore_index=True), 7, require_seasonal=False)
        self.assertEqual(acct2["n_common"], 50)

    def test_disagreeing_truth_is_rejected(self) -> None:
        a, b = _long("lightgbm", _keys(["S1"], range(30))), _long("patchtst", _keys(["S1"], range(30)))
        b.loc[b.index[3], "y_true"] += 0.5                                           # same (fold, station, origin) but a different target
        with self.assertRaises(ValueError):
            rc.common_samples(pd.concat([a, b], ignore_index=True), 7)

    def test_duplicate_rows_rejected(self) -> None:
        a = _long("lightgbm", _keys(["S1"], range(10)))
        with self.assertRaises(ValueError):
            rc.common_samples(pd.concat([a, a.iloc[:2]], ignore_index=True), 7)

    def test_tabular_only_set_added_when_sequence_models_shrink_the_sample(self) -> None:
        long = pd.concat([_long("xgboost", _keys(["S1", "S2"], range(100))), _long("lightgbm", _keys(["S1", "S2"], range(100))),
                          _long("patchtst", _keys(["S1", "S2"], range(60, 100)))], ignore_index=True)
        comp = rc.build_comparison(long)
        self.assertEqual(comp["comparisons"]["all_models_7d"]["n"], 80)
        self.assertEqual(comp["comparisons"]["tabular_only_7d"]["n"], 200)


# ------------------------------------------------------------------------------------------------ metrics
class TestMetrics(unittest.TestCase):
    def test_pooled_nse_is_not_mean_station_nse(self) -> None:
        rng = np.random.default_rng(0)
        parts = []
        for k, mean in enumerate((10.0, 50.0, 90.0, 130.0)):                         # big between-station differences, small within-station swings
            d = np.arange(200)
            y = mean + 0.5 * np.sin(d / 9.0)
            parts.append(pd.DataFrame({"fold": 0, "model": "m", "horizon": 7, "sensor_id": f"S{k}", "origin_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(d, "D"),
                                       "y_true": y, "y_model": mean + rng.normal(0, 0.05, 200) + 0.3, "y_persistence": mean + 0.3, "y_seasonal_naive": mean + 0.3,
                                       "lo": np.nan, "hi": np.nan, "target_date": pd.NaT}))
        g = pd.concat(parts, ignore_index=True)
        r = rc.score_predictor(g, "y_model")
        self.assertGreater(r["nse_pooled"], 0.999)                                    # looks excellent ...
        self.assertLess(r["nse_mean_station"], 0.0)                                   # ... but is worse than the per-station mean
        self.assertNotAlmostEqual(r["nse_pooled"], r["nse_mean_station"], places=2)
        self.assertEqual(r["n_stations_scored"], 4)

    def test_station_wins_losses_ties_on_identical_rows(self) -> None:
        d = np.arange(40)
        rows = []
        for sid, model_err in (("WIN", 0.1), ("LOSS", 1.0), ("TIE", None)):
            y = 10 + np.sin(d / 5.0)
            pers = y + 0.5
            mod = pers.copy() if model_err is None else y + model_err * (0.2 if sid == "WIN" else 1.0)
            rows.append(pd.DataFrame({"fold": 0, "model": "m", "horizon": 7, "sensor_id": sid, "origin_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(d, "D"),
                                      "target_date": pd.NaT, "y_true": y, "y_model": mod, "y_persistence": pers, "y_seasonal_naive": pers, "lo": np.nan, "hi": np.nan}))
        so = rc.station_outcomes(pd.concat(rows, ignore_index=True))
        out = dict(zip(so["sensor_id"], so["outcome_vs_persistence"]))
        self.assertEqual(out, {"WIN": "win", "LOSS": "loss", "TIE": "tie"})
        comp = rc.compare_horizon(pd.concat(rows, ignore_index=True), 7, "all_models", ["m"])
        v = comp["predictors"]["m"]["vs_persistence"]
        self.assertEqual((v["stations_win"], v["stations_loss"], v["stations_tie"], v["stations_compared"]), (1, 1, 1, 3))

    def test_by_fold_counts_and_values_match_manual(self) -> None:
        a = pd.concat([_long("lightgbm", _keys(["S1", "S2"], range(40), fold=0), y_model_shift=0.2),
                       _long("lightgbm", _keys(["S1", "S2"], range(40, 90), fold=1), y_model_shift=0.4)], ignore_index=True)
        b = a.assign(model="patchtst", y_model=a["y_model"] + 0.1)
        long = pd.concat([a, b], ignore_index=True)
        comp = rc.build_comparison(long)["comparisons"]["all_models_7d"]
        self.assertEqual(sum(comp["by_fold"][f]["lightgbm"]["n"] for f in (0, 1)), comp["n"])
        self.assertAlmostEqual(comp["by_fold"][0]["lightgbm"]["rmse"], 0.2, places=9)
        self.assertAlmostEqual(comp["by_fold"][1]["patchtst"]["rmse"], 0.5, places=9)
        self.assertAlmostEqual(comp["by_fold"][0]["persistence"]["rmse"], float(np.sqrt(np.mean((a[a.fold == 0]["y_true"] - a[a.fold == 0]["y_persistence"]) ** 2))), places=9)
        t = rc.by_fold_table(rc.build_comparison(long))
        self.assertEqual(set(t["predictor"]), {"LightGBM", "PatchTST", "Persistence", "Seasonal naive"})

    def test_min_station_n_excludes_short_stations_from_station_metrics_only(self) -> None:
        long = pd.concat([_long("m", _keys(["BIG"], range(60))), _long("m", _keys(["TINY"], range(5)))], ignore_index=True)
        r = rc.score_predictor(long, "y_model")
        self.assertEqual((r["n"], r["n_stations"], r["n_stations_scored"]), (65, 2, 1))


# ------------------------------------------------------------------------------------------------ alignment on toy data
class TestRollingAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = toy(n=1500)
        cls.clean, cls.feats = pipeline(cls.raw)
        cls.imp = cls.clean.set_index(["sensor_id", "date"])["depth_imputed"]
        cls.depth = cls.clean.set_index(["sensor_id", "date"])[TARGET]
        cls.folds = ro.make_rolling_folds(cls.clean["date"], n_folds=2, test_days=60, val_days=90, min_train_days=365)
        cls.h = 7
        cls.tab = ro.rolling_xgb(cls.feats, cls.clean, cls.folds, (cls.h,))
        cls.seq = ro.rolling_sequence(cls.clean, cls.folds, (cls.h,), lambda h: FakeSequenceModel(h), "patchtst")

    def test_both_families_use_the_same_target_date_window(self) -> None:
        for f in self.folds:
            for long in (self.tab, self.seq):
                g = long[long["fold"] == f.index]
                self.assertFalse(g.empty)
                self.assertTrue((g["target_date"] > f.splits.val_end).all() and (g["target_date"] <= f.test_end).all())
            # first scored target day is the day after the fold origin for BOTH families (was origin + h for sequence models)
            self.assertEqual(self.tab[self.tab["fold"] == f.index]["target_date"].min(), f.splits.val_end + pd.Timedelta(days=1))
            self.assertEqual(self.seq[self.seq["fold"] == f.index]["target_date"].min(), f.splits.val_end + pd.Timedelta(days=1))

    def test_legacy_flag_reproduces_old_start_date(self) -> None:
        legacy = ro.rolling_sequence(self.clean, self.folds, (self.h,), lambda h: FakeSequenceModel(h), "patchtst", include_straddling_origins=False)
        for f in self.folds:
            g = legacy[legacy["fold"] == f.index]
            self.assertTrue((g["origin_date"] >= f.splits.val_end).all())
            self.assertEqual(g["target_date"].min(), f.splits.val_end + pd.Timedelta(days=self.h))
            self.assertGreater(len(self.seq[self.seq["fold"] == f.index]), len(g))

    def test_masks_every_scored_row_uses_only_real_observations(self) -> None:
        for name, long in (("tabular", self.tab), ("sequence", self.seq)):
            o = list(zip(long["sensor_id"], pd.to_datetime(long["origin_date"])))
            t = list(zip(long["sensor_id"], pd.to_datetime(long["target_date"])))
            self.assertTrue((self.imp.reindex(o).to_numpy() == 0).all(), f"{name}: interpolated origin value")
            self.assertTrue((self.imp.reindex(t).to_numpy() == 0).all(), f"{name}: interpolated target treated as observed")
            np.testing.assert_allclose(self.depth.reindex(t).to_numpy(), long["y_true"].to_numpy(), atol=1e-9)         # target is the station's own series
            np.testing.assert_allclose(self.depth.reindex(o).to_numpy(), long["y_persistence"].to_numpy(), atol=1e-9)  # persistence = origin-day depth
        # sequence windows additionally need every intermediate target day and the whole context observed (respecting station boundaries)
        for r in self.seq.sample(60, random_state=1).itertuples():
            days = pd.date_range(r.origin_date - pd.Timedelta(days=29), r.target_date)
            idx = pd.MultiIndex.from_arrays([[r.sensor_id] * len(days), days])
            self.assertTrue((self.imp.reindex(idx).to_numpy() == 0).all(), "gap inside a sequence window")

    def test_metric_alignment_between_families_on_common_rows(self) -> None:
        long = pd.concat([self.tab, self.seq], ignore_index=True)
        comp = rc.build_comparison(long)
        c = comp["comparisons"][f"all_models_{self.h}d"]
        self.assertGreater(c["n"], 100)
        self.assertEqual(set(c["models"]), {"xgboost", "patchtst"})
        rows, _ = rc.common_samples(long, self.h)
        a = rows[rows["model"] == "xgboost"].set_index(rc.KEY).sort_index()
        b = rows[rows["model"] == "patchtst"].set_index(rc.KEY).sort_index()
        self.assertTrue(a.index.equals(b.index))
        np.testing.assert_array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy())
        np.testing.assert_array_equal(a["y_persistence"].to_numpy(), b["y_persistence"].to_numpy())
        # persistence scored once on the common rows equals persistence computed by hand on them
        want = float(np.sqrt(np.mean((a["y_true"] - a["y_persistence"]) ** 2)))
        self.assertAlmostEqual(c["predictors"]["persistence"]["rmse"], want, places=12)
        self.assertEqual(c["predictors"]["xgboost"]["n"], c["predictors"]["patchtst"]["n"])
        self.assertLess(c["n"], min(len(self.tab), len(self.seq)) + 1)               # intersection can only shrink
        # the fake model is persistence + 0.05 m: its pooled RMSE against truth must reflect exactly that on these rows
        np.testing.assert_allclose(b["y_model"].to_numpy() - b["y_persistence"].to_numpy(), 0.05, atol=1e-9)

    def test_folds_split_chronologically_for_tabular_models(self) -> None:
        for f in self.folds:
            p = split_rows(self.feats, self.h, f.splits)
            self.assertTrue((p["train"]["date"] + pd.Timedelta(days=self.h) <= f.splits.train_end).all())
            td = p["val"]["date"] + pd.Timedelta(days=self.h)
            self.assertTrue(((td > f.splits.train_end) & (td <= f.splits.val_end)).all())
            self.assertTrue((p["test"]["date"] + pd.Timedelta(days=self.h) > f.splits.val_end).all())

    def test_future_data_cannot_reach_a_fold_tabular(self) -> None:
        """Poison every observation after fold 0's origin: train/val rows and the fitted model's val predictions must not move."""
        f = self.folds[0]
        poisoned = self.raw.copy()
        m = pd.to_datetime(poisoned["date"]) > f.splits.val_end
        poisoned.loc[m & poisoned[TARGET].notna(), TARGET] += 50.0
        poisoned.loc[m, "rainfall_mm"] += 100.0
        c2, f2 = pipeline(poisoned)
        p1, p2 = split_rows(self.feats, self.h, f.splits), split_rows(f2, self.h, f.splits)
        used = [*FEATURE_COLUMNS, f"target_{self.h}d", f"target_{self.h}d_observed", INPUT_OK_COL, "sensor_id", "date"]   # other horizons' targets look further ahead
        for part in ("train", "val"):
            pd.testing.assert_frame_equal(p1[part][used].reset_index(drop=True), p2[part][used].reset_index(drop=True))
        m1, m2 = XGBForecaster(self.h).fit(p1["train"], p1["val"]), XGBForecaster(self.h).fit(p2["train"], p2["val"])
        np.testing.assert_allclose(m1.predict_level(p1["val"]), m2.predict_level(p2["val"]), atol=1e-6)

    def test_future_data_cannot_reach_a_fold_sequence_windows(self) -> None:
        f = self.folds[0]
        poisoned = self.raw.copy()
        m = pd.to_datetime(poisoned["date"]) > f.splits.val_end
        poisoned.loc[m & poisoned[TARGET].notna(), TARGET] += 50.0
        c2, _ = pipeline(poisoned)
        w1, w2 = FakeSequenceModel(self.h).fit(self.clean, f.splits), FakeSequenceModel(self.h).fit(c2, f.splits)
        for split in ("train", "val"):
            a, b = w1["meta"]["split"].to_numpy() == split, w2["meta"]["split"].to_numpy() == split
            np.testing.assert_array_equal(w1["X"][a], w2["X"][b])
            np.testing.assert_array_equal(w1["Y"][a], w2["Y"][b])
        self.assertTrue((w1["meta"][w1["meta"]["split"] == "train"]["origin_date"] + pd.Timedelta(days=self.h) <= f.splits.train_end).all())

    def test_each_fold_gets_a_new_model_with_its_own_boundaries(self) -> None:
        made: list[FakeSequenceModel] = []

        def factory(h: int) -> FakeSequenceModel:
            made.append(FakeSequenceModel(h))
            return made[-1]
        ro.rolling_sequence(self.clean, self.folds, (self.h,), factory, "patchtst")
        self.assertEqual(len(made), len(self.folds))
        self.assertEqual([m.fit_args["train_end"] for m in made], [f.splits.train_end for f in self.folds])
        self.assertEqual([m.fit_args["val_end"] for m in made], [f.splits.val_end for f in self.folds])

    def test_diagnostics_are_collected(self) -> None:
        diag: list[dict] = []
        ro.rolling_sequence(self.clean, self.folds, (self.h,), lambda h: FakeSequenceModel(h), "patchtst", diagnostics=diag)
        self.assertEqual([d["fold"] for d in diag], [f.index for f in self.folds])
        self.assertTrue(all(d["status"] == "ok" and d["n_train"] > 0 and d["n_test_scored"] > 0 for d in diag))
        diag2: list[dict] = []
        ro.rolling_xgb(self.feats, self.clean, self.folds, (self.h,), diagnostics=diag2)
        self.assertTrue(all(d["status"] == "ok" and d["n_test_scored"] > 0 for d in diag2))


# ------------------------------------------------------------------------------------------------ optional: torch / lightgbm
@unittest.skipUnless(has("torch"), "PyTorch not installed")
class TestTorchModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = toy(n=1500)
        cls.clean, cls.feats = pipeline(cls.raw)
        cls.folds = ro.make_rolling_folds(cls.clean["date"], n_folds=2, test_days=60, val_days=90, min_train_days=365)

    def test_scalers_are_fitted_on_the_fold_training_block_only(self) -> None:
        from src.models.lstm_forecaster import LSTMForecaster
        f = self.folds[0]
        poisoned = self.clean.copy()
        late = poisoned["date"] > f.splits.train_end
        poisoned.loc[late, TARGET] += 100.0
        a, b = LSTMForecaster(), LSTMForecaster()
        a._fit_scalers(self.clean, f.splits.train_end)
        b._fit_scalers(poisoned, f.splits.train_end)
        for k in a.scalers:
            np.testing.assert_allclose(a.scalers[k].mean_, b.scalers[k].mean_)
            np.testing.assert_allclose(a.scalers[k].scale_, b.scalers[k].scale_)
        real = self.clean[(self.clean["date"] <= f.splits.train_end) & (self.clean["depth_imputed"] == 0)][TARGET]
        self.assertAlmostEqual(float(a.scalers["depth"].mean_[0]), float(real.mean()), places=9)

    def test_rolling_patchtst_uses_shared_scoring_and_records_diagnostics(self) -> None:
        from src.models.patchtst_forecaster import PatchTSTConfig
        fast = PatchTSTConfig(context_length=32, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.1,
                              head_dropout=0.0, batch_size=64, eval_batch_size=128, max_epochs=1, patience=1, amp="off", seed=7)
        diag: list[dict] = []
        long = ro.rolling_patchtst(self.clean, self.folds, (7,), fast, max_epochs=1, device="cpu", diagnostics=diag)
        self.assertFalse(long.empty)
        for f in self.folds:
            g = long[long["fold"] == f.index]
            self.assertTrue((g["target_date"] > f.splits.val_end).all() and (g["target_date"] <= f.test_end).all())
            self.assertEqual(g["target_date"].min(), f.splits.val_end + pd.Timedelta(days=1))
        self.assertTrue(all(d["status"] == "ok" and d["best_epoch"] == 1 for d in diag))
        tab = ro.rolling_xgb(self.feats, self.clean, self.folds, (7,))
        c = rc.build_comparison(pd.concat([tab, long], ignore_index=True))["comparisons"]["all_models_7d"]
        self.assertGreater(c["n"], 50)

    def test_rolling_lstm_still_works_through_the_shared_helper(self) -> None:
        long = ro.rolling_lstm(self.clean, self.folds, max_epochs=1)
        self.assertEqual(set(long["model"]), {"lstm"})
        self.assertEqual(set(long["horizon"]), {7})
        self.assertTrue((long["target_date"] <= self.folds[-1].test_end).all())


@unittest.skipUnless(has("lightgbm"), "lightgbm not installed")
class TestLightGBMIntegration(unittest.TestCase):
    def test_lightgbm_horizon_7_aligns_with_xgboost_rows(self) -> None:
        from src.models.lightgbm_forecaster import load_lgbm_config
        cfg = {"params": {**load_lgbm_config()["params"], "n_estimators": 30, "n_jobs": 1}, "early_stopping_rounds": 5}
        clean, feats = pipeline(toy(n=1500))
        folds = ro.make_rolling_folds(clean["date"], n_folds=2, test_days=60, val_days=90, min_train_days=365)
        lg = ro.rolling_lightgbm(feats, clean, folds, horizons=(7,), config=cfg)
        xg = ro.rolling_xgb(feats, clean, folds, (7,))
        self.assertEqual(set(lg["model"]), {"lightgbm"})
        pd.testing.assert_frame_equal(lg[rc.KEY].reset_index(drop=True), xg[rc.KEY].reset_index(drop=True))   # identical eligible rows/order
        c = rc.build_comparison(pd.concat([lg, xg], ignore_index=True))["comparisons"]["all_models_7d"]
        self.assertEqual(c["predictors"]["lightgbm"]["n"], c["predictors"]["xgboost"]["n"])


# ------------------------------------------------------------------------------------------------ script
class TestScript(unittest.TestCase):
    def _run(self, out: Path, *extra: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        if not has("xgboost"):
            env["AQUASENSE_ALLOW_SKLEARN_FALLBACK"] = "1"
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_rolling_comparison.py"), "--source", "synthetic", "--models", "xgboost",
                               "--horizons", "7", "30", "--limit-folds", "2", "--out-dir", str(out), *extra],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)

    def test_end_to_end_refuses_overwrite_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            r = self._run(out)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for name in ("comparison_common_samples.csv", "comparison_by_fold.csv", "comparison_station_level.csv", "comparison_sample_accounting.csv",
                         "evaluation_own_samples.json", "run_manifest.json", "RESULTS_GUIDE.md", "forecasts_long.csv.gz", "fold_definitions.csv"):
                self.assertTrue((out / name).exists(), name)
            tab = pd.read_csv(out / "comparison_common_samples.csv")
            self.assertEqual(set(tab["predictor"]), {"XGBoost", "Persistence", "Seasonal naive"})
            self.assertEqual(set(tab["horizon_days"]), {7, 30})
            man = json.loads((out / "run_manifest.json").read_text())
            self.assertTrue(any("SMOKE TEST" in x for x in man["limitations"]))
            self.assertTrue(any("Ridge" in x for x in man["limitations"]))
            r2 = self._run(out)                                                      # existing results are never overwritten
            self.assertNotEqual(r2.returncode, 0)
            self.assertIn("already contains files", r2.stderr + r2.stdout)
            r3 = self._run(out, "--resume")                                          # resume reuses finished units and reproduces the table
            self.assertEqual(r3.returncode, 0, r3.stderr[-2000:])
            self.assertIn("[reuse]", r3.stderr + r3.stdout)
            pd.testing.assert_frame_equal(tab, pd.read_csv(out / "comparison_common_samples.csv"))


if __name__ == "__main__":
    unittest.main()
