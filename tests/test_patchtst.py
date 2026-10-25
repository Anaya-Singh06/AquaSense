"""Lightweight PatchTST tests (tiny networks, tiny synthetic data, CPU, a few seconds each).

Tests that need PyTorch are SKIPPED when it is not installed; config-file / plumbing tests that do not import torch always run.
No test trains on the real dataset.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests._helpers import ROOT, allow_xgb_standin, has

allow_xgb_standin()
from src.evaluation import rolling_origin as ro  # noqa: E402
from src.inference import compute_all_forecasts  # noqa: E402
from src.preprocessing.splits import make_split_dates  # noqa: E402
from tests.test_leakage_masking import imputed_lookup, pipeline, toy  # noqa: E402

HAS_TORCH = has("torch")
CONFIG_FILE = ROOT / "config" / "patchtst_config.json"

if HAS_TORCH:
    import torch  # noqa: E402

    from src.config import TARGET  # noqa: E402
    from src.models.patchtst_forecaster import (  # noqa: E402
        PatchTSTConfig, PatchTSTForecaster, PatchTSTNet, artifact_paths, load_patchtst_config, replace_config)

    # tiny, fast settings for tests only (context 32 days, 7 patches of 8 days every 4 days)
    FAST = PatchTSTConfig(context_length=32, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.1,
                          head_dropout=0.0, batch_size=64, eval_batch_size=128, max_epochs=2, patience=2, amp="off", seed=7)


class TestConfigAndPlumbing(unittest.TestCase):
    """No PyTorch needed."""

    def test_config_file_contents(self) -> None:
        raw = json.loads(CONFIG_FILE.read_text())
        for k in ("context_length", "patch_len", "stride", "d_model", "n_heads", "n_layers", "dropout"):
            self.assertIn(k, raw["model"])
        for k in ("lr", "batch_size", "max_epochs", "patience"):
            self.assertIn(k, raw["training"])
        self.assertEqual(raw["horizons"], [7])

    def test_sequence_models_hook_and_existing_keys_untouched(self) -> None:
        class Stub:
            def forecast_latest(self, clean):          # LSTM-style interface; "S1" is skipped
                return {"S2": [{"date": "2020-01-01", "depth_m_bgl": 1.0}]}
        clean, feats = pipeline(toy(n=400))
        out = compute_all_forecasts(clean, feats, None, None, None, sequence_models={"patchtst_7d": Stub(), "patchtst_30d": None})
        self.assertEqual(set(out), {"S1", "S2"})
        self.assertIsNone(out["S1"]["patchtst_7d"])
        self.assertEqual(out["S2"]["patchtst_7d"], [{"date": "2020-01-01", "depth_m_bgl": 1.0}])
        self.assertNotIn("patchtst_30d", out["S1"])                        # None model -> no key
        self.assertTrue(all(v["xgb_30d"] is None and v["lstm_7d"] is None for v in out.values()))

    def test_metrics_table_label(self) -> None:
        long = pd.DataFrame({"fold": 0, "model": "patchtst", "horizon": 7, "sensor_id": "A", "y_true": np.arange(20.0),
                             "y_model": np.arange(20.0) + 0.1, "y_persistence": np.arange(20.0) + 1, "y_seasonal_naive": np.arange(20.0) + 2,
                             "lo": np.nan, "hi": np.nan})
        t = ro.metrics_table(ro.summarize(long), "Test")
        self.assertEqual(set(t["Model"]), {"PatchTST", "Persistence", "Seasonal naive"})

    @unittest.skipIf(HAS_TORCH, "torch installed")
    def test_clear_error_without_torch(self) -> None:
        with self.assertRaises(ImportError):
            import src.models.patchtst_forecaster  # noqa: F401


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestConfigLoading(unittest.TestCase):
    def test_shipped_config_loads_and_validates(self) -> None:
        r = load_patchtst_config()
        self.assertIsInstance(r["config"], PatchTSTConfig)
        self.assertEqual(r["horizons"], [7])
        self.assertEqual(r["config"].n_patches, (96 - 16) // 8 + 1)

    def test_invalid_settings_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PatchTSTConfig(context_length=8, patch_len=16).validate()
        with self.assertRaises(ValueError):
            PatchTSTConfig(d_model=30, n_heads=4).validate()
        with self.assertRaises(ValueError):
            PatchTSTConfig(amp="maybe").validate()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"model": {"not_a_setting": 1}}))
            with self.assertRaises(ValueError):
                load_patchtst_config(p)

    def test_custom_config_file_and_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"horizons": [7, 30], "model": {"patch_len": 8, "stride": 4, "context_length": 32},
                                     "training": {"max_epochs": 3, "amp": False}, "rolling_origin": {"max_epochs": 2}}))
            r = load_patchtst_config(p)
            self.assertEqual((r["horizons"], r["config"].max_epochs, r["config"].amp, r["rolling_max_epochs"]), ([7, 30], 3, "off", 2))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestPatchTSTNet(unittest.TestCase):
    def test_init_and_shapes(self) -> None:
        net = PatchTSTNet(n_channels=7, context_length=96, horizon=7)
        self.assertEqual(net.n_patches, 11)
        x = torch.randn(4, 96, 7)
        self.assertEqual(tuple(net(x).shape), (4, 7))
        self.assertEqual(tuple(net.patchify(x).shape), (4 * 7, 11, 16))
        self.assertEqual(tuple(PatchTSTNet(1, 40, 30, patch_len=8, stride=4)(torch.randn(3, 40, 1)).shape), (3, 30))
        with self.assertRaises(ValueError):
            net(torch.randn(4, 95, 7))
        with self.assertRaises(ValueError):
            net(torch.randn(4, 96, 6))

    def test_patching_is_overlapping_and_ends_on_latest_day(self) -> None:
        net = PatchTSTNet(2, context_length=100, horizon=3, patch_len=16, stride=8)    # (100-16) % 8 = 4 oldest days dropped
        self.assertEqual((net.trim, net.n_patches), (4, 11))
        x = torch.arange(100.0).view(1, 100, 1).repeat(1, 1, 2) + torch.tensor([0.0, 1000.0])
        p = net.patchify(x)
        self.assertTrue(torch.equal(p[0, 0], torch.arange(4.0, 20.0)))                  # first patch: days 4..19 of channel 0
        self.assertTrue(torch.equal(p[0, 1], torch.arange(12.0, 28.0)))                 # stride 8 -> overlap of 8 days
        self.assertEqual(float(p[0, -1, -1]), 99.0)                                     # last patch ends on the latest day
        self.assertTrue(torch.equal(p[1, 0], torch.arange(4.0, 20.0) + 1000.0))         # channel 1 is its own sequence

    def test_untrained_network_equals_persistence(self) -> None:
        for revin in (True, False):
            net = PatchTSTNet(3, 32, 5, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, revin=revin).eval()
            x = torch.randn(6, 32, 3)
            with torch.no_grad():
                np.testing.assert_allclose(net(x).numpy(), x[:, -1, :1].repeat(1, 5).numpy(), atol=1e-6)

    def test_calendar_channels_not_instance_normalised(self) -> None:
        m = PatchTSTForecaster(7, FAST, "cpu")
        self.assertEqual(m._build_net().norm_mask.flatten().tolist(), [True] * 5 + [False] * 2)   # x_* normalised, sin/cos not
        self.assertEqual(PatchTSTForecaster(7, replace_config(FAST, use_covariates=False), "cpu").cols, ["x_depth"])

    def test_output_is_finite_and_trainable(self) -> None:
        torch.manual_seed(0)
        net = PatchTSTNet(2, 32, 4, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32)
        x = torch.randn(16, 32, 2)
        y = x[:, -1, :1] + 0.5                                   # target: +0.5 above persistence
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        first = None
        for _ in range(60):
            opt.zero_grad(); loss = ((net(x) - y) ** 2).mean(); loss.backward(); opt.step()
            first = loss.item() if first is None else first
        self.assertTrue(np.isfinite(loss.item()))
        self.assertLess(loss.item(), first)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean, cls.feats = pipeline(toy())
        cls.imp = imputed_lookup(cls.clean)
        cls.splits = make_split_dates(cls.clean["date"])
        cls.model = PatchTSTForecaster(7, FAST, "cpu")
        cls.w = cls.model.fit(cls.clean, cls.splits)

    # ---- training / evaluation -------------------------------------------------------------
    def test_training_ran_and_selected_best_epoch(self) -> None:
        m = self.model
        self.assertGreaterEqual(len(m.history), 1)
        self.assertTrue(all(np.isfinite([r["train_loss"], r["val_loss"]]).all() for r in m.history))
        by_epoch = {r["epoch"]: r["val_loss"] for r in m.history}
        self.assertIn(m.best_epoch, by_epoch)
        self.assertAlmostEqual(m.best_val_loss, by_epoch[m.best_epoch])
        self.assertAlmostEqual(m.best_val_loss, min(by_epoch.values()), places=4)

    def test_evaluate_structure_and_station_metrics(self) -> None:
        r = self.model.evaluate(self.w, "test")
        for k in ("mae", "rmse", "nse_pooled", "nse_per_sensor_mean"):
            self.assertIn(k, r["model"]); self.assertIn(k, r["persistence"])
        self.assertIn("day7_only", r)
        a = r["_arrays"]
        self.assertEqual(a["pred"].shape, a["true"].shape)
        self.assertEqual(a["pred"].shape[1], 7)
        self.assertTrue(np.isfinite(a["pred"]).all())
        self.assertEqual(set(a["meta"]["sensor_id"]), {"S1", "S2"})              # station identity preserved
        with self.assertRaises(ValueError):
            self.model.evaluate(self.w, "nonexistent")

    def test_30_day_horizon_smoke(self) -> None:
        m = PatchTSTForecaster(30, replace_config(FAST, max_epochs=1), "cpu")
        w = m.fit(self.clean, self.splits)
        r = m.evaluate(w, "test")
        self.assertEqual(r["_arrays"]["pred"].shape[1], 30)
        self.assertEqual(r["horizon_days"], 30)

    def test_too_little_data_raises(self) -> None:
        short = self.clean[self.clean["date"] < "2015-03-01"]
        with self.assertRaises(ValueError):
            PatchTSTForecaster(7, FAST, "cpu").fit(short, make_split_dates(short["date"]))

    def test_amp_only_on_cuda(self) -> None:
        self.assertFalse(PatchTSTForecaster(7, replace_config(FAST, amp="on"), "cpu")._use_amp())
        self.assertFalse(PatchTSTForecaster(7, replace_config(FAST, amp="auto"), "cpu")._use_amp())

    # ---- leakage ------------------------------------------------------------------------------
    def test_windows_contain_only_real_observations_and_obey_split_order(self) -> None:
        L, h = FAST.context_length, 7
        meta = self.w["meta"]
        self.assertGreater(len(meta), 500)
        self.assertEqual(self.w["X"].shape[1:], (L, 7))
        for sid, od in zip(meta["sensor_id"], meta["origin_date"]):
            span = [(sid, od + pd.Timedelta(days=k)) for k in range(-(L - 1), h + 1)]       # context days + target days
            self.assertTrue((self.imp.reindex(span).to_numpy() == 0).all(), "interpolated value inside a window")
        self.assertTrue(self.w["obs"].all())
        s = self.splits
        tgt_first, tgt_last = meta["origin_date"] + pd.Timedelta(days=1), meta["origin_date"] + pd.Timedelta(days=h)
        sp = meta["split"]
        self.assertTrue((tgt_last[sp == "train"] <= s.train_end).all())
        self.assertTrue(((tgt_first[sp == "val"] > s.train_end) & (tgt_last[sp == "val"] <= s.val_end)).all())
        self.assertTrue((tgt_first[sp == "test"] > s.val_end).all())
        for name in ("train", "val", "test"):
            self.assertGreater(int((sp == name).sum()), 0, name)
        st = self.model.window_stats
        for c in st.values():
            self.assertEqual(c["retained"] + c["dropped_imputed"], c["candidate"])
        self.assertGreater(sum(c["dropped_imputed"] for c in st.values()), 0)                 # the toy data really has interpolation

    def test_windows_never_cross_station_boundaries(self) -> None:
        L = FAST.context_length
        scaled = self.model._scale(self.clean).set_index(["sensor_id", "date"])["x_depth"]
        meta, X = self.w["meta"], self.w["X"]
        for i in np.linspace(0, len(meta) - 1, 60).astype(int):
            sid, od = meta["sensor_id"].iloc[i], meta["origin_date"].iloc[i]
            dates = pd.date_range(end=od, periods=L, freq="D")
            own = scaled.reindex(pd.MultiIndex.from_arrays([[sid] * L, dates])).to_numpy()
            np.testing.assert_allclose(X[i, :, 0], own, rtol=1e-5, atol=1e-6)
        other = scaled.reindex(pd.MultiIndex.from_arrays([["S2" if meta["sensor_id"].iloc[0] == "S1" else "S1"] * L,
                                                         pd.date_range(end=meta["origin_date"].iloc[0], periods=L, freq="D")])).to_numpy()
        self.assertFalse(np.allclose(X[0, :, 0], other))           # detector sanity: the other station's series differs

    def test_scalers_use_training_period_only(self) -> None:
        cut = (self.splits.train_end - pd.Timestamp("2015-01-01")).days
        clean2, _ = pipeline(toy(perturb_after=cut))                   # data after train_end changed
        m2 = PatchTSTForecaster(7, FAST, "cpu")
        m2._fit_scalers(clean2, self.splits.train_end)
        for k, sc in self.model.scalers.items():
            np.testing.assert_allclose(m2.scalers[k].mean_, sc.mean_)
            np.testing.assert_allclose(m2.scalers[k].scale_, sc.scale_)
        m3 = PatchTSTForecaster(7, FAST, "cpu")
        m3._fit_scalers(clean2, self.clean["date"].max())              # detector sanity: using ALL data would change them
        self.assertFalse(np.isclose(m3.scalers["depth"].mean_, self.model.scalers["depth"].mean_).all())

    def test_forecast_ignores_the_future(self) -> None:
        cutoff_idx = 1250                                              # both stations are gap-free in the 32 days before it
        origin = pd.Timestamp("2015-01-01") + pd.Timedelta(days=cutoff_idx)
        clean2, _ = pipeline(toy(perturb_after=cutoff_idx))            # everything after the origin differs
        a = self.model.forecast_latest(self.clean[self.clean["date"] <= origin])
        b = self.model.forecast_latest(clean2[clean2["date"] <= origin])
        self.assertEqual(set(a), {"S1", "S2"})
        self.assertEqual(a, b)
        later = origin + pd.Timedelta(days=5)                          # detector sanity: once perturbed days enter the context it changes
        a2 = self.model.forecast_latest(self.clean[self.clean["date"] <= later])
        c2 = self.model.forecast_latest(clean2[clean2["date"] <= later])
        self.assertNotEqual([p["depth_m_bgl"] for p in a2["S1"]], [p["depth_m_bgl"] for p in c2["S1"]])

    # ---- inference --------------------------------------------------------------------------------
    def test_forecast_latest_format_and_dates(self) -> None:
        out = self.model.forecast_latest(self.clean)
        self.assertEqual(set(out), {"S1", "S2"})
        last = self.clean["date"].max()
        for pts in out.values():
            self.assertEqual(len(pts), 7)
            self.assertEqual([p["date"] for p in pts], [str((last + pd.Timedelta(days=k)).date()) for k in range(1, 8)])
            self.assertTrue(all(set(p) == {"date", "depth_m_bgl"} and np.isfinite(p["depth_m_bgl"]) for p in pts))
        full = compute_all_forecasts(self.clean, self.feats, None, None, None, sequence_models={"patchtst_7d": self.model})
        self.assertEqual(full["S1"]["patchtst_7d"], out["S1"])

    def test_strict_inference_skips_interpolated_context(self) -> None:
        cutoff = pd.Timestamp("2015-01-01") + pd.Timedelta(days=1310)     # S1 has interpolated days 1300-1301 inside the last 32
        trunc = self.clean[self.clean["date"] <= cutoff]
        self.assertEqual(set(self.model.forecast_latest(trunc)), {"S2"})
        lax = PatchTSTForecaster(7, replace_config(FAST, strict_inference_inputs=False), "cpu")
        lax.net, lax.scalers = self.model.net, self.model.scalers
        self.assertEqual(set(lax.forecast_latest(trunc)), {"S1", "S2"})

    def test_incomplete_context_is_skipped(self) -> None:
        short = self.clean[(self.clean["sensor_id"] != "S1") | (self.clean["date"] <= pd.Timestamp("2015-01-20"))]
        self.assertEqual(set(self.model.forecast_latest(short)), {"S2"})

    # ---- checkpoints ----------------------------------------------------------------------------------
    def test_save_load_roundtrip_and_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            existing = {"lstm_model.pt": b"L", "lstm_scalers.joblib": b"S", "xgb_30d.json": b"X", "xgb_30d_meta.json": b"M", "lgbm_30d.txt": b"G",
                        "lgbm_30d_meta.json": b"H", "anomaly_model.joblib": b"A"}
            for n, b in existing.items():
                (d / n).write_bytes(b)
            self.model.save(d)
            new = sorted(x.name for x in d.iterdir() if x.name not in existing)
            self.assertEqual(new, ["patchtst_7d.pt", "patchtst_7d_meta.json", "patchtst_7d_scalers.joblib"])
            self.assertTrue(all((d / n).read_bytes() == b for n, b in existing.items()), "another model's artifact was modified")
            m2 = PatchTSTForecaster.load(d, 7, "cpu")
            X = self.w["X"][:200]
            np.testing.assert_allclose(m2._predict_scaled(X), self.model._predict_scaled(X), rtol=1e-5, atol=1e-6)
            self.assertEqual(m2.forecast_latest(self.clean), self.model.forecast_latest(self.clean))
            self.assertEqual(m2.config, self.model.config)
            for k in self.model.scalers:
                np.testing.assert_allclose(m2.scalers[k].mean_, self.model.scalers[k].mean_)
            meta = json.loads((d / "patchtst_7d_meta.json").read_text())
            for k in ("config", "input_columns", "split_dates", "scaler_params", "window_accounting", "history", "leakage_rules"):
                self.assertIn(k, meta)
            self.assertEqual(meta["split_dates"], self.splits.as_dict())
            with self.assertRaises(FileNotFoundError):
                PatchTSTForecaster.load(d, 30)

    def test_resume_continues_from_interrupted_run(self) -> None:
        class Stop(Exception):
            pass
        orig = PatchTSTForecaster._write_resume

        def write_then_stop(self, *a, **k):
            orig(self, *a, **k)
            raise Stop()
        cfg = replace_config(FAST, max_epochs=2, patience=5)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            PatchTSTForecaster._write_resume = write_then_stop
            try:
                with self.assertRaises(Stop):
                    PatchTSTForecaster(7, cfg, "cpu").fit(self.clean, self.splits, checkpoint_dir=d)
            finally:
                PatchTSTForecaster._write_resume = orig
            rp = artifact_paths(d, 7)["resume"]
            self.assertTrue(rp.exists())
            with self.assertRaises(ValueError):                                     # different settings -> refuse to resume
                PatchTSTForecaster(7, replace_config(cfg, d_model=32), "cpu").fit(self.clean, self.splits, checkpoint_dir=d, resume=True)
            m = PatchTSTForecaster(7, cfg, "cpu")
            m.fit(self.clean, self.splits, checkpoint_dir=d, resume=True)
            self.assertEqual([r["epoch"] for r in m.history], [1, 2])
            self.assertFalse(rp.exists())                                           # removed once training has finished
            self.assertEqual(len(list(d.iterdir())), 0)

    # ---- rolling origin ------------------------------------------------------------------------------------
    def test_rolling_patchtst_same_folds_no_future_targets(self) -> None:
        folds = ro.make_rolling_folds(self.clean["date"], n_folds=2, test_days=60, val_days=90, min_train_days=365)
        long = ro.rolling_patchtst(self.clean, folds, (7,), FAST, max_epochs=1, device="cpu")
        self.assertFalse(long.empty)
        self.assertEqual(set(long["model"]), {"patchtst"})
        self.assertEqual(set(long["sensor_id"]), {"S1", "S2"})
        f0 = {f.index: f for f in folds}
        td, od = pd.to_datetime(long["target_date"]), pd.to_datetime(long["origin_date"])
        self.assertTrue(all(t > f0[f].splits.val_end and t <= f0[f].test_end for t, f in zip(td, long["fold"])))
        # same scoring rule as the tabular models: a forecast belongs to a fold iff its TARGET date is in (origin, test_end]; its origin may be up to
        # h-1 days before the fold origin (inputs still end before the origin). Legacy rule: include_straddling_origins=False.
        self.assertTrue(all(o + pd.Timedelta(days=7) > f0[f].splits.val_end for o, f in zip(od, long["fold"])))
        o = list(zip(long["sensor_id"], od)); t = list(zip(long["sensor_id"], td))
        self.assertTrue((self.imp.reindex(o).to_numpy() == 0).all(), "imputed rolling origin")
        self.assertTrue((self.imp.reindex(t).to_numpy() == 0).all(), "imputed rolling target")
        self.assertTrue(np.isfinite(long["y_seasonal_naive"]).any())              # baselines are carried alongside
        s = ro.summarize(long)["patchtst_7d"]
        self.assertIn("rmse_skill_vs_persistence_pooled", s)
        self.assertEqual(ro.metrics_table(ro.summarize(long), "Toy")["Model"].tolist().count("PatchTST"), 1)


if __name__ == "__main__":
    unittest.main()
