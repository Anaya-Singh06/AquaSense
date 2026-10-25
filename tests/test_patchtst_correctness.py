"""Deterministic correctness tests for the PatchTST pipeline (tiny CPU networks, tiny synthetic data, no real data, no test-set tuning).

What these tests pin down (each one maps to a review question):
  * window rejection is EXACTLY the set of windows that touch an interpolated day (closed-form counts), longer context rejects more;
  * windows never use future information; the test period never influences training / early stopping / scalers;
  * patch extraction == manual slicing, for every (sample, channel, patch); output dimensions; batch independence;
  * RevIN + persistence reconstruction: untrained net == persistence, depth-affine equivariance, metres round-trip;
  * evaluation uses identical samples and the intended horizon for model and persistence;
  * strict inference skips exactly the sensors whose last ``context_length`` days touch an interpolated day (same rule as training);
  * a tiny overfit test proves the network + loss + optimiser can learn a learnable target beyond persistence.

The window tests need only NumPy/pandas; the rest are skipped when PyTorch is missing.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests._helpers import allow_xgb_standin, has

allow_xgb_standin()
from src.config import TARGET  # noqa: E402
from src.preprocessing.preprocess import fill_missing  # noqa: E402
from src.preprocessing.splits import SplitDates, make_split_dates  # noqa: E402
from src.preprocessing.windows import make_windows  # noqa: E402
from tests.test_leakage_masking import imputed_lookup, pipeline, toy  # noqa: E402

HAS_TORCH = has("torch")

if HAS_TORCH:
    import torch  # noqa: E402

    from src.models.patchtst_forecaster import PatchTSTConfig, PatchTSTForecaster, PatchTSTNet, replace_config  # noqa: E402

    FAST = PatchTSTConfig(context_length=32, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.1,
                          head_dropout=0.0, batch_size=64, eval_batch_size=128, max_epochs=2, patience=2, amp="off", seed=7)


# ------------------------------------------------------------------------------------------------ helpers
def _frame(depth_by_sensor: dict[str, np.ndarray], start: str = "2020-01-01") -> pd.DataFrame:
    """Raw daily frame (no NaN unless the caller put them in). Climate inputs are smooth deterministic functions of the day."""
    rows = []
    for k, (sid, y) in enumerate(depth_by_sensor.items()):
        n = len(y)
        t = np.arange(n)
        rows.append(pd.DataFrame({"sensor_id": sid, "date": pd.date_range(start, periods=n, freq="D"), "latitude": 10.0 + k, "longitude": 78.0,
                                  TARGET: y, "rainfall_mm": 2 + np.sin(t / 7.0) ** 2, "temperature_c": 25 + 3 * np.sin(t / 50.0),
                                  "evapotranspiration_mm": 4 + np.cos(t / 40.0), "soil_moisture": 0.3 + 0.05 * np.sin(t / 30.0)}))
    return pd.concat(rows, ignore_index=True)


def _all_train(df: pd.DataFrame) -> SplitDates:
    """Split object that labels every window 'train' (isolates the observed-only rule from the split rule)."""
    end = df["date"].max()
    return SplitDates(df["date"].min(), end, end, end)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = fill_missing(df)
    out["x"] = out[TARGET]
    out["obs"] = (out["depth_imputed"] == 0).astype(int)
    return out


# ================================================================================================ window rejection (no torch)
class TestWindowRejectionPolicy(unittest.TestCase):
    """Why windows are rejected, in closed form. A window of L input days + h target days is kept iff all L+h days are observed."""

    def test_single_interpolated_day_rejects_exactly_the_windows_that_touch_it(self) -> None:
        y = np.linspace(10, 12, 60)
        y[20] = np.nan                                    # one missing day -> interpolated (flag 1)
        d = _prep(_frame({"A": y}))
        L, h = 10, 3
        w = make_windows(d, ["x"], "x", L, h, _all_train(d), observed_col="obs")
        st = w["stats"]["train"]
        n_win = 60 - (L + h) + 1                           # 48 candidate windows
        # window i spans days [i, i+L+h-1]; it touches day 20 for i in [20-(L+h-1), 20] = [8, 20] -> 13 windows
        self.assertEqual((st["candidate"], st["dropped_imputed"], st["retained"]), (n_win, 13, n_win - 13))
        origin_idx = (w["meta"]["origin_date"] - d["date"].min()).dt.days.to_numpy()   # window i has origin day i+L-1
        starts = origin_idx - (L - 1)
        self.assertTrue(((starts < 8) | (starts > 20)).all())
        # target-only contamination (start 8..10: day 20 falls on a target day) and input-only contamination (start 11..20) are both dropped
        for s in (8, 9, 10, 11, 15, 20):
            self.assertNotIn(s, set(starts))
        self.assertIn(7, set(starts)); self.assertIn(21, set(starts))     # the neighbours that do not touch day 20 survive
        self.assertTrue(w["obs"].all())

    def test_gap_of_g_days_rejects_span_plus_g_minus_1_windows(self) -> None:
        for L, h, g in ((16, 7, 5), (96, 7, 3), (30, 7, 3)):
            n = 400
            y = np.linspace(10, 14, n)
            y[200:200 + g] = np.nan
            d = _prep(_frame({"A": y}))
            st = make_windows(d, ["x"], "x", L, h, _all_train(d), observed_col="obs")["stats"]["train"]
            span = L + h
            self.assertEqual(st["candidate"], n - span + 1)
            self.assertEqual(st["dropped_imputed"], span + g - 1, (L, h, g))

    def test_longer_context_rejects_a_larger_share(self) -> None:
        rng = np.random.default_rng(0)
        y = np.linspace(10, 14, 700)
        for s in rng.choice(np.arange(10, 690), size=25, replace=False):      # scattered short gaps (~3.5% of days)
            y[s:s + int(rng.integers(1, 4))] = np.nan
        d = _prep(_frame({"A": y}))
        share = {}
        for L in (16, 30, 96):
            st = make_windows(d, ["x"], "x", L, 7, _all_train(d), observed_col="obs")["stats"]["train"]
            share[L] = st["retained"] / st["candidate"]
        self.assertGreater(share[16], share[30]); self.assertGreater(share[30], share[96])

    def test_require_observed_false_keeps_interpolated_windows(self) -> None:
        y = np.linspace(10, 12, 60); y[20] = np.nan
        d = _prep(_frame({"A": y}))
        w = make_windows(d, ["x"], "x", 10, 3, _all_train(d), observed_col="obs", require_observed=False)
        self.assertEqual(w["stats"]["train"]["retained"], 48)       # detector sanity: the strict rule is what removes them

    def test_split_by_target_dates_and_straddlers_discarded(self) -> None:
        d = _prep(_frame({"A": np.linspace(10, 14, 200)}))
        s = make_split_dates(d["date"])
        L, h = 20, 7
        w = make_windows(d, ["x"], "x", L, h, s, observed_col="obs")
        m = w["meta"]
        first, last = m["origin_date"] + pd.Timedelta(days=1), m["origin_date"] + pd.Timedelta(days=h)
        tr, va, te, di = (m["split"] == k for k in ("train", "val", "test", "discard"))
        self.assertTrue((last[tr] <= s.train_end).all())
        self.assertTrue(((first[va] > s.train_end) & (last[va] <= s.val_end)).all())
        self.assertTrue((first[te] > s.val_end).all())
        self.assertTrue(di.any())
        # every discarded window straddles a split boundary (some target days on each side)
        self.assertTrue((((first[di] <= s.train_end) & (last[di] > s.train_end)) | ((first[di] <= s.val_end) & (last[di] > s.val_end))).all())
        # a validation window may use TRAIN-period days as inputs (the past) but its targets are all after train_end
        self.assertLessEqual(m["origin_date"][va].min() - pd.Timedelta(days=L - 1), s.train_end)
        self.assertEqual(int(tr.sum() + va.sum() + te.sum() + di.sum()), len(m))

    def test_inputs_are_strictly_before_targets(self) -> None:
        y = np.arange(80, dtype=float)                     # depth == day index, so windows are self-describing
        d = _prep(_frame({"A": y}))
        L, h = 12, 5
        w = make_windows(d, ["x"], "x", L, h, _all_train(d), observed_col="obs")
        self.assertTrue((w["X"][:, :, 0].max(axis=1) + 1 == w["Y"][:, 0]).all())            # first target day directly follows the last input day
        np.testing.assert_array_equal(w["Y"] - w["Y"][:, :1], np.tile(np.arange(h), (len(w["Y"]), 1)))
        self.assertEqual(w["X"].shape[1:], (L, 1)); self.assertEqual(w["Y"].shape[1], h)


# ================================================================================================ network math
@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestPatchExtractionAndShapes(unittest.TestCase):
    def test_patchify_equals_manual_slicing_for_every_sample_channel_patch(self) -> None:
        for (L, P, S) in ((40, 8, 4), (100, 16, 8), (37, 8, 5), (16, 16, 8)):        # includes cases where the oldest days are trimmed
            net = PatchTSTNet(3, L, 4, patch_len=P, stride=S, d_model=8, n_heads=2, n_layers=1, d_ff=16)
            torch.manual_seed(1)
            x = torch.randn(4, L, 3)
            p = net.patchify(x)
            n_p = (L - P) // S + 1
            trim = (L - P) % S
            self.assertEqual((net.n_patches, net.trim), (n_p, trim))
            self.assertEqual(tuple(p.shape), (4 * 3, n_p, P))
            for b in range(4):
                for c in range(3):
                    for n in range(n_p):
                        a = trim + n * S
                        self.assertTrue(torch.equal(p[b * 3 + c, n], x[b, a:a + P, c]))
            self.assertTrue(torch.equal(p[0, -1], x[0, L - P:, 0]))                  # last patch always ends on the most recent day

    def test_output_dimensions_for_all_horizons(self) -> None:
        for h in (1, 7, 30, 90):
            net = PatchTSTNet(7, 96, h).eval()
            self.assertEqual(tuple(net(torch.randn(5, 96, 7)).shape), (5, h))
            self.assertEqual(net.head.in_features, 7 * net.n_patches * 64)           # head sees (channels x patches x d_model), channel-major

    def test_samples_do_not_interact(self) -> None:
        torch.manual_seed(0)
        net = PatchTSTNet(3, 40, 4, patch_len=8, stride=4, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, head_dropout=0.0).eval()
        with torch.no_grad():
            net.head.weight.normal_(0, 0.1)
            x = torch.randn(6, 40, 3)
            full = net(x)
            for i in range(6):
                torch.testing.assert_close(net(x[i:i + 1]), full[i:i + 1], atol=1e-5, rtol=1e-5)
            x2 = x.clone(); x2[2] += 5.0                                             # change one sample only
            out2 = net(x2)
            keep = [0, 1, 3, 4, 5]
            torch.testing.assert_close(out2[keep], full[keep], atol=1e-5, rtol=1e-5)
            self.assertFalse(torch.allclose(out2[2], full[2]))

    def test_channels_are_distinguished_by_the_head(self) -> None:
        # shared encoder weights (channel independence) but a channel-specific head: swapping two channels must change the output
        torch.manual_seed(0)
        net = PatchTSTNet(3, 40, 4, patch_len=8, stride=4, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, head_dropout=0.0).eval()
        with torch.no_grad():
            net.head.weight.normal_(0, 0.1)
            x = torch.randn(2, 40, 3)
            xs = x[:, :, [0, 2, 1]]
            self.assertFalse(torch.allclose(net(x), net(xs)))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestNormalisationAndPersistenceReconstruction(unittest.TestCase):
    def test_untrained_network_is_exactly_persistence_with_and_without_revin(self) -> None:
        torch.manual_seed(0)
        for revin in (True, False):
            net = PatchTSTNet(7, 96, 7, revin=revin, norm_mask=[True] * 5 + [False] * 2).eval()
            x = torch.randn(8, 96, 7) * 3 + 5
            with torch.no_grad():
                torch.testing.assert_close(net(x), x[:, -1:, 0].repeat(1, 7), atol=1e-6, rtol=0)

    def test_prediction_is_affine_equivariant_in_depth(self) -> None:
        """pred(a*depth + b) == a*pred(depth) + b: RevIN scale/shift and the last + scale*delta reconstruction are mutually consistent."""
        torch.manual_seed(0)
        net = PatchTSTNet(3, 40, 4, patch_len=8, stride=4, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, head_dropout=0.0,
                          norm_mask=[True, True, False]).eval()
        with torch.no_grad():
            net.head.weight.normal_(0, 0.1); net.head.bias.normal_(0, 0.1)
            x = torch.randn(5, 40, 3)
            a, b = 2.5, 7.0
            xs = x.clone(); xs[:, :, 0] = a * x[:, :, 0] + b
            torch.testing.assert_close(net(xs), a * net(x) + b, atol=2e-3, rtol=1e-3)
            # covariate channels are instance-normalised: shifting/scaling a covariate must not change the output (up to eps)
            xc = x.clone(); xc[:, :, 1] = 4.0 * x[:, :, 1] - 3.0
            torch.testing.assert_close(net(xc), net(x), atol=2e-3, rtol=1e-3)
            # the calendar channel (norm_mask False) is NOT normalised, so shifting it does change the output
            xk = x.clone(); xk[:, :, 2] = x[:, :, 2] + 3.0
            self.assertFalse(torch.allclose(net(xk), net(x), atol=1e-3))

    def test_revin_ignores_nothing_it_should_not(self) -> None:
        net = PatchTSTNet(2, 32, 3, patch_len=8, stride=4, d_model=8, n_heads=2, n_layers=1, d_ff=16).eval()
        x = torch.zeros(2, 32, 2)                          # constant window: std -> sqrt(eps), no NaN/inf
        with torch.no_grad():
            self.assertTrue(torch.isfinite(net(x)).all())

    def test_zero_init_head_gradient_flow(self) -> None:
        torch.manual_seed(0)
        net = PatchTSTNet(2, 32, 3, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, head_dropout=0.0)
        x = torch.randn(32, 32, 2); y = x[:, -1:, 0] + 0.7
        loss = ((net(x) - y) ** 2).mean(); loss.backward()
        self.assertGreater(float(net.head.weight.grad.abs().sum()), 0)               # the head learns from step 1 ...
        self.assertEqual(float(sum(p.grad.abs().sum() for p in net.encoder.parameters())), 0.0)     # ... the encoder only from step 2 (W_head = 0)
        with torch.no_grad():
            for p in net.head.parameters():
                p -= 1e-2 * p.grad
        net.zero_grad()
        loss = ((net(x) - y) ** 2).mean(); loss.backward()
        self.assertGreater(float(sum(p.grad.abs().sum() for p in net.encoder.parameters())), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTinyOverfit(unittest.TestCase):
    """The architecture + loss + optimiser must be able to beat persistence on a learnable target."""

    @staticmethod
    def _sine_windows(L: int = 32, h: int = 3, period: int = 20, n: int = 400):
        s = np.sin(2 * np.pi * np.arange(n) / period).astype(np.float32)
        idx = np.arange(0, n - L - h + 1)
        X = np.stack([s[i:i + L] for i in idx])[:, :, None]
        Y = np.stack([s[i + L:i + L + h] for i in idx])
        return torch.from_numpy(X), torch.from_numpy(Y)

    def test_network_overfits_deterministic_sine_far_below_persistence(self) -> None:
        torch.manual_seed(0)
        X, Y = self._sine_windows()
        persistence = float(((X[:, -1:, 0] - Y) ** 2).mean())
        net = PatchTSTNet(1, 32, 3, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, head_dropout=0.0)
        self.assertAlmostEqual(float(((net(X) - Y) ** 2).mean().detach()), persistence, places=5)       # step 0 == persistence
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        losses = []
        for _ in range(300):
            opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); losses.append(loss.item())
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(losses[-1], 0.1 * persistence, f"final {losses[-1]:.4f} vs persistence {persistence:.4f}")

    def test_forecaster_fit_learns_learnable_series_and_beats_persistence(self) -> None:
        """Full fit()/evaluate() path (scaling, windows, splits, early stopping, metres) on two noise-free, fully observed series."""
        n = 600
        t = np.arange(n)
        d = _frame({"A": 12 + 1.5 * np.sin(2 * np.pi * t / 24), "B": 18 + 2.0 * np.sin(2 * np.pi * t / 24 + 1.0)})
        clean = fill_missing(d)
        self.assertTrue((clean["depth_imputed"] == 0).all())
        splits = make_split_dates(clean["date"])
        cfg = replace_config(FAST, max_epochs=20, patience=20, lr=3e-3, dropout=0.0, seed=3)
        m = PatchTSTForecaster(7, cfg, "cpu")
        w = m.fit(clean, splits)
        self.assertLess(m.best_val_loss, 0.5 * m.persistence_val_loss, (m.best_val_loss, m.persistence_val_loss))
        r = m.evaluate(w, "test")
        self.assertLess(r["model"]["rmse"], 0.7 * r["persistence"]["rmse"], (r["model"]["rmse"], r["persistence"]["rmse"]))
        self.assertGreater(r["model"]["nse_per_sensor_mean"], r["persistence"]["nse_per_sensor_mean"])
        self.assertTrue(r["model_beats_persistence"])


# ================================================================================================ forecaster-level checks
@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestForecasterConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean, _ = pipeline(toy())
        cls.imp = imputed_lookup(cls.clean)
        cls.depth = cls.clean.set_index(["sensor_id", "date"])[TARGET]
        cls.splits = make_split_dates(cls.clean["date"])
        cls.model = PatchTSTForecaster(7, FAST, "cpu")
        cls.w = cls.model.fit(cls.clean, cls.splits)

    # ---- scaling / metres ---------------------------------------------------------------------------
    def test_scaler_roundtrip_returns_metres(self) -> None:
        sc = self.model._scale(self.clean)
        ok = sc[TARGET].notna()
        np.testing.assert_allclose(self.model._inv_depth(sc.loc[ok, "x_depth"].to_numpy()), sc.loc[ok, TARGET].to_numpy(), atol=1e-9)
        tr = self.clean[(self.clean["date"] <= self.splits.train_end) & (self.clean["depth_imputed"] == 0)]
        p = self.model.scalers["depth"]
        self.assertAlmostEqual(float(p.mean_[0]), float(tr[TARGET].mean()), places=9)       # scaler is the TRAIN observed-depth mean / std
        self.assertAlmostEqual(float(p.scale_[0]), float(tr[TARGET].std(ddof=0)), places=9)

    def test_evaluate_targets_and_persistence_are_the_real_metre_values(self) -> None:
        """true == recorded depth at the target dates, base == recorded depth at the origin date (looked up independently of the scaler path)."""
        r = self.model.evaluate(self.w, "test")
        a = r["_arrays"]; meta = a["meta"]
        h = self.model.horizon
        self.assertEqual(a["true"].shape, (len(meta), h))
        for k in range(h):
            keys = list(zip(meta["sensor_id"], meta["origin_date"] + pd.Timedelta(days=k + 1)))
            np.testing.assert_allclose(a["true"][:, k], self.depth.reindex(keys).to_numpy(), atol=1e-4)
        origin = self.depth.reindex(list(zip(meta["sensor_id"], meta["origin_date"]))).to_numpy()
        for k in range(h):
            np.testing.assert_allclose(a["base"][:, k], origin, atol=1e-4)            # persistence: the same last observed depth at every lead

    def test_untrained_forecaster_scores_exactly_persistence_in_metres(self) -> None:
        m0 = PatchTSTForecaster(7, FAST, "cpu")
        m0.scalers = self.model.scalers
        m0.net = m0._build_net()
        r = m0.evaluate(self.w, "test")
        np.testing.assert_allclose(r["_arrays"]["pred"], r["_arrays"]["base"], atol=1e-4)
        self.assertAlmostEqual(r["model"]["rmse"], r["persistence"]["rmse"], places=4)
        self.assertAlmostEqual(r["rmse_skill_vs_persistence"], 0.0, places=4)

    # ---- metrics: identical samples, intended horizon --------------------------------------------------
    def test_model_and_persistence_are_scored_on_identical_samples_and_full_horizon(self) -> None:
        r = self.model.evaluate(self.w, "test")
        a = r["_arrays"]; h = 7
        n_win = r["n_windows"]
        self.assertEqual(r["n"], n_win * h)                                  # every (window, lead-day) pair, model and persistence alike
        self.assertEqual(r["day7_only"]["n"], n_win)
        self.assertEqual(r["horizon_days"], h)
        # recompute both blocks independently with NumPy on the very same arrays
        for key, arr in (("model", a["pred"]), ("persistence", a["base"])):
            self.assertAlmostEqual(r[key]["rmse"], float(np.sqrt(np.mean((a["true"] - arr) ** 2))), places=6)
            self.assertAlmostEqual(r[key]["mae"], float(np.mean(np.abs(a["true"] - arr))), places=6)
            self.assertAlmostEqual(r["day7_only"][key]["rmse"], float(np.sqrt(np.mean((a["true"][:, -1] - arr[:, -1]) ** 2))), places=6)
        # per-sensor NSE: sensor labels line up with the (N, h) arrays (row-major ravel <-> np.repeat)
        sens = a["meta"]["sensor_id"].to_numpy()
        per = []
        for s in np.unique(sens):
            y, p = a["true"][sens == s].ravel(), a["pred"][sens == s].ravel()
            per.append(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
        self.assertAlmostEqual(r["model"]["nse_per_sensor_mean"], float(np.mean(per)), places=6)
        # the 'test' windows are exactly the windows whose targets are after val_end
        self.assertTrue((a["meta"]["origin_date"] + pd.Timedelta(days=1) > self.splits.val_end).all())

    def test_persistence_val_loss_reference_matches_manual_computation(self) -> None:
        va = (self.w["meta"]["split"] == "val").to_numpy()
        manual = float(np.mean((self.w["X"][va][:, -1, 0][:, None].astype(np.float64) - self.w["Y"][va].astype(np.float64)) ** 2))
        self.assertAlmostEqual(self.model.persistence_val_loss, manual, places=9)

    # ---- leakage safeguards ---------------------------------------------------------------------------
    def test_test_period_never_influences_training_or_model_selection(self) -> None:
        cut = (self.splits.val_end - pd.Timestamp("2015-01-01")).days
        clean2, _ = pipeline(toy(perturb_after=cut))                       # every observation after val_end is changed
        m2 = PatchTSTForecaster(7, FAST, "cpu")
        m2.fit(clean2, make_split_dates(clean2["date"]))
        self.assertEqual(m2.split_dates, self.model.split_dates)
        self.assertEqual(m2.best_epoch, self.model.best_epoch)
        for ra, rb in zip(self.model.history, m2.history):
            self.assertAlmostEqual(ra["train_loss"], rb["train_loss"], places=6)
            self.assertAlmostEqual(ra["val_loss"], rb["val_loss"], places=6)
        self.assertAlmostEqual(m2.persistence_val_loss, self.model.persistence_val_loss, places=9)
        for k, sc in self.model.scalers.items():
            np.testing.assert_allclose(m2.scalers[k].mean_, sc.mean_); np.testing.assert_allclose(m2.scalers[k].scale_, sc.scale_)

    def test_strict_inference_rule_matches_the_training_window_rule(self) -> None:
        """A sensor is forecast at cut-off D iff none of its last L days is interpolated - the same condition that lets a context
        into training. (Training additionally needs observed TARGET days, which cannot be known at inference.)"""
        L = FAST.context_length
        checked = skipped = 0
        for day in range(900, 1450, 13):
            cutoff = pd.Timestamp("2015-01-01") + pd.Timedelta(days=day)
            got = set(self.model.forecast_latest(self.clean[self.clean["date"] <= cutoff]))
            expected = set()
            for sid in ("S1", "S2"):
                span = [(sid, cutoff - pd.Timedelta(days=k)) for k in range(L)]
                if (self.imp.reindex(span).to_numpy() == 0).all():
                    expected.add(sid)
            self.assertEqual(got, expected, f"cutoff day {day}")
            checked += 1; skipped += int(len(expected) < 2)
        self.assertGreater(skipped, 0); self.assertLess(skipped, checked)       # detector sanity: some cut-offs skip, some do not

    def test_all_sensors_skipped_returns_empty_and_logs_reason(self) -> None:
        raw = toy(n=1320, gaps={"S1": [(1300, 2)], "S2": [(1305, 1)]})          # both sensors interpolated inside their last 32 days
        clean, _ = pipeline(raw)
        with self.assertLogs(level="WARNING") as cm:
            out = self.model.forecast_latest(clean)
        self.assertEqual(out, {})
        self.assertTrue(any("no sensor has a usable last 32-day context" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
