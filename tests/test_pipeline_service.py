"""Smoke test of the full pipeline (synthetic AND 'real-schema' data) and of the API service layer."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tests._helpers import ROOT, allow_xgb_standin, has  # noqa: F401
from src.config import LABEL_COLUMN, Paths
import src.pipeline as pipeline
import backend.service as service

allow_xgb_standin()
TMP: Path | None = None


def _paths_factory(root: Path):
    return lambda source="synthetic": Paths(source, root=root)


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp())
    (TMP / "data").mkdir()
    with mock.patch.object(pipeline, "Paths", _paths_factory(TMP)):
        pipeline.run("synthetic", regenerate=True, skip_lstm=not has("torch"), lstm_epochs=3)
        # "real-schema" smoke test: same numbers, but no synthetic label and USGS-style ids (NOT real data)
        df = pd.read_csv(TMP / "data" / "dwlr_data.csv").drop(columns=[LABEL_COLUMN])
        df["sensor_id"] = df["sensor_id"].str.replace("DWLR_", "USGS_", regex=False)
        df.to_csv(TMP / "data" / "real_data.csv", index=False)
        pipeline.run("real", regenerate=False, skip_lstm=True)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TestPipelineOutputs(unittest.TestCase):
    def test_artifacts_exist(self):
        out, mdl = TMP / "outputs", TMP / "models"
        for f in ("anomalies.csv", "spatial_grid.geojson", "forecasts.json", "metrics.json", "xgb_30d_actual_vs_predicted.png"):
            self.assertTrue((out / f).exists(), f)
        self.assertTrue((mdl / "anomaly_model.joblib").exists())
        self.assertTrue(any(p.name.startswith("xgb_30d.") for p in mdl.iterdir()))
        self.assertTrue(any(p.name.startswith("xgb_90d.") for p in mdl.iterdir()))

    def test_metrics_schema_and_baseline_reported(self):
        m = json.loads((TMP / "outputs" / "metrics.json").read_text())
        for k in ("xgb_30d", "xgb_90d"):
            self.assertIn("persistence", m[k]); self.assertIn("nse_per_sensor_mean", m[k]["model"])
            self.assertIn("model_beats_persistence", m[k])
        self.assertIn("precision", m["anomaly"]["test_period"]); self.assertIn("roc_auc", m["anomaly"]["all_dates"])

    def test_anomaly_output_schema(self):
        a = pd.read_csv(TMP / "outputs" / "anomalies.csv")
        self.assertEqual(list(a.columns), ["sensor_id", "date", "anomaly_score", "anomaly_flag", "severity", "reason"])
        self.assertTrue(a["severity"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all())
        self.assertTrue(a["anomaly_score"].between(0, 1).all()); self.assertTrue(a["reason"].str.contains("potential").all())
        self.assertFalse(a["reason"].str.contains("illegal").any())

    def test_forecast_output_schema(self):
        f = json.loads((TMP / "outputs" / "forecasts.json").read_text())
        s = next(iter(f.values()))
        x = s["xgb_30d"]
        self.assertEqual(len(x["trajectory"]), 30); self.assertEqual(len(s["xgb_90d"]["trajectory"]), 90)
        self.assertLess(x["endpoint_lower"], x["endpoint_depth_m_bgl"]); self.assertLess(x["endpoint_depth_m_bgl"], x["endpoint_upper"])
        self.assertIn("approximate", x["interval_type"])
        if has("torch"):
            self.assertEqual(len(s["lstm_7d"]), 7)

    def test_real_schema_run_has_no_fake_labels(self):
        m = json.loads((TMP / "outputs" / "real" / "metrics.json").read_text())
        self.assertNotIn("precision", m["anomaly"]); self.assertIn("unsupervised", m["anomaly"]["note"])
        self.assertTrue((TMP / "outputs" / "real" / "spatial_grid.geojson").exists())


class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(service, "Paths", _paths_factory(TMP)):
            cls.svc = service.AquaSenseService("synthetic")

    def test_health_and_sensors(self):
        h = self.svc.health(); self.assertEqual(h["status"], "ok"); self.assertEqual(h["n_sensors"], 10)
        s = self.svc.sensors(); self.assertEqual(len(s), 10)
        self.assertTrue({x["status"] for x in s} <= {"SAFE", "SEMI_CRITICAL", "CRITICAL"})

    def test_forecast_and_errors(self):
        f = self.svc.forecast("DWLR_001", 60)
        self.assertEqual(len(f["history"]), 60); self.assertIsNotNone(f["xgb_30d"])
        with self.assertRaises(service.UnknownSensor):
            self.svc.forecast("NOPE")

    def test_anomalies_filters(self):
        a = self.svc.anomalies_list(limit=5, min_severity="HIGH")
        self.assertLessEqual(len(a), 5); self.assertTrue(all(x["severity"] in ("HIGH", "CRITICAL") for x in a))
        with self.assertRaises(service.UnknownSensor):
            self.svc.anomalies_list(sensor_id="NOPE")

    def test_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(service, "Paths", _paths_factory(Path(d))):
            with self.assertRaises(service.ArtifactsMissing):
                service.AquaSenseService("synthetic")


@unittest.skipUnless(has("fastapi") and has("httpx"), "fastapi/httpx not installed (BLOCKED in this sandbox)")
class TestHTTPApi(unittest.TestCase):
    def test_endpoints(self):
        from fastapi.testclient import TestClient
        import backend.main as main
        with mock.patch.object(service, "Paths", _paths_factory(TMP)), TestClient(main.app) as c:
            self.assertEqual(c.get("/api/health").status_code, 200)
            self.assertEqual(len(c.get("/api/sensors").json()), 10)
            self.assertEqual(c.get("/api/forecast/DWLR_001").status_code, 200)
            self.assertEqual(c.get("/api/forecast/NOPE").status_code, 404)
            self.assertEqual(c.get("/api/anomalies?min_severity=BAD").status_code, 422)
            self.assertEqual(c.get("/api/anomalies?limit=3").status_code, 200)
            self.assertEqual(c.get("/api/spatial-grid").json()["type"], "FeatureCollection")


@unittest.skipUnless(has("torch"), "PyTorch not installed (BLOCKED in this sandbox)")
class TestLSTM(unittest.TestCase):
    def test_lstm_artifacts_and_forecast(self):
        self.assertTrue((TMP / "models" / "lstm_model.pt").exists())
        self.assertTrue((TMP / "models" / "lstm_scalers.joblib").exists())
        m = json.loads((TMP / "outputs" / "metrics.json").read_text())["lstm_7d"]
        self.assertIn("persistence", m)


if __name__ == "__main__":
    unittest.main()
