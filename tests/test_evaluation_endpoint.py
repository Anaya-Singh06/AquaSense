"""Lightweight tests for the Model Evaluation data layer (stdlib only; no models, no training)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from backend import evaluation as ev

try:
    from fastapi.testclient import TestClient
except Exception:  # fastapi/httpx not installed
    TestClient = None

REF = ev.DEFAULT_PATH


class EvaluationLoadTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("AQUASENSE_EVAL_SUMMARY", None)
        os.environ.pop("AQUASENSE_EVAL_STATIONS", None)

    def tearDown(self):
        os.environ.pop("AQUASENSE_EVAL_SUMMARY", None)
        os.environ.pop("AQUASENSE_EVAL_STATIONS", None)

    def test_reference_file_matches_payload(self):
        raw = json.loads(REF.read_text(encoding="utf-8"))
        out = ev.load_evaluation()
        self.assertEqual(out["metrics"], raw["metrics"])          # served verbatim, not re-typed
        self.assertEqual(out["kind"], "completed_offline_evaluation")
        self.assertEqual(out["horizons"], ["7d", "30d", "90d"])
        self.assertEqual(out["data"]["stations_selected"], 60)
        self.assertEqual(out["data"]["rows"], 366660)

    def test_derived_findings_match_reported_results(self):
        d = ev.load_evaluation()["derived"]
        self.assertEqual(d["7d"]["best_rmse"], "PatchTST")
        self.assertEqual(d["7d"]["best_mean_nse"], "PatchTST")
        self.assertEqual(d["30d"]["best_rmse"], "XGBoost")
        self.assertEqual(d["90d"]["best_rmse"], "LightGBM")
        self.assertIn("LightGBM", d["90d"]["mean_nse_below_zero"])
        self.assertIn("LightGBM", d["90d"]["pooled_better_station_worse"])
        self.assertIn("XGBoost", d["30d"]["pooled_better_station_worse"])

    def test_missing_file_raises(self):
        os.environ["AQUASENSE_EVAL_SUMMARY"] = str(Path(tempfile.gettempdir()) / "does_not_exist_aquasense.json")
        with self.assertRaises(ev.EvaluationUnavailable):
            ev.load_evaluation()
        os.environ["AQUASENSE_EVAL_STATIONS"] = str(Path(tempfile.gettempdir()) / "does_not_exist_stations.csv")
        with self.assertRaises(ev.EvaluationUnavailable):
            ev.load_station_evaluation()

    def test_station_file_has_patchtst_and_lightgbm(self):
        out = ev.load_station_evaluation()
        self.assertEqual(out["models"], ["PatchTST", "LightGBM"])
        self.assertEqual(out["horizons"], ["7d", "30d", "90d"])
        self.assertGreaterEqual(out["n_stations"], 50)
        sample = next(iter(out["stations"].values()))
        self.assertIn("PatchTST", sample["7d"])
        self.assertIn("LightGBM", sample["7d"])
        self.assertIn("rmse_m", sample["7d"]["PatchTST"])
        self.assertEqual(out["summary"]["7d"]["PatchTST"]["station_wins"], 41)
        self.assertEqual(out["summary"]["7d"]["LightGBM"]["station_wins"], 38)
        self.assertEqual(out["summary"]["90d"]["LightGBM"]["station_wins"], 30)

    def test_malformed_files_raise(self):
        for body in ("{not json", "[]", json.dumps({"data": {}, "evaluation": {}, "metrics": {"7d": {}}})):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
                f.write(body)
            os.environ["AQUASENSE_EVAL_SUMMARY"] = f.name
            try:
                with self.assertRaises(ev.EvaluationUnavailable):
                    ev.load_evaluation()
            finally:
                os.unlink(f.name)


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class EvaluationEndpointTests(unittest.TestCase):
    def test_endpoint_works_without_operational_service(self):
        from backend.main import STATE, app
        STATE["svc"], STATE["error"] = None, "models missing"      # simulate unavailable models
        client = TestClient(app)                                    # no lifespan: service stays unavailable
        self.assertEqual(client.get("/api/health").status_code, 503)
        r = client.get("/api/evaluation")
        self.assertEqual(r.status_code, 200)
        self.assertIn("derived", r.json())
        s = client.get("/api/evaluation/stations")
        self.assertEqual(s.status_code, 200)
        body = s.json()
        self.assertIn("PatchTST", body["models"])
        self.assertIn("LightGBM", body["models"])
        self.assertGreater(body["n_stations"], 0)

    def test_endpoint_503_when_file_missing(self):
        from backend.main import app
        os.environ["AQUASENSE_EVAL_SUMMARY"] = "/nonexistent/summary.json"
        try:
            self.assertEqual(TestClient(app).get("/api/evaluation").status_code, 503)
        finally:
            os.environ.pop("AQUASENSE_EVAL_SUMMARY", None)


if __name__ == "__main__":
    unittest.main()
