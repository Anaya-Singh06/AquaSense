"""Framework-independent service layer for the API: loads artifacts once, answers queries.

It NEVER trains models. Missing artifacts raise ``ArtifactsMissing`` (mapped to HTTP 503).
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from src.config import ACTIVE_ANOMALY_WINDOW_DAYS, SOURCE_LABELS, TARGET, Paths, classify_depth, get_logger
from src.data_summary import compute_data_summary
from src.inference import compute_all_forecasts
from src.models.xgboost_forecaster import XGBForecaster
from src.preprocessing.features import add_targets, build_features

log = get_logger("service")
SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class ArtifactsMissing(RuntimeError):
    """Raised when pipeline outputs / model files are not present."""


class UnknownSensor(KeyError):
    """Raised for an unknown sensor_id."""


class AquaSenseService:
    def __init__(self, source: str | None = None) -> None:
        self.paths = Paths(source or os.environ.get("AQUASENSE_SOURCE", "synthetic"))
        p = self.paths
        needed = [p.clean_csv, p.outputs_dir / "anomalies.csv", p.outputs_dir / "spatial_grid.geojson"]
        missing = [str(x) for x in needed if not x.exists()]
        if missing:
            raise ArtifactsMissing("Missing: " + ", ".join(missing) + ". Run `python scripts/run_pipeline.py` first.")
        self.clean = pd.read_csv(p.clean_csv, parse_dates=["date"])
        self.anomalies = pd.read_csv(p.outputs_dir / "anomalies.csv", parse_dates=["date"])
        self.grid = json.loads((p.outputs_dir / "spatial_grid.geojson").read_text())
        self.models_loaded = {"lstm": False, "xgb_30d": False, "xgb_90d": False, "anomaly": (p.models_dir / "anomaly_model.joblib").exists()}
        lstm, x30, x90 = self._load_models()
        feats = add_targets(build_features(self.clean))
        self.forecasts = compute_all_forecasts(self.clean, feats, lstm, x30, x90, extra_models=self._load_lgbm(),
                                               sequence_models=self._load_patchtst())  # inference only
        self.data_end = self.clean["date"].max()
        self.names: dict[str, str] = {}
        if p.stations_csv.exists():
            st = pd.read_csv(p.stations_csv)
            if "name" in st.columns:
                self.names = {str(a): str(b) for a, b in zip(st["sensor_id"], st["name"].fillna(""))}
        sj = p.outputs_dir / "data_summary.json"
        self.summary = json.loads(sj.read_text()) if sj.exists() else compute_data_summary(self.clean, self.paths.source)

    def _load_lgbm(self) -> dict[str, Any]:
        """Optional LightGBM models (only if scripts/train_lightgbm.py produced them and lightgbm is installed)."""
        out: dict[str, Any] = {}
        for h in (30, 90):
            try:
                from src.models.lightgbm_forecaster import LGBMForecaster
                out[f"lgbm_{h}d"] = LGBMForecaster.load(self.paths.models_dir, h)
                self.models_loaded[f"lgbm_{h}d"] = True
            except (FileNotFoundError, ImportError, ValueError) as exc:
                log.info("LightGBM %dd not loaded (optional): %s", h, exc)
        return out

    def _load_patchtst(self) -> dict[str, Any]:
        """Optional PatchTST models (only if scripts/train_patchtst.py produced them and PyTorch is installed)."""
        out: dict[str, Any] = {}
        for h in (7, 30, 90):
            if not (self.paths.models_dir / f"patchtst_{h}d.pt").exists():
                continue
            try:
                from src.models.patchtst_forecaster import PatchTSTForecaster
                out[f"patchtst_{h}d"] = PatchTSTForecaster.load(self.paths.models_dir, h)
                self.models_loaded[f"patchtst_{h}d"] = True
            except (FileNotFoundError, ImportError, ValueError) as exc:
                log.info("PatchTST %dd not loaded (optional): %s", h, exc)
        return out

    def _load_models(self) -> tuple[Any, XGBForecaster | None, XGBForecaster | None]:
        md = self.paths.models_dir
        x = {}
        for h in (30, 90):
            try:
                x[h] = XGBForecaster.load(md, h)
                self.models_loaded[f"xgb_{h}d"] = True
            except (FileNotFoundError, ImportError) as exc:
                log.warning("XGB %dd not loaded: %s", h, exc)
                x[h] = None
        lstm = None
        try:
            from src.models.lstm_forecaster import LSTMForecaster
            lstm = LSTMForecaster.load(md / "lstm_model.pt")
            self.models_loaded["lstm"] = True
        except (ImportError, FileNotFoundError) as exc:
            log.warning("LSTM not loaded: %s", exc)
        return lstm, x[30], x[90]

    # ---- queries ---------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {"status": "ok", "source": self.paths.source, "source_label": SOURCE_LABELS[self.paths.source],
                "models_loaded": self.models_loaded,
                "n_sensors": int(self.clean["sensor_id"].nunique()), "data_end_date": str(self.data_end.date())}

    def _active(self) -> pd.DataFrame:
        cutoff = self.data_end - pd.Timedelta(days=ACTIVE_ANOMALY_WINDOW_DAYS)
        return self.anomalies[self.anomalies["date"] > cutoff]

    def sensors(self) -> list[dict[str, Any]]:
        act_df = self._active()
        act = act_df.groupby("sensor_id").size().to_dict()
        top_sev = {sid: max(g["severity"], key=lambda x: SEV_ORDER.get(x, -1)) for sid, g in act_df.groupby("sensor_id")}
        latest_score = self.anomalies.sort_values("date").groupby("sensor_id").tail(1).set_index("sensor_id")["anomaly_score"].to_dict()
        out = []
        for sid, g in self.clean.groupby("sensor_id", sort=True):
            g = g.dropna(subset=[TARGET]).sort_values("date")
            if g.empty:
                continue
            r = g.iloc[-1]
            out.append({"sensor_id": sid, "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
                        "latest_depth_m_bgl": round(float(r[TARGET]), 2), "latest_date": str(r["date"].date()),
                        "status": classify_depth(float(r[TARGET])), "active_anomalies": int(act.get(sid, 0)),
                        "name": self.names.get(sid) or None, "max_active_severity": top_sev.get(sid),
                        "latest_anomaly_score": float(latest_score[sid]) if sid in latest_score else None})
        return out

    def forecast(self, sensor_id: str, history_days: int = 180) -> dict[str, Any]:
        if sensor_id not in self.forecasts:
            raise UnknownSensor(sensor_id)
        g = self.clean[self.clean["sensor_id"] == sensor_id].sort_values("date").tail(history_days)
        hist = [{"date": str(d.date()), "depth_m_bgl": None if pd.isna(v) else round(float(v), 3)}
                for d, v in zip(g["date"], g[TARGET])]
        f = self.forecasts[sensor_id]
        return {"sensor_id": sensor_id, "units": f["units"], "last_observation_date": f["last_observation_date"],
                "history": hist, "lstm_7d": f["lstm_7d"], "xgb_30d": f["xgb_30d"], "xgb_90d": f["xgb_90d"],
                "lgbm_30d": f.get("lgbm_30d"), "lgbm_90d": f.get("lgbm_90d"),
                "patchtst_7d": f.get("patchtst_7d"), "patchtst_30d": f.get("patchtst_30d"), "patchtst_90d": f.get("patchtst_90d"),
                "notes": ["Higher m bgl = deeper groundwater (5 m bgl = water table 5 m below the ground surface).",
                          "XGBoost curves are linear interpolation to a direct end-point forecast (prototype). Bands are approximate empirical "
                          "90% intervals (split-conformal-style) - coverage is not guaranteed for autocorrelated series."]}

    def anomalies_list(self, limit: int = 100, min_severity: str | None = None, sensor_id: str | None = None) -> list[dict[str, Any]]:
        a = self.anomalies
        if sensor_id:
            if sensor_id not in set(self.clean["sensor_id"]):
                raise UnknownSensor(sensor_id)
            a = a[a["sensor_id"] == sensor_id]
        if min_severity:
            a = a[a["severity"].map(SEV_ORDER).fillna(-1) >= SEV_ORDER[min_severity]]
        a = a.sort_values(["date", "anomaly_score"], ascending=[False, False]).head(limit)
        return [{"sensor_id": r.sensor_id, "date": str(r.date.date()), "anomaly_score": float(r.anomaly_score),
                 "severity": r.severity, "reason": r.reason} for r in a.itertuples()]

    def spatial_grid(self) -> dict[str, Any]:
        return self.grid

    def data_summary(self) -> dict[str, Any]:
        return self.summary
