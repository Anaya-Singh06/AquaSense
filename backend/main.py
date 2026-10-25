"""FastAPI application. Run from the repo root:  uvicorn backend.main:app --reload --port 8000

Models are loaded from files in ``models/`` at startup; the API never trains anything.
Set AQUASENSE_SOURCE=real to serve the real-data artifacts.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.evaluation import EvaluationUnavailable, load_evaluation, load_station_evaluation  # noqa: E402
from backend.schemas import Anomaly, DataSummary, ForecastResponse, Health, Sensor  # noqa: E402
from backend.service import AquaSenseService, ArtifactsMissing, UnknownSensor  # noqa: E402

STATE: dict[str, Any] = {"svc": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["svc"] = AquaSenseService()
    except ArtifactsMissing as exc:
        STATE["error"] = str(exc)  # API starts, endpoints answer 503 with a helpful message
    yield


app = FastAPI(title="AquaSense API", version="1.0", lifespan=lifespan,
              description="Groundwater intelligence & early-warning prototype. Depth is in metres below ground level (higher = deeper).")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


def _svc() -> AquaSenseService:
    if STATE["svc"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "service not initialised")
    return STATE["svc"]


@app.get("/api/health", response_model=Health)
def health() -> dict[str, Any]:
    return _svc().health()


@app.get("/api/sensors", response_model=list[Sensor])
def sensors() -> list[dict[str, Any]]:
    return _svc().sensors()


@app.get("/api/forecast/{sensor_id}", response_model=ForecastResponse)
def forecast(sensor_id: str, history_days: int = Query(180, ge=7, le=1000)) -> dict[str, Any]:
    try:
        return _svc().forecast(sensor_id, history_days)
    except UnknownSensor:
        raise HTTPException(status_code=404, detail=f"unknown sensor_id '{sensor_id}'") from None


@app.get("/api/anomalies", response_model=list[Anomaly])
def anomalies(limit: int = Query(100, ge=1, le=1000),
              min_severity: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
              sensor_id: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        return _svc().anomalies_list(limit, min_severity, sensor_id)
    except UnknownSensor:
        raise HTTPException(status_code=404, detail=f"unknown sensor_id '{sensor_id}'") from None


@app.get("/api/spatial-grid")
def spatial_grid() -> dict[str, Any]:
    return _svc().spatial_grid()


@app.get("/api/data-summary", response_model=DataSummary)
def data_summary() -> dict[str, Any]:
    """Data source, station count, date range, observation count and missingness of the ACTIVE dataset."""
    return _svc().data_summary()


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    """Completed offline rolling-origin evaluation results (static JSON). Independent of models and the data service."""
    try:
        return load_evaluation()
    except EvaluationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.get("/api/evaluation/stations")
def evaluation_stations() -> dict[str, Any]:
    """Per-station PatchTST and LightGBM scores for the map. Independent of operational models."""
    try:
        return load_station_evaluation()
    except EvaluationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
