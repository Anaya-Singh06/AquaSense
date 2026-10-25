"""Pydantic response models for the AquaSense API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    source: str
    source_label: str = ""
    models_loaded: dict[str, bool]
    n_sensors: int
    data_end_date: str


class Sensor(BaseModel):
    sensor_id: str
    latitude: float
    longitude: float
    latest_depth_m_bgl: float = Field(description="metres below ground level; higher = deeper")
    latest_date: str
    status: str = Field(description="PROTOTYPE class: SAFE / SEMI_CRITICAL / CRITICAL")
    active_anomalies: int
    name: Optional[str] = None
    max_active_severity: Optional[str] = None
    latest_anomaly_score: Optional[float] = None


class HistoryPoint(BaseModel):
    date: str
    depth_m_bgl: Optional[float] = None


class ForecastPoint(BaseModel):
    date: str
    depth_m_bgl: float


class BandPoint(ForecastPoint):
    lower: float
    upper: float


class XGBForecast(BaseModel):
    horizon_days: int
    method: str
    interval_type: str
    endpoint_depth_m_bgl: float
    endpoint_lower: float
    endpoint_upper: float
    trajectory: list[BandPoint]


class ForecastResponse(BaseModel):
    sensor_id: str
    units: str
    last_observation_date: str
    history: list[HistoryPoint]
    lstm_7d: Optional[list[ForecastPoint]] = None
    xgb_30d: Optional[XGBForecast] = None
    xgb_90d: Optional[XGBForecast] = None
    lgbm_30d: Optional[XGBForecast] = None   # optional LightGBM (same shape as XGBoost forecasts)
    lgbm_90d: Optional[XGBForecast] = None
    patchtst_7d: Optional[list[ForecastPoint]] = None    # optional PatchTST daily trajectories (no interval bands)
    patchtst_30d: Optional[list[ForecastPoint]] = None
    patchtst_90d: Optional[list[ForecastPoint]] = None
    notes: list[str]


class Anomaly(BaseModel):
    sensor_id: str
    date: str
    anomaly_score: float
    severity: str
    reason: str


class DataSummary(BaseModel):
    source: str
    source_label: str
    source_badge: str
    station_count: int
    date_range: list[str]
    observation_count: int
    calendar_rows: int
    missing_fraction: float
    missing_fraction_by_station: dict[str, float]
    imputed_fraction: Optional[float] = None
    has_synthetic_labels: bool
    units: str
    note: str
