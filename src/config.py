"""Central configuration: paths, constants, seeds and logging for AquaSense."""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SEED: int = 42

# ---- Common schema (identical for synthetic and real data) -------------------
TARGET: str = "groundwater_depth_m_bgl"  # metres below ground level (higher = deeper water)
LABEL_COLUMN: str = "is_synthetic_anomalous_extraction"  # synthetic data ONLY
CLIMATE_COLUMNS: list[str] = ["rainfall_mm", "temperature_c", "evapotranspiration_mm", "soil_moisture"]
COMMON_COLUMNS: list[str] = [
    "sensor_id", "latitude", "longitude", "date", TARGET, *CLIMATE_COLUMNS,
]

# ---- Prototype thresholds (NOT official regulatory classes) -------------------
# Depth to water (m bgl) -> status. Only for demonstration of IDW / map colouring.
STATUS_SEMI_CRITICAL_M: float = 10.0
STATUS_CRITICAL_M: float = 20.0

ACTIVE_ANOMALY_WINDOW_DAYS: int = 30  # "active" = flagged within this many days of the last data date

# Validation ranges used by preprocessing (values outside become NaN).
VALID_RANGES: dict[str, tuple[float, float]] = {
    TARGET: (0.0, 500.0),
    "rainfall_mm": (0.0, 1000.0),
    "temperature_c": (-60.0, 60.0),
    "evapotranspiration_mm": (0.0, 30.0),
    "soil_moisture": (0.0, 1.0),
}


SOURCE_LABELS: dict[str, str] = {"synthetic": "Synthetic demonstration", "real": "USGS + NASA POWER",
                                 "real_expanded": "USGS + NASA POWER (expanded station set)"}
SOURCE_BADGES: dict[str, str] = {"synthetic": "SYNTHETIC DEMO", "real": "USGS + NASA POWER",
                                 "real_expanded": "USGS + NASA POWER (EXPANDED)"}

# One data source = one dataset + its own model/output folders. ``real_expanded`` is a SEPARATE, larger USGS/NASA
# dataset (built by scripts/run_expanded_pipeline.py): its files never collide with the original ``real`` ones.
SOURCES: tuple[str, ...] = ("synthetic", "real", "real_expanded")
DATA_FILES: dict[str, str] = {"synthetic": "dwlr_data.csv", "real": "real_data.csv", "real_expanded": "real_data_expanded.csv"}
STATION_FILES: dict[str, str] = {"synthetic": "stations.csv", "real": "real_stations.csv", "real_expanded": "real_stations_expanded.csv"}


@dataclass(frozen=True)
class Paths:
    """All file locations for one data source ('synthetic', 'real' or 'real_expanded')."""

    source: str = "synthetic"
    root: Path = REPO_ROOT

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")

    @property
    def data_csv(self) -> Path:
        return self.root / "data" / DATA_FILES[self.source]

    @property
    def stations_csv(self) -> Path:
        return self.root / "data" / STATION_FILES[self.source]

    @property
    def clean_csv(self) -> Path:
        return self.root / "data" / "processed" / f"{self.source}_clean.csv"

    @property
    def features_csv(self) -> Path:
        return self.root / "data" / "processed" / f"{self.source}_features.csv"

    @property
    def models_dir(self) -> Path:
        return self.root / "models" if self.source == "synthetic" else self.root / "models" / self.source

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs" if self.source == "synthetic" else self.root / "outputs" / self.source

    def ensure_dirs(self) -> None:
        for p in (self.clean_csv.parent, self.models_dir, self.outputs_dir):
            p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch. Full bit-determinism is not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a one-time basic configuration."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def classify_depth(depth_m: float, semi: float = STATUS_SEMI_CRITICAL_M, crit: float = STATUS_CRITICAL_M) -> str:
    """Prototype status class from depth (m bgl): SAFE / SEMI_CRITICAL / CRITICAL."""
    if depth_m >= crit:
        return "CRITICAL"
    if depth_m >= semi:
        return "SEMI_CRITICAL"
    return "SAFE"
