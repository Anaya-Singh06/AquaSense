"""Shared helpers for tests."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def allow_xgb_standin() -> None:
    """If xgboost is missing, let tests use the labelled sklearn stand-in (sandbox only)."""
    if not has("xgboost"):
        os.environ["AQUASENSE_ALLOW_SKLEARN_FALLBACK"] = "1"
