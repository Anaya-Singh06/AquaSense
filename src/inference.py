"""Shared inference helpers used by the pipeline (writes outputs) and by the API (reads artifacts)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import TARGET, get_logger
from src.models.xgboost_forecaster import XGBForecaster
from src.preprocessing.features import FEATURE_COLUMNS

log = get_logger("inference")


def compute_all_forecasts(clean: pd.DataFrame, feats: pd.DataFrame, lstm: Any | None,
                          xgb30: XGBForecaster | None, xgb90: XGBForecaster | None,
                          extra_models: dict[str, XGBForecaster | None] | None = None,
                          sequence_models: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Forecast every sensor from its latest data. Missing models simply yield missing keys (no retraining).

    ``extra_models`` maps an output key (e.g. ``"lgbm_30d"``) to any direct forecaster with the XGBForecaster interface.
    ``sequence_models`` maps an output key (e.g. ``"patchtst_7d"``) to a forecaster with the LSTM interface
    (``forecast_latest(clean) -> {sensor_id: [{date, depth_m_bgl}, ...]}``); sensors it skips get ``None``."""
    lstm_out = lstm.forecast_latest(clean) if lstm is not None else {}
    seq_out = {name: mdl.forecast_latest(clean) for name, mdl in (sequence_models or {}).items() if mdl is not None}
    result: dict[str, dict[str, Any]] = {}
    for sid, g in feats.groupby("sensor_id", sort=True):
        g = g.sort_values("date")
        valid = g.dropna(subset=[*FEATURE_COLUMNS])
        entry: dict[str, Any] = {"units": "m below ground level (higher = deeper)",
                                 "last_observation_date": str(g["date"].iloc[-1].date()), "lstm_7d": lstm_out.get(sid)}
        for name, out in seq_out.items():
            entry[name] = out.get(sid)
        # use the latest row with complete features (usually the last day)
        latest = valid.tail(1)
        for name, mdl in (("xgb_30d", xgb30), ("xgb_90d", xgb90), *(extra_models or {}).items()):
            if mdl is None or latest.empty:
                entry[name] = None
                continue
            try:
                entry[name] = mdl.forecast_latest(latest)
            except ValueError as exc:
                log.warning("%s %s: %s", sid, name, exc)
                entry[name] = None
        result[sid] = entry
    return result
