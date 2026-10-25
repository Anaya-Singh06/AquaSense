"""Expanding-window (rolling-origin) evaluation of persistence, seasonal-naive, XGBoost and LSTM.

Train -> forecast -> advance the origin -> retrain -> forecast ...
For fold k the origin is ``val_end_k``: models train on targets up to ``train_end_k`` (= origin - val_days), early-stop /
calibrate on the validation block, and are then scored on target dates in (origin, origin + test_days]. Origins advance by
``test_days`` so test blocks tile the end of the record without overlap. Hyper-parameters are FIXED in the model classes;
nothing here is tuned on test data.

Scoring rule shared by EVERY model (tabular and sequence): a forecast is scored on its day-h end point, and belongs to a fold iff
its TARGET date lies in (origin, test_end]. The forecast origin itself may therefore lie up to h-1 days before the fold origin
(its inputs are still strictly in the past of the origin). ``rolling_sequence`` used to start at origin >= fold origin, which
silently dropped the first h-1 target days of every fold for LSTM / PatchTST (see ``include_straddling_origins``).
Which rows are comparable ACROSS models is decided in ``src.evaluation.rolling_comparison`` (common-sample intersection).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET, get_logger
from src.evaluation.baselines import persistence, seasonal_naive, seasonal_reference
from src.evaluation.metrics import mae, nse, rmse
from src.models.xgboost_forecaster import XGBForecaster, split_rows
from src.preprocessing.splits import SplitDates

log = get_logger("rolling")


@dataclass(frozen=True)
class Fold:
    index: int
    splits: SplitDates       # start, train_end, val_end(=origin); ``end`` = last test target date
    test_end: pd.Timestamp


def default_rolling_params(n_days: int) -> dict[str, int]:
    """Fold layout chosen from the record length only (never from results)."""
    if n_days >= 3000:
        return dict(n_folds=5, test_days=180, val_days=180, min_train_days=1095)
    return dict(n_folds=4, test_days=60, val_days=90, min_train_days=365)


def make_rolling_folds(dates: pd.Series | pd.DatetimeIndex, n_folds: int, test_days: int, val_days: int,
                       min_train_days: int) -> list[Fold]:
    u = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).unique())).sort_values()
    end = u[-1]
    folds: list[Fold] = []
    for k in range(n_folds):
        origin = end - pd.Timedelta(days=(n_folds - k) * test_days)
        train_end = origin - pd.Timedelta(days=val_days)
        if (train_end - u[0]).days < min_train_days:
            continue  # too little history for this origin: skip rather than evaluate a starved model
        folds.append(Fold(len(folds), SplitDates(u[0], train_end, origin, origin + pd.Timedelta(days=test_days)),
                          origin + pd.Timedelta(days=test_days)))
    if not folds:
        raise ValueError("Record too short for rolling-origin evaluation with the requested layout")
    return folds


def rolling_xgb(feats: pd.DataFrame, clean: pd.DataFrame, folds: list[Fold], horizons: tuple[int, ...] = (30, 90),
                forecaster_cls: type = XGBForecaster, model_name: str = "xgboost", forecaster_kwargs: dict | None = None,
                diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """Long frame with one row per (fold, sensor, horizon, origin day).

    Defaults reproduce the XGBoost evaluation exactly; ``rolling_lightgbm`` reuses this loop (same folds, same ``split_rows``).
    ``diagnostics`` (optional list) receives one dict per (fold, horizon): row counts, backend, best iteration, or the skip reason."""
    ref = seasonal_reference(clean)
    out = []
    for f in folds:
        for h in horizons:
            parts = split_rows(feats, h, f.splits)
            test = parts["test"]
            tdate = test["date"] + pd.Timedelta(days=h)
            test = test[(tdate <= f.test_end) & (test[f"target_{h}d_observed"] == 1)]
            try:
                m = forecaster_cls(h, **(forecaster_kwargs or {})).fit(parts["train"], parts["val"])
            except ValueError as exc:
                log.warning("fold %d h=%d skipped: %s", f.index, h, exc)
                if diagnostics is not None:
                    diagnostics.append({"fold": f.index, "model": model_name, "horizon": h, "status": f"skipped: {exc}"})
                continue
            if diagnostics is not None:
                diagnostics.append({"fold": f.index, "model": model_name, "horizon": h, "status": "ok", "n_train": int(len(parts["train"])),
                                    "n_val": int(len(parts["val"])), "n_test_scored": int(len(test)), "backend": getattr(m, "backend", None),
                                    "best_iteration": getattr(m, "best_iteration", None)})
            if test.empty:
                continue
            pred, lo, hi = m.predict_interval(test)
            td = test["date"] + pd.Timedelta(days=h)
            out.append(pd.DataFrame({
                "fold": f.index, "model": model_name, "horizon": h, "sensor_id": test["sensor_id"].to_numpy(),
                "origin_date": test["date"].to_numpy(), "target_date": td.to_numpy(), "y_true": test[f"target_{h}d"].to_numpy(),
                "y_model": pred, "lo": lo, "hi": hi, "y_persistence": persistence(test[TARGET].to_numpy()),
                "y_seasonal_naive": seasonal_naive(ref, test["sensor_id"].to_numpy(), td)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rolling_lightgbm(feats: pd.DataFrame, clean: pd.DataFrame, folds: list[Fold], horizons: tuple[int, ...] = (30, 90),
                     config: dict | None = None, diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """LightGBM through the identical rolling-origin loop as XGBoost (raises ImportError if lightgbm is absent)."""
    from src.models.lightgbm_forecaster import LGBMForecaster  # noqa: PLC0415
    return rolling_xgb(feats, clean, folds, horizons, forecaster_cls=LGBMForecaster, model_name="lightgbm",
                       forecaster_kwargs={"config": config} if config else None, diagnostics=diagnostics)


def _score_sequence_windows(model: Any, w: dict[str, Any], fold: Fold, h: int, ref: pd.Series,
                            include_straddling_origins: bool = True) -> pd.DataFrame:
    """Day-h end-point forecasts of a fitted sequence model on the windows whose TARGET date falls in (val_end, test_end].

    ``make_windows`` labels a window 'test' only if its FIRST target day is after the fold origin, which would start the day-h
    scoring at origin + h, whereas the tabular models are scored from origin + 1. With ``include_straddling_origins`` (default) windows
    that straddle the origin (labelled 'discard' for training purposes) are scored too, when their day-h target is after the origin:
    their inputs end at or before the origin, so nothing from the test block is seen, and they were never used for fitting or early
    stopping. ``False`` restores the legacy rule (origin >= fold origin).
    Every input day and every one of the h target days of a scored window is a real observation (``make_windows``)."""
    meta = w["meta"]
    od = pd.to_datetime(meta["origin_date"])
    td = od + pd.Timedelta(days=h)
    if include_straddling_origins:
        in_block = ((td > fold.splits.val_end) & (td <= fold.test_end)).to_numpy()
    else:
        in_block = ((meta["split"].to_numpy() == "test") & (td <= fold.test_end).to_numpy())
    m = in_block & np.asarray(w["obs"]).all(axis=1)
    if not m.any():
        return pd.DataFrame()
    X = w["X"][m]
    pred = model._inv_depth(model._predict_scaled(X))
    true = model._inv_depth(w["Y"][m])
    last = model._inv_depth(X[:, -1, 0])          # depth on the origin day = persistence forecast
    sid = meta["sensor_id"].to_numpy()[m]
    tdm = td[m]
    return pd.DataFrame({"sensor_id": sid, "origin_date": od[m].to_numpy(), "target_date": tdm.to_numpy(), "y_true": true[:, -1],
                         "y_model": pred[:, -1], "lo": np.nan, "hi": np.nan, "y_persistence": last,
                         "y_seasonal_naive": seasonal_naive(ref, sid, tdm)})


def rolling_sequence(clean: pd.DataFrame, folds: list[Fold], horizons: tuple[int, ...], make_model: Any, model_name: str,
                     include_straddling_origins: bool = True, diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """Shared rolling-origin loop for window-based (sequence) models: ``make_model(h)`` -> object with ``fit(clean, splits)`` returning the
    ``make_windows`` dict, plus ``_predict_scaled`` / ``_inv_depth``. Each fold builds a NEW model: scalers, early stopping and weights
    only ever see data up to the fold's validation end."""
    ref = seasonal_reference(clean)
    out = []
    for f in folds:
        for h in horizons:
            m = make_model(h)
            try:
                w = m.fit(clean, f.splits)
            except ValueError as exc:
                log.warning("%s fold %d h=%d skipped: %s", model_name, f.index, h, exc)
                if diagnostics is not None:
                    diagnostics.append({"fold": f.index, "model": model_name, "horizon": h, "status": f"skipped: {exc}"})
                continue
            rows = _score_sequence_windows(m, w, f, h, ref, include_straddling_origins)
            if diagnostics is not None:
                ws = getattr(m, "window_stats", {}) or {}
                diagnostics.append({"fold": f.index, "model": model_name, "horizon": h, "status": "ok",
                                    "n_train": ws.get("train", {}).get("retained"), "n_val": ws.get("val", {}).get("retained"),
                                    "n_test_scored": int(len(rows)), "epochs_run": len(getattr(m, "history", []) or []),
                                    "best_epoch": getattr(m, "best_epoch", None), "best_val_loss": getattr(m, "best_val_loss", None),
                                    "persistence_val_loss": getattr(m, "persistence_val_loss", None),
                                    "device": (getattr(m, "device_info", {}) or {}).get("device")})
            if rows.empty:
                continue
            out.append(rows.assign(fold=f.index, model=model_name, horizon=h)[
                ["fold", "model", "horizon", "sensor_id", "origin_date", "target_date", "y_true", "y_model", "lo", "hi", "y_persistence", "y_seasonal_naive"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rolling_lstm(clean: pd.DataFrame, folds: list[Fold], max_epochs: int = 40, include_straddling_origins: bool = True,
                 diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """7-day LSTM, evaluated on the day-7 value of each window. Requires PyTorch (raises ImportError if absent)."""
    from src.models.lstm_forecaster import HORIZON, LSTMForecaster  # noqa: PLC0415
    return rolling_sequence(clean, folds, (HORIZON,), lambda h: LSTMForecaster(max_epochs=max_epochs), "lstm",
                            include_straddling_origins, diagnostics)


def rolling_patchtst(clean: pd.DataFrame, folds: list[Fold], horizons: tuple[int, ...] = (7,), config: Any | None = None,
                     max_epochs: int | None = None, device: str = "auto", include_straddling_origins: bool = True,
                     diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """PatchTST through the same folds as the other models; scored on the day-h end point of each window (as ``rolling_lstm``).

    Windows need every input and target day observed, so the sample set differs from the XGBoost/LightGBM rows and (because of
    the longer context) from the LSTM rows - use ``rolling_comparison`` (common-sample intersection) for cross-model comparisons.
    Requires PyTorch (raises ImportError if absent)."""
    from src.models.patchtst_forecaster import PatchTSTConfig, PatchTSTForecaster, replace_config  # noqa: PLC0415
    cfg = config or PatchTSTConfig()
    if max_epochs is not None:
        cfg = replace_config(cfg, max_epochs=max_epochs)
    return rolling_sequence(clean, folds, tuple(horizons), lambda h: PatchTSTForecaster(h, cfg, device), "patchtst",
                            include_straddling_origins, diagnostics)


def rolling_itransformer(clean: pd.DataFrame, folds: list[Fold], horizons: tuple[int, ...] = (7,), config: Any | None = None,
                         max_epochs: int | None = None, device: str = "auto", include_straddling_origins: bool = True,
                         diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """iTransformer through the same folds as the other models; scored on the day-h end point of each window.

    Uses the same window rules as PatchTST (context 96 days, every input and target day observed), so both Transformer
    variants share a sample set - use ``rolling_comparison`` (common-sample intersection) for cross-model comparisons.
    Requires PyTorch (raises ImportError if absent)."""
    from src.models.itransformer_forecaster import ITransformerConfig, ITransformerForecaster, replace_config  # noqa: PLC0415
    cfg = config or ITransformerConfig()
    if max_epochs is not None:
        cfg = replace_config(cfg, max_epochs=max_epochs)
    return rolling_sequence(clean, folds, tuple(horizons), lambda h: ITransformerForecaster(h, cfg, device), "itransformer",
                            include_straddling_origins, diagnostics)


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "nse_pooled": nse(y, p)}


def summarize(long: pd.DataFrame) -> dict[str, Any]:
    """Pooled, per-station and median-station metrics for model / persistence / seasonal-naive on IDENTICAL samples."""
    res: dict[str, Any] = {}
    if long.empty:
        return res
    for (model, h), g in long.groupby(["model", "horizon"]):
        valid = g[np.isfinite(g["y_seasonal_naive"]) & np.isfinite(g["y_model"])]
        entry: dict[str, Any] = {"model": model, "horizon_days": int(h), "n_samples_total": int(len(g)), "n_samples_compared": int(len(valid)),
                                 "n_stations": int(valid["sensor_id"].nunique()), "n_folds": int(valid["fold"].nunique())}
        if valid.empty:
            res[f"{model}_{h}d"] = {**entry, "note": "no comparable samples"}
            continue
        y = valid["y_true"].to_numpy()
        per_station: dict[str, dict[str, Any]] = {}
        for name, col in (("model", "y_model"), ("persistence", "y_persistence"), ("seasonal_naive", "y_seasonal_naive")):
            entry[name] = {"pooled": _scores(y, valid[col].to_numpy())}
            rows = []
            for sid, s in valid.groupby("sensor_id"):
                if len(s) < 10:
                    continue
                sc = _scores(s["y_true"].to_numpy(), s[col].to_numpy())
                rows.append({"sensor_id": sid, "n": int(len(s)), **sc})
                per_station.setdefault(sid, {"n": int(len(s))})[name] = sc
            df = pd.DataFrame(rows)
            if len(df):
                entry[name]["median_station"] = {k: float(np.nanmedian(df[k])) for k in ("mae", "rmse", "nse_pooled")}
                entry[name]["median_station"]["nse"] = entry[name]["median_station"].pop("nse_pooled")
                entry[name]["mean_station_nse"] = float(np.nanmean(df["nse_pooled"]))
        for ref_name in ("persistence", "seasonal_naive"):
            wins = [v["model"]["rmse"] < v[ref_name]["rmse"] for v in per_station.values() if "model" in v and ref_name in v]
            entry[f"stations_where_model_beats_{ref_name}"] = {"n_better": int(sum(wins)), "n_stations": len(wins)}
            entry[f"rmse_skill_vs_{ref_name}_pooled"] = float(1 - entry["model"]["pooled"]["rmse"] / entry[ref_name]["pooled"]["rmse"])
        if valid["lo"].notna().any():
            cover = ((valid["y_true"] >= valid["lo"]) & (valid["y_true"] <= valid["hi"])).to_numpy()
            entry["interval"] = {"nominal": 0.90, "empirical_coverage": float(cover.mean()),
                                 "mean_halfwidth_m": float(((valid["hi"] - valid["lo"]) / 2).mean()),
                                 "coverage_by_fold": {int(k): float(v) for k, v in
                                                      pd.Series(cover, index=valid.index).groupby(valid["fold"]).mean().items()}}
        entry["by_fold_rmse"] = {int(k): {"model": rmse(s["y_true"], s["y_model"]), "persistence": rmse(s["y_true"], s["y_persistence"]),
                                          "seasonal_naive": rmse(s["y_true"], s["y_seasonal_naive"])} for k, s in valid.groupby("fold")}
        entry["per_station"] = per_station
        res[f"{model}_{h}d"] = entry
    return res


MODEL_LABELS = {"lstm": "LSTM", "xgboost": "XGBoost", "lightgbm": "LightGBM", "patchtst": "PatchTST",
                "itransformer": "iTransformer"}


def metrics_table(summary: dict[str, Any], dataset: str) -> pd.DataFrame:
    """Flat table: Dataset | Model | Horizon | MAE | RMSE | NSE (pooled) | median-station NSE | n."""
    rows = []
    for key, e in summary.items():
        if "model" not in e or not isinstance(e["model"], dict):
            continue
        h = e["horizon_days"]
        for label, name in ((MODEL_LABELS.get(key.rsplit("_", 1)[0], key), "model"),
                            ("Persistence", "persistence"), ("Seasonal naive", "seasonal_naive")):
            b = e[name]
            rows.append({"Dataset": dataset, "Model": label, "Horizon": f"{h}d", "MAE": b["pooled"]["mae"], "RMSE": b["pooled"]["rmse"],
                         "NSE_pooled": b["pooled"]["nse_pooled"], "NSE_median_station": b.get("median_station", {}).get("nse", np.nan),
                         "n": e["n_samples_compared"], "stations": e["n_stations"], "folds": e["n_folds"]})
    t = pd.DataFrame(rows)
    return t.drop_duplicates(["Model", "Horizon"]).sort_values(["Horizon", "Model"], key=lambda c: c.map(lambda x: (len(str(x)), str(x))) if c.name == "Horizon" else c) if len(t) else t
