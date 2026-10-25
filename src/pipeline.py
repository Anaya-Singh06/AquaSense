"""End-to-end orchestration: data -> preprocess -> features -> train/evaluate -> outputs.

Used by ``scripts/run_pipeline.py`` and ``scripts/train_all.py``.
"""
from __future__ import annotations

import importlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, REPO_ROOT, SOURCE_LABELS, TARGET, Paths, get_logger, set_seed
from src.data_summary import compute_data_summary
from src.evaluation import rolling_origin as ro
from src.evaluation.injection import evaluate_injection
from src.evaluation.sufficiency import overall_conclusion, sufficiency_table
from src.inference import compute_all_forecasts
from src.models.anomaly_detector import AnomalyDetector
from src.models.xgboost_forecaster import XGBForecaster, split_rows
from src.preprocessing.features import add_targets, build_features
from src.preprocessing.preprocess import load_dataset, preprocess
from src.preprocessing.splits import make_split_dates
from src.spatial.idw import build_spatial_grid

log = get_logger("pipeline")


def _clean_json(o: Any) -> Any:
    """Make nested structures JSON-safe (NaN -> null, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items() if k != "_arrays"}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _versions() -> dict[str, str]:
    v = {"python": platform.python_version()}
    for m in ("numpy", "pandas", "sklearn", "torch", "xgboost"):
        try:
            v[m] = importlib.import_module(m).__version__
        except ImportError:
            v[m] = "NOT INSTALLED"
    return v


def _fetch_real(paths: Paths, refresh: bool, offline: bool) -> None:
    """Build data/real_data.csv from USGS + NASA POWER (cached under data/raw/)."""
    from data.fetch_real_data import FetchError, RawCache, build_real_dataset, load_config
    try:
        df, st, report = build_real_dataset(load_config(), RawCache(refresh=refresh, offline=offline))
    except FetchError as exc:
        raise RuntimeError(f"Real-data download failed: {exc}\nRun `python scripts/check_usgs_api.py` and see the README.") from exc
    df.to_csv(paths.data_csv, index=False, date_format="%Y-%m-%d")
    st.to_csv(paths.stations_csv, index=False)
    (paths.outputs_dir / "station_selection.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("[1/8] real data: discovered=%d rejected=%d selected=%d", report["stations_discovered"], report["stations_rejected"], report["stations_selected"])


def run(source: str = "synthetic", regenerate: bool = True, skip_lstm: bool = False,
        lstm_epochs: int | None = None, refresh: bool = False, fetch: bool = True, offline: bool = False,
        rolling: bool = True) -> dict[str, Any]:
    """Run the full workflow for ``source`` ('synthetic' or 'real'). Returns the metrics dict.

    real: data/real_data.csv is built from USGS + NASA POWER when missing or when ``refresh`` is set (``fetch`` False
    forbids downloading; ``offline`` allows only cached raw files).
    """
    paths = Paths(source)
    paths.ensure_dirs()
    set_seed()
    # ---- 1. data -----------------------------------------------------------------
    if source == "synthetic" and (regenerate or not paths.data_csv.exists()):
        from data.generate_synthetic_data import generate
        df, st = generate()
        df.to_csv(paths.data_csv, index=False, date_format="%Y-%m-%d")
        st.to_csv(paths.stations_csv, index=False)
        log.info("[1/7] generated synthetic data -> %s", paths.data_csv)
    elif source == "real" and fetch and (refresh or not paths.data_csv.exists()):
        _fetch_real(paths, refresh, offline)
    elif not paths.data_csv.exists():
        raise FileNotFoundError(f"{paths.data_csv} not found. For real data run: python data/fetch_real_data.py")
    # ---- 2-3. preprocess + features -------------------------------------------------
    raw = load_dataset(paths.data_csv)
    clean, prep_report = preprocess(raw)
    clean.to_csv(paths.clean_csv, index=False, date_format="%Y-%m-%d")
    feats = add_targets(build_features(clean))
    feats.to_csv(paths.features_csv, index=False, date_format="%Y-%m-%d")
    splits = make_split_dates(clean["date"])
    log.info("[2-3/7] preprocessed + features; split dates %s", splits.as_dict())
    metrics: dict[str, Any] = {"source": source, "source_label": SOURCE_LABELS[source], "split_dates": splits.as_dict(),
                               "preprocessing": prep_report, "versions": _versions(), "notes": []}
    # ---- 3b. data sufficiency (thresholds fixed in advance in config/real_data_config.json) -------------------
    th = json.loads((REPO_ROOT / "config" / "real_data_config.json").read_text())["sufficiency"]
    suff = sufficiency_table(raw, raw["date"].pipe(pd.to_datetime).min(), raw["date"].pipe(pd.to_datetime).max(), th)
    suff.to_csv(paths.outputs_dir / "data_sufficiency.csv", index=False)
    metrics["data_sufficiency"] = overall_conclusion(suff)
    log.info("[3b] data sufficiency: %s", {k: metrics["data_sufficiency"][k] for k in ("answer", "PASS", "WARN", "FAIL")})
    gw_n = int(clean[TARGET].notna().sum())
    metrics["groundwater_observation_accounting"] = {
        "rows_total": int(len(clean)), "rows_with_groundwater_value": gw_n,
        "observed": int((clean["depth_imputed"] == 0).sum() - clean[TARGET].isna().sum()),
        "imputed_flag_1": int((clean["depth_imputed"] == 1).sum()), "imputed_flag_2": int((clean["depth_imputed"] == 2).sum()),
        "pct_interpolated_of_values": round(100 * float((clean["depth_imputed"] > 0).sum()) / max(gw_n, 1), 3),
        "stations_with_imputation": int(clean.loc[clean["depth_imputed"] > 0, "sensor_id"].nunique()),
        "imputed_rows_per_station": {k: int(v) for k, v in clean[clean["depth_imputed"] > 0].groupby("sensor_id").size().sort_values(ascending=False).items()},
        "rule": "Final model evaluation excludes interpolated groundwater values from model inputs and targets."}
    summary = compute_data_summary(clean, source)
    (paths.outputs_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- 4. LSTM -----------------------------------------------------------------
    lstm = None
    if skip_lstm:
        metrics["lstm_7d"] = {"status": "skipped (--skip-lstm)"}
    else:
        try:
            from src.models.lstm_forecaster import LSTMForecaster
        except ImportError as exc:
            metrics["lstm_7d"] = {"status": f"SKIPPED: PyTorch not installed ({exc})"}
            log.error("LSTM skipped: %s", exc)
        else:
            from src.evaluation.plots import plot_lstm_actual_vs_pred
            lstm = LSTMForecaster(max_epochs=lstm_epochs or 60)
            w = lstm.fit(clean, splits, paths.models_dir / "lstm_model.pt")
            res = lstm.evaluate(w, "test")
            plot_lstm_actual_vs_pred(res["_arrays"], paths.outputs_dir / "lstm_actual_vs_predicted.png")
            res["epochs_run"] = len(lstm.history)
            res["window_accounting"] = lstm.window_stats   # candidate = retained + dropped_imputed, per split
            res["device"] = lstm.device_info
            metrics["lstm_7d"] = res
            log.info("[4/7] LSTM test: model=%s persistence=%s", res["model"], res["persistence"])

    # ---- 5. XGBoost ----------------------------------------------------------------
    from src.evaluation.plots import plot_xgb_test
    xgbs: dict[int, XGBForecaster] = {}
    for h in (30, 90):
        xstats: dict = {}
        parts = split_rows(feats, h, splits, stats=xstats)
        m = XGBForecaster(h).fit(parts["train"], parts["val"])
        m.save(paths.models_dir)
        res = m.evaluate(parts["test"])
        plot_xgb_test(parts["test"], m.predict_level(parts["test"]), h, paths.outputs_dir / f"xgb_{h}d_actual_vs_predicted.png")
        res["rows"] = {k: len(v) for k, v in parts.items()}
        res["row_accounting"] = xstats   # candidate = retained + dropped_imputed, per split
        metrics[f"xgb_{h}d"] = res
        xgbs[h] = m
    log.info("[5/7] XGBoost trained")

    # ---- 5b. rolling-origin evaluation vs persistence + seasonal-naive ----------------------------------------
    if rolling:
        try:
            folds = ro.make_rolling_folds(clean["date"], **ro.default_rolling_params(clean["date"].nunique()))
            long = ro.rolling_xgb(feats, clean, folds)
            lstm_status = "skipped (--skip-lstm)" if skip_lstm else None
            if not skip_lstm:
                try:
                    long = pd.concat([long, ro.rolling_lstm(clean, folds, max_epochs=lstm_epochs or 40)], ignore_index=True)
                    lstm_status = "run"
                except ImportError as exc:
                    lstm_status = f"NOT RUN: PyTorch not installed ({exc})"
            summ = ro.summarize(long)
            table = ro.metrics_table(summ, SOURCE_LABELS[source])
            table.to_csv(paths.outputs_dir / "metrics_table.csv", index=False)
            (paths.outputs_dir / "evaluation.json").write_text(json.dumps(_clean_json(summ), indent=1))
            metrics["rolling_origin"] = {"folds": [{"fold": f.index, "train_end": str(f.splits.train_end.date()), "origin": str(f.splits.val_end.date()),
                                                    "test_end": str(f.test_end.date())} for f in folds],
                                         "lstm_status": lstm_status, "seasonal_naive_lag_days": 365,
                                         "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_station"} for k, v in summ.items()}}
            log.info("[5b] rolling-origin: %d folds, %d forecast rows", len(folds), len(long))
        except ValueError as exc:
            metrics["rolling_origin"] = {"status": f"NOT RUN: {exc}"}
            log.warning("rolling-origin evaluation not run: %s", exc)

    # ---- 6. anomalies ----------------------------------------------------------------
    from src.evaluation.plots import plot_anomalies
    det = AnomalyDetector().fit(feats[feats["date"] <= splits.train_end])
    det.save(paths.models_dir / "anomaly_model.joblib")
    scored = det.detect(feats)
    scored[scored["anomaly_flag"]].to_csv(paths.outputs_dir / "anomalies.csv", index=False, date_format="%Y-%m-%d")
    plot_anomalies(clean, scored, paths.outputs_dir / "anomalies.png")
    if LABEL_COLUMN in clean.columns:
        metrics["anomaly"] = {"ground_truth": "SYNTHETIC labels (controlled experiment)",
                              "test_period": AnomalyDetector.evaluate_synthetic(scored, clean, splits.val_end),
                              "all_dates": AnomalyDetector.evaluate_synthetic(scored, clean)}
    else:
        metrics["anomaly"] = {**AnomalyDetector.summarize_unlabelled(scored),   # real detections: counts/dates/severity only
                              "ground_truth": "NONE for real anomalies - no precision/recall/F1/ROC-AUC is reported for them",
                              "injected_evaluation": evaluate_injection(clean, det, splits)}
    log.info("[6/7] anomalies: %d flagged", int(scored["anomaly_flag"].sum()))

    # ---- 7. spatial + forecasts ---------------------------------------------------------
    last = (clean.dropna(subset=[TARGET]).sort_values("date").groupby("sensor_id").tail(1))
    as_of = str(clean["date"].max().date())
    grid = build_spatial_grid(last["latitude"].to_numpy(), last["longitude"].to_numpy(), last[TARGET].to_numpy(),
                              power=2.0, as_of=as_of)  # bounds, centre and mask distance are derived from the stations
    (paths.outputs_dir / "spatial_grid.geojson").write_text(json.dumps(grid))
    forecasts = compute_all_forecasts(clean, feats, lstm, xgbs[30], xgbs[90])
    (paths.outputs_dir / "forecasts.json").write_text(json.dumps(_clean_json(forecasts), indent=1))
    metrics["xgb_backend"] = xgbs[30].backend
    if xgbs[30].backend != "xgboost":
        metrics["notes"].append("XGBoost rows come from a scikit-learn STAND-IN, not XGBoost.")
    (paths.outputs_dir / "metrics.json").write_text(json.dumps(_clean_json(metrics), indent=2))
    log.info("[7/7] wrote spatial grid (%d cells), forecasts and metrics to %s", len(grid["features"]), paths.outputs_dir)
    return metrics


def run_lightgbm(source: str = "synthetic", rolling: bool = True, compare_xgb: bool = True,
                 lgbm_config: str | None = None) -> dict[str, Any]:
    """Add-on workflow: train/evaluate the LightGBM 30 d / 90 d models on the EXISTING data CSV.

    Reuses preprocess -> features -> ``split_rows`` -> rolling-origin folds exactly as ``run``. It never regenerates or
    downloads data and never overwrites existing artifacts: models are saved as ``lgbm_{h}d.txt`` / ``lgbm_{h}d_meta.json``
    in ``models_dir``; results as ``*_lightgbm.*`` / ``lgbm_*`` files in ``outputs_dir``. If ``compare_xgb`` the rolling-origin
    loop also re-runs XGBoost (same folds, same samples) so one table holds both; skipped with a note if xgboost is missing.
    """
    from src.evaluation.plots import plot_xgb_test
    from src.models.lightgbm_forecaster import LGBMForecaster, load_lgbm_config

    paths = Paths(source)
    paths.ensure_dirs()
    set_seed()
    if not paths.data_csv.exists():
        raise FileNotFoundError(f"{paths.data_csv} not found. Run scripts/run_pipeline.py (or data/fetch_real_data.py for real data) first.")
    cfg = load_lgbm_config(lgbm_config)
    clean, prep_report = preprocess(load_dataset(paths.data_csv))
    feats = add_targets(build_features(clean))
    splits = make_split_dates(clean["date"])
    metrics: dict[str, Any] = {"source": source, "source_label": SOURCE_LABELS[source], "split_dates": splits.as_dict(),
                               "preprocessing": prep_report, "versions": _versions(), "lightgbm_config": cfg, "notes": []}
    try:
        import lightgbm
        metrics["versions"]["lightgbm"] = lightgbm.__version__
    except ImportError:
        raise RuntimeError("lightgbm is not installed: pip install lightgbm") from None

    lgbms: dict[int, LGBMForecaster] = {}
    for h in (30, 90):
        stats: dict = {}
        parts = split_rows(feats, h, splits, stats=stats)
        m = LGBMForecaster(h, config=cfg).fit(parts["train"], parts["val"])
        m.save(paths.models_dir)
        res = m.evaluate(parts["test"])
        plot_xgb_test(parts["test"], m.predict_level(parts["test"]), h, paths.outputs_dir / f"lgbm_{h}d_actual_vs_predicted.png")
        res["rows"] = {k: len(v) for k, v in parts.items()}
        res["row_accounting"] = stats
        res["best_iteration"] = m.best_iteration
        metrics[f"lgbm_{h}d"] = res
        lgbms[h] = m
    log.info("LightGBM trained; models in %s", paths.models_dir)

    if rolling:
        try:
            folds = ro.make_rolling_folds(clean["date"], **ro.default_rolling_params(clean["date"].nunique()))
            parts_long = [ro.rolling_lightgbm(feats, clean, folds, config=cfg)]
            xgb_status = "not requested (--no-xgb-compare)"
            if compare_xgb:
                try:
                    parts_long.append(ro.rolling_xgb(feats, clean, folds))
                    xgb_status = "run"
                except ImportError as exc:
                    xgb_status = f"NOT RUN: xgboost not installed ({exc})"
            summ = ro.summarize(pd.concat(parts_long, ignore_index=True))
            ro.metrics_table(summ, SOURCE_LABELS[source]).to_csv(paths.outputs_dir / "metrics_table_lightgbm.csv", index=False)
            (paths.outputs_dir / "evaluation_lightgbm.json").write_text(json.dumps(_clean_json(summ), indent=1))
            metrics["rolling_origin"] = {"folds": [{"fold": f.index, "train_end": str(f.splits.train_end.date()), "origin": str(f.splits.val_end.date()),
                                                    "test_end": str(f.test_end.date())} for f in folds],
                                         "xgboost_comparison": xgb_status, "seasonal_naive_lag_days": 365,
                                         "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_station"} for k, v in summ.items()}}
        except ValueError as exc:
            metrics["rolling_origin"] = {"status": f"NOT RUN: {exc}"}
            log.warning("rolling-origin evaluation not run: %s", exc)

    forecasts = compute_all_forecasts(clean, feats, None, None, None, extra_models={f"lgbm_{h}d": lgbms[h] for h in (30, 90)})
    for v in forecasts.values():       # this file holds LightGBM only; XGBoost/LSTM forecasts stay in forecasts.json
        v.pop("xgb_30d", None), v.pop("xgb_90d", None), v.pop("lstm_7d", None)
    (paths.outputs_dir / "forecasts_lightgbm.json").write_text(json.dumps(_clean_json(forecasts), indent=1))
    (paths.outputs_dir / "metrics_lightgbm.json").write_text(json.dumps(_clean_json(metrics), indent=2))
    log.info("wrote LightGBM results to %s", paths.outputs_dir)
    return metrics


def run_patchtst(source: str = "synthetic", rolling: bool = True, compare_lstm: bool = True, horizons: list[int] | None = None,
                 patchtst_config: str | None = None, resume: bool = False, epochs: int | None = None,
                 device: str = "auto", run_dir: str | None = None) -> dict[str, Any]:
    """Add-on workflow: train/evaluate PatchTST on the EXISTING data CSV (never regenerates/downloads data).

    Reuses preprocess -> ``make_windows`` (observed-only windows) -> chronological splits -> rolling-origin folds exactly as the LSTM.
    Models are saved as ``patchtst_{h}d.pt|_scalers.joblib|_meta.json`` in ``models_dir``; results as ``*_patchtst.*`` /
    ``patchtst_*`` files in ``outputs_dir``. Existing LSTM / XGBoost / LightGBM artifacts are never touched. If ``compare_lstm`` and
    7 is among the horizons, the rolling-origin loop also re-runs the LSTM on the same folds so one table holds both (their sample
    sets still differ - see ``n`` in the table).

    ``run_dir`` (optional): isolated experiment. Models go to ``<run_dir>/models`` and results to ``<run_dir>/outputs``; the default
    ``models/`` and ``outputs/`` folders are not written. Refuses to run if ``run_dir`` already holds PatchTST files, so nothing is overwritten.
    """
    from src.evaluation.plots import plot_sequence_actual_vs_pred
    from src.models.patchtst_forecaster import PatchTSTForecaster, load_patchtst_config, replace_config

    paths = Paths(source)
    paths.ensure_dirs()
    models_dir, outputs_dir = paths.models_dir, paths.outputs_dir
    if run_dir is not None:
        models_dir, outputs_dir = Path(run_dir) / "models", Path(run_dir) / "outputs"
        existing = [*models_dir.glob("patchtst_*"), *outputs_dir.glob("*patchtst*"), *outputs_dir.glob("experiment_summary.json")]
        if existing:
            raise ValueError(f"--run-dir {run_dir} already contains PatchTST results ({existing[0].name}, ...); choose a new directory")
        models_dir.mkdir(parents=True, exist_ok=True); outputs_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    if not paths.data_csv.exists():
        raise FileNotFoundError(f"{paths.data_csv} not found. Run scripts/run_pipeline.py (or data/fetch_real_data.py for real data) first.")
    loaded = load_patchtst_config(patchtst_config)
    cfg = loaded["config"]
    horizons = sorted(set(horizons or loaded["horizons"]))
    if epochs is not None:
        cfg = replace_config(cfg, max_epochs=epochs)
    rolling_epochs = min(epochs, loaded["rolling_max_epochs"]) if epochs is not None else loaded["rolling_max_epochs"]
    clean, prep_report = preprocess(load_dataset(paths.data_csv))
    feats = add_targets(build_features(clean))
    splits = make_split_dates(clean["date"])
    metrics: dict[str, Any] = {"source": source, "source_label": SOURCE_LABELS[source], "split_dates": splits.as_dict(),
                               "preprocessing": prep_report, "versions": _versions(), "patchtst_config": asdict(cfg),
                               "horizons": horizons, "notes": [
                                   "Windows require every input and target day to be a real (non-interpolated) observation; windows never cross stations.",
                                   "PatchTST uses a longer context than the LSTM, so its evaluated sample set differs from the LSTM / XGBoost / LightGBM "
                                   "sets: compare models only through the rolling-origin table (same folds) and read the sample counts 'n' beside every number."]}

    seq_models: dict[str, PatchTSTForecaster] = {}
    for h in horizons:
        m = PatchTSTForecaster(h, cfg, device)
        w = m.fit(clean, splits, checkpoint_dir=models_dir, resume=resume)
        m.save(models_dir)
        res = m.evaluate(w, "test")
        plot_sequence_actual_vs_pred(res["_arrays"], outputs_dir / f"patchtst_{h}d_actual_vs_predicted.png", h, "PatchTST")
        res.update(epochs_run=len(m.history), best_epoch=m.best_epoch, best_val_loss=m.best_val_loss,
                   persistence_val_loss=m.persistence_val_loss, best_val_beats_persistence=bool(m.best_val_loss < m.persistence_val_loss),
                   window_accounting=m.window_stats,
                   device=m.device_info, peak_gpu_mb=m.peak_gpu_mb)
        metrics[f"patchtst_{h}d"] = res
        seq_models[f"patchtst_{h}d"] = m
        log.info("PatchTST %dd test: model=%s persistence=%s", h, res["model"], res["persistence"])

    if rolling:
        try:
            folds = ro.make_rolling_folds(clean["date"], **ro.default_rolling_params(clean["date"].nunique()))
            parts_long = [ro.rolling_patchtst(clean, folds, tuple(horizons), cfg, max_epochs=rolling_epochs, device=device)]
            lstm_status = "not requested (--no-lstm-compare)" if not compare_lstm else "not applicable (7-day horizon not selected)"
            if compare_lstm and 7 in horizons:
                try:
                    parts_long.append(ro.rolling_lstm(clean, folds, max_epochs=40))
                    lstm_status = "run"
                except ImportError as exc:
                    lstm_status = f"NOT RUN: {exc}"
            summ = ro.summarize(pd.concat(parts_long, ignore_index=True))
            ro.metrics_table(summ, SOURCE_LABELS[source]).to_csv(outputs_dir / "metrics_table_patchtst.csv", index=False)
            (outputs_dir / "evaluation_patchtst.json").write_text(json.dumps(_clean_json(summ), indent=1))
            metrics["rolling_origin"] = {"folds": [{"fold": f.index, "train_end": str(f.splits.train_end.date()), "origin": str(f.splits.val_end.date()),
                                                    "test_end": str(f.test_end.date())} for f in folds],
                                         "rolling_max_epochs": rolling_epochs, "lstm_comparison": lstm_status, "seasonal_naive_lag_days": 365,
                                         "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_station"} for k, v in summ.items()}}
        except ValueError as exc:
            metrics["rolling_origin"] = {"status": f"NOT RUN: {exc}"}
            log.warning("rolling-origin evaluation not run: %s", exc)

    forecasts = compute_all_forecasts(clean, feats, None, None, None, sequence_models=seq_models)
    for v in forecasts.values():       # this file holds PatchTST only; other models' forecasts stay in their own files
        v.pop("xgb_30d", None), v.pop("xgb_90d", None), v.pop("lstm_7d", None)
    (outputs_dir / "forecasts_patchtst.json").write_text(json.dumps(_clean_json(forecasts), indent=1))
    (outputs_dir / "metrics_patchtst.json").write_text(json.dumps(_clean_json(metrics), indent=2))
    if run_dir is not None:      # compact, self-contained record of this experiment (validation-based; test metrics live in metrics_patchtst.json)
        summary = {"run_dir": str(Path(run_dir).resolve()), "config_file": str(patchtst_config or "config/patchtst_config.json"), "epochs_cap": cfg.max_epochs,
                   "learning_rate": cfg.lr, "patience": cfg.patience, "device": str(device)}
        for h, mdl in ((h_, seq_models[f"patchtst_{h_}d"]) for h_ in horizons):
            summary[f"patchtst_{h}d"] = {"best_epoch": mdl.best_epoch, "best_val_loss": mdl.best_val_loss, "persistence_val_loss": mdl.persistence_val_loss,
                                         "best_val_beats_persistence": bool(mdl.best_val_loss < mdl.persistence_val_loss),
                                         "history": mdl.history, "epochs_run": len(mdl.history)}
        (outputs_dir / "experiment_summary.json").write_text(json.dumps(_clean_json(summary), indent=2))
    log.info("wrote PatchTST results to %s", outputs_dir)
    return metrics
