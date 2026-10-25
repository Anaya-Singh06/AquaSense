"""One reproducible rolling-origin comparison of LSTM, XGBoost, LightGBM, PatchTST and the two baselines on the EXISTING data CSV.

    python scripts/run_rolling_comparison.py --source real --out-dir experiments/rolling_comparison_real_v1
        [--models lstm xgboost lightgbm patchtst] [--horizons 7 30 90] [--patchtst-config FILE] [--lstm-epochs 40]
        [--device auto|cuda|cpu] [--resume] [--limit-folds K]

What it does (nothing here downloads data or touches models/ or outputs/):
  * preprocess -> features -> ONE set of rolling-origin folds (``rolling_origin.make_rolling_folds`` with its existing layout rule);
  * for every (model, horizon, fold) "unit" a FRESH model is trained on that fold's training block only (scalers, early stopping and
    weights never see data after the fold's validation end) and its day-h end-point forecasts for target dates in (origin, test_end]
    are written to ``<out-dir>/units/``;  ``--resume`` skips units that already finished (same data, folds and settings required);
  * all units are intersected on (fold, station, origin, horizon) and scored together with persistence and seasonal-naive on exactly the
    same rows -> ``comparison_*.csv`` (PRIMARY), plus each model on its own eligible rows -> ``evaluation_own_samples.json`` (SECONDARY,
    not comparable between models).
Supported horizons: LSTM 7 only (architecture is fixed to 7 outputs); XGBoost / LightGBM 7, 30, 90 (7 uses the same generic code path and
unchanged hyper-parameters; it was not part of the previous 30/90 runs); PatchTST any (default 7, 30, 90 - the longer horizons need
96 + h consecutive real observations per window and leave far fewer windows). Unsupported combinations are reported, never forced.
Refuses to write into a non-empty --out-dir unless --resume is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, SOURCE_LABELS, Paths, get_logger, set_seed  # noqa: E402
from src.evaluation import rolling_comparison as rc  # noqa: E402
from src.evaluation import rolling_origin as ro  # noqa: E402
from src.preprocessing.features import HORIZONS as TABULAR_HORIZONS  # noqa: E402
from src.preprocessing.features import add_targets, build_features  # noqa: E402
from src.preprocessing.preprocess import load_dataset, preprocess  # noqa: E402

log = get_logger("rolling_comparison")
DEFAULT_PATCHTST_CONFIG = REPO_ROOT / "config" / "patchtst_rolling_lr1e-4.json"
DEFAULT_ITRANSFORMER_CONFIG = REPO_ROOT / "config" / "itransformer_rolling.json"
LSTM_HORIZON = 7
LONG_COLS = ["fold", "model", "horizon", "sensor_id", "origin_date", "target_date", "y_true", "y_model", "lo", "hi", "y_persistence", "y_seasonal_naive"]
NOT_IN_CODEBASE = ["Ridge", "Random Forest", "Extra Trees", "HistGradientBoosting (as a model in its own right)"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict[str, str]:
    import importlib
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "sklearn", "torch", "xgboost", "lightgbm"):
        try:
            v[m] = importlib.import_module(m).__version__
        except ImportError:
            v[m] = "NOT INSTALLED"
    try:
        import torch
        v["torch_cuda"] = str(torch.version.cuda)
        v["cuda_available"] = str(torch.cuda.is_available())
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        pass
    return v


def supported(model: str, h: int) -> tuple[bool, str]:
    if model == "lstm":
        return (h == LSTM_HORIZON, "" if h == LSTM_HORIZON else f"LSTM architecture is fixed to a {LSTM_HORIZON}-day output")
    if model in ("xgboost", "lightgbm"):
        return (h in TABULAR_HORIZONS, "" if h in TABULAR_HORIZONS else f"no target_{h}d column (features.HORIZONS={TABULAR_HORIZONS})")
    return (h >= 1, "")


def run_unit(model: str, h: int, fold: ro.Fold, clean: pd.DataFrame, feats: pd.DataFrame, a: argparse.Namespace, ctx: dict) -> tuple[pd.DataFrame, list[dict]]:
    diag: list[dict] = []
    if model == "xgboost":
        long = ro.rolling_xgb(feats, clean, [fold], (h,), diagnostics=diag)
    elif model == "lightgbm":
        long = ro.rolling_lightgbm(feats, clean, [fold], (h,), config=ctx["lgbm_cfg"], diagnostics=diag)
    elif model == "lstm":
        long = ro.rolling_lstm(clean, [fold], max_epochs=a.lstm_epochs, diagnostics=diag)
    elif model == "patchtst":
        long = ro.rolling_patchtst(clean, [fold], (h,), ctx["pt_cfg"], max_epochs=ctx["pt_rolling_epochs"], device=a.device, diagnostics=diag)
    elif model == "itransformer":
        long = ro.rolling_itransformer(clean, [fold], (h,), ctx["it_cfg"], max_epochs=ctx["it_rolling_epochs"], device=a.device, diagnostics=diag)
    else:
        raise ValueError(model)
    return (long if len(long) else pd.DataFrame(columns=LONG_COLS)), diag


def fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def results_guide(out: Path, manifest: dict, comp: dict, table: pd.DataFrame) -> str:
    L = [f"# Rolling-origin comparison - {manifest['source_label']}", "",
         f"Generated {manifest['finished_utc']} | data sha256 {manifest['data']['sha256'][:12]}... | rows {manifest['data']['rows']} | stations {manifest['data']['stations']} "
         f"| {manifest['data']['date_min']} to {manifest['data']['date_max']}", "",
         "Scoring: day-h end point only; a forecast belongs to a fold iff its TARGET date is in (origin, test_end]; every model, persistence and seasonal-naive "
         "are scored on the SAME rows (intersection of eligible rows). These numbers are NOT comparable with the single-split 'full-horizon' RMSE/MAE "
         "(which average all h days); the matching single-split quantity is the 'day-h only' block.", "",
         "## Folds", "", "| fold | train_end | origin (val_end) | test_end |", "|---|---|---|---|"]
    L += [f"| {f['fold']} | {f['train_end']} | {f['origin']} | {f['test_end']} |" for f in manifest["folds"]]
    for h in sorted({c["horizon_days"] for c in comp["comparisons"].values()}):
        for set_name in ("all_models", "tabular_only"):
            c = comp["comparisons"].get(f"{set_name}_{h}d")
            if not c or not c.get("n"):
                continue
            L += ["", f"## {h}-day horizon - set `{set_name}` ({', '.join(rc.LABELS.get(m, m) for m in c['models'])})", "",
                  f"{c['n']} identical samples, {c['n_stations']} stations, folds {c['folds']}.", "",
                  "| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for name, p in c["predictors"].items():
                v = p.get("vs_persistence")
                wl = f"{v['stations_win']}/{v['stations_loss']}/{v['stations_tie']}" if v else "-"
                fb = f"{v['folds_rmse_better']}/{v['folds_compared']}" if v else "-"
                L.append(f"| {rc.LABELS.get(name, name)} | {fmt(p['rmse'])} | {fmt(p['mae'])} | {fmt(p['nse_pooled'], 6)} | {fmt(p['nse_mean_station'], 3)} | "
                         f"{fmt(p['nse_median_station'], 3)} | {fmt(v['rmse_skill_pooled'], 3) if v else '-'} | {wl} | {fb} |")
    L += ["", "## Not evaluated / limitations", ""]
    L += [f"- {x}" for x in manifest["limitations"]]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real", "real_expanded"], default="real")
    ap.add_argument("--out-dir", default="experiments/rolling_comparison_real", help="NEW output folder (refused if it already has files, unless --resume)")
    ap.add_argument("--models", nargs="+", default=["lstm", "xgboost", "lightgbm", "patchtst"], choices=["lstm", "xgboost", "lightgbm", "patchtst", "itransformer"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 90])
    ap.add_argument("--patchtst-config", default=str(DEFAULT_PATCHTST_CONFIG), help="PatchTST settings JSON (default: lr 1e-4 rolling config)")
    ap.add_argument("--itransformer-config", default=str(DEFAULT_ITRANSFORMER_CONFIG), help="iTransformer settings JSON (default: config/itransformer_rolling.json)")
    ap.add_argument("--lgbm-config", default=None, help="LightGBM settings JSON (default config/lightgbm_config.json)")
    ap.add_argument("--lstm-epochs", type=int, default=40, help="LSTM max epochs per fold (default 40 = the project's existing rolling setting)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--resume", action="store_true", help="re-use finished units in --out-dir (same data / folds / settings required)")
    ap.add_argument("--limit-folds", type=int, default=None, help="SMOKE TEST ONLY: use just the first K folds")
    a = ap.parse_args()

    paths = Paths(a.source)
    if not paths.data_csv.exists():
        sys.exit(f"STOPPED: {paths.data_csv} not found (this script never downloads data).")
    out = Path(a.out_dir)
    out = out if out.is_absolute() else REPO_ROOT / out
    if out.exists() and any(out.iterdir()) and not a.resume:
        sys.exit(f"STOPPED: {out} already contains files. Choose a new --out-dir (nothing is overwritten) or pass --resume.")
    (out / "units").mkdir(parents=True, exist_ok=True)
    set_seed()
    t_start = time.time()

    raw = load_dataset(paths.data_csv)
    clean, prep = preprocess(raw)
    feats = add_targets(build_features(clean))
    folds = ro.make_rolling_folds(clean["date"], **ro.default_rolling_params(clean["date"].nunique()))
    if a.limit_folds:
        folds = folds[:a.limit_folds]
    horizons = sorted(set(a.horizons))
    ctx: dict = {}
    limitations: list[str] = []
    if "lightgbm" in a.models:
        from src.models.lightgbm_forecaster import load_lgbm_config
        ctx["lgbm_cfg"] = load_lgbm_config(a.lgbm_config)
    if "patchtst" in a.models:
        try:
            from src.models.patchtst_forecaster import load_patchtst_config
        except ImportError as exc:       # PyTorch missing: report, do not crash, do not pretend
            a.models = [m for m in a.models if m != "patchtst"]
            limitations.append(f"PatchTST: NOT RUN - PyTorch is not installed ({exc})")
        else:
            loaded = load_patchtst_config(a.patchtst_config)
            ctx["pt_cfg"], ctx["pt_rolling_epochs"] = loaded["config"], loaded["rolling_max_epochs"]
    if "itransformer" in a.models:
        try:
            from src.models.itransformer_forecaster import load_itransformer_config
        except ImportError as exc:       # PyTorch missing: report, do not crash, do not pretend
            a.models = [m for m in a.models if m != "itransformer"]
            limitations.append(f"iTransformer: NOT RUN - PyTorch is not installed ({exc})")
        else:
            loaded = load_itransformer_config(a.itransformer_config)
            ctx["it_cfg"], ctx["it_rolling_epochs"] = loaded["config"], loaded["rolling_max_epochs"]

    fold_rows = [{"fold": f.index, "start": str(f.splits.start.date()), "train_end": str(f.splits.train_end.date()),
                  "origin": str(f.splits.val_end.date()), "test_end": str(f.test_end.date())} for f in folds]
    run_config = {"source": a.source, "models": a.models, "horizons": horizons, "folds": fold_rows, "lstm_epochs": a.lstm_epochs,
                  "patchtst_config_file": a.patchtst_config if "patchtst" in a.models else None,
                  "patchtst": asdict(ctx["pt_cfg"]) if "pt_cfg" in ctx else None, "patchtst_rolling_max_epochs": ctx.get("pt_rolling_epochs"),
                  "itransformer_config_file": a.itransformer_config if "itransformer" in a.models else None,
                  "itransformer": asdict(ctx["it_cfg"]) if "it_cfg" in ctx else None, "itransformer_rolling_max_epochs": ctx.get("it_rolling_epochs"),
                  "lightgbm": ctx.get("lgbm_cfg"), "data_sha256": _sha256(paths.data_csv), "limit_folds": a.limit_folds,
                  "seasonal_naive_lag_days": 365, "min_station_n": rc.MIN_STATION_N}
    cfg_path = out / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        # --resume may ADD models / horizons; everything that changes numbers of finished units must match
        keys = ["source", "folds", "lstm_epochs", "patchtst", "patchtst_rolling_max_epochs", "itransformer",
                "itransformer_rolling_max_epochs", "lightgbm", "data_sha256", "limit_folds"]
        bad = [k for k in keys if old.get(k) != run_config.get(k) and not (old.get(k) is None or run_config.get(k) is None)]
        if bad:
            sys.exit(f"STOPPED: --resume but settings differ from the earlier run in: {bad}. Use a new --out-dir.")
    cfg_path.write_text(json.dumps(run_config, indent=2, default=str))

    unit_status: list[dict] = []
    frames: list[pd.DataFrame] = []
    diag_all: list[dict] = []
    for model in a.models:
        for h in horizons:
            ok, why = supported(model, h)
            if not ok:
                limitations.append(f"{rc.LABELS[model]} at {h} d: NOT EVALUATED - {why}.")
                unit_status.append({"model": model, "horizon": h, "fold": None, "status": f"unsupported: {why}"})
                continue
            for f in folds:
                stem = out / "units" / f"{model}_{h}d_fold{f.index}"
                csv_p, js_p = stem.with_suffix(".csv"), stem.with_suffix(".json")
                if a.resume and csv_p.exists() and js_p.exists() and json.loads(js_p.read_text()).get("status") == "ok":
                    frames.append(pd.read_csv(csv_p, parse_dates=["origin_date", "target_date"], dtype={"sensor_id": str}))
                    diag_all += json.loads(js_p.read_text())["diagnostics"]
                    unit_status.append({"model": model, "horizon": h, "fold": f.index, "status": "reused"})
                    log.info("[reuse] %s %dd fold %d", model, h, f.index)
                    continue
                t0 = time.time()
                log.info("[run] %s %dd fold %d  (train_end %s, origin %s, test_end %s)", model, h, f.index,
                         f.splits.train_end.date(), f.splits.val_end.date(), f.test_end.date())
                try:
                    long, diag = run_unit(model, h, f, clean, feats, a, ctx)
                except ImportError as exc:
                    msg = f"{rc.LABELS[model]}: NOT RUN - required package missing ({exc})"
                    if msg not in limitations:
                        limitations.append(msg)
                    unit_status.append({"model": model, "horizon": h, "fold": f.index, "status": f"NOT RUN: {exc}"})
                    break
                dt = round(time.time() - t0, 1)
                status = "ok" if diag and all(d.get("status") == "ok" for d in diag) and len(long) else "skipped/empty"
                long.to_csv(csv_p, index=False, date_format="%Y-%m-%d")
                js_p.write_text(json.dumps({"status": status, "seconds": dt, "diagnostics": diag, "rows": int(len(long))}, indent=1, default=str))
                diag_all += diag
                unit_status.append({"model": model, "horizon": h, "fold": f.index, "status": status, "rows": int(len(long)), "seconds": dt})
                if len(long):
                    frames.append(long)
                if status != "ok":
                    limitations.append(f"{rc.LABELS[model]} {h} d fold {f.index}: {status} ({[d.get('status') for d in diag]}).")
                log.info("[done] %s %dd fold %d: %d rows in %.1fs", model, h, f.index, len(long), dt)

    if not frames:
        sys.exit("STOPPED: no model produced any forecast rows; nothing to compare.")
    long = pd.concat(frames, ignore_index=True)[LONG_COLS]
    long["sensor_id"] = long["sensor_id"].astype(str)
    long["origin_date"], long["target_date"] = pd.to_datetime(long["origin_date"]), pd.to_datetime(long["target_date"])
    long.to_csv(out / "forecasts_long.csv.gz", index=False, date_format="%Y-%m-%d")

    comp = rc.build_comparison(long)
    tab = rc.comparison_table(comp)
    tab.to_csv(out / "comparison_common_samples.csv", index=False)
    rc.by_fold_table(comp).to_csv(out / "comparison_by_fold.csv", index=False)
    rc.station_table_all(comp).to_csv(out / "comparison_station_level.csv", index=False)
    rc.accounting_table(comp).to_csv(out / "comparison_sample_accounting.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_definitions.csv", index=False)
    pd.DataFrame(diag_all).to_csv(out / "training_diagnostics.csv", index=False)
    pd.DataFrame(unit_status).to_csv(out / "unit_status.csv", index=False)
    from src.pipeline import _clean_json  # noqa: PLC0415  (JSON-safe NaN handling; no torch import)
    nested = {k: {kk: vv for kk, vv in v.items() if kk != "station_outcomes"} for k, v in comp["comparisons"].items()}
    (out / "comparison_common_samples.json").write_text(json.dumps(_clean_json({"comparisons": nested, "accounting": comp["accounting"]}), indent=1))
    own = ro.summarize(long)
    (out / "evaluation_own_samples.json").write_text(json.dumps(_clean_json(own), indent=1))
    ro.metrics_table(own, SOURCE_LABELS[a.source]).to_csv(out / "metrics_table_own_samples.csv", index=False)

    # ---- completeness / validity notes -------------------------------------------------------------------------------------
    for model in a.models:
        for h in horizons:
            if supported(model, h)[0]:
                got = sorted(long.loc[(long["model"] == model) & (long["horizon"] == h), "fold"].unique())
                if got != [f.index for f in folds]:
                    limitations.append(f"{rc.LABELS[model]} {h} d produced rows for folds {got} only (of {[f.index for f in folds]}): its common-sample comparison "
                                       "is restricted to those folds.")
    backends = sorted({d.get("backend") for d in diag_all if d.get("backend")})
    if any(b and b != "xgboost" and b != "lightgbm" for b in backends):
        limitations.append(f"XGBoost rows come from a stand-in backend {backends} - NOT real XGBoost.")
    limitations += [f"{x}: not found anywhere in the project source, so not evaluated (nothing was invented)." for x in NOT_IN_CODEBASE]
    limitations += ["Single seed (42) per model and fold: no multi-seed variability is reported.",
                    f"The {len(folds)} rolling-origin test blocks cover target dates {folds[0].splits.val_end.date() + pd.Timedelta(days=1)} to {folds[-1].test_end.date()} only; "
                    "folds share most of their training data, so they are not independent replications.",
                    "Pooled NSE is close to 1 for every forecaster because between-station differences dominate; use mean per-station NSE, skill and station wins/losses."]
    if a.limit_folds:
        limitations.insert(0, f"SMOKE TEST: only the first {a.limit_folds} fold(s) were used - not a final result.")

    manifest = {"source": a.source, "source_label": SOURCE_LABELS[a.source], "out_dir": str(out), "models_requested": a.models, "horizons_requested": horizons,
                "folds": fold_rows, "data": {"path": str(paths.data_csv), "sha256": run_config["data_sha256"], "rows": int(len(clean)),
                                             "stations": int(clean["sensor_id"].nunique()), "date_min": str(clean["date"].min().date()),
                                             "date_max": str(clean["date"].max().date())},
                "preprocessing": prep, "versions": _versions(), "unit_status": unit_status, "limitations": limitations,
                "finished_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"), "wall_seconds": round(time.time() - t_start, 1)}
    (out / "run_manifest.json").write_text(json.dumps(_clean_json(manifest), indent=2, default=str))
    (out / "RESULTS_GUIDE.md").write_text(results_guide(out, manifest, comp, tab), encoding="utf-8")

    print("\n=== ROLLING-ORIGIN COMPARISON (primary: identical samples) ===")
    if len(tab):
        cols = ["set", "horizon_days", "predictor", "n_samples", "n_stations", "RMSE_m", "MAE_m", "NSE_pooled", "NSE_mean_station", "stations_win", "stations_loss"]
        with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print(tab[cols].to_string(index=False))
    print(f"\nWritten to {out}\nRead RESULTS_GUIDE.md first. Limitations/NOT-evaluated list is in run_manifest.json.")


if __name__ == "__main__":
    main()
