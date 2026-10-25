"""Add the iTransformer to a COMPLETED rolling-origin comparison WITHOUT retraining any other model.

    python scripts/run_itransformer_rolling.py [--source real]
        [--out-dir experiments/rolling_comparison_real_v1] [--horizons 7 30 90]
        [--itransformer-config config/itransformer_rolling.json] [--device auto|cuda|cpu]
        [--resume] [--limit-folds K] [--no-summary] [--summary-path FILE]

Why this script exists (kept deliberately SEPARATE from scripts/run_rolling_comparison.py):
  The LSTM / XGBoost / LightGBM / PatchTST units already trained in ``--out-dir`` are RE-USED verbatim (they are read from
  ``forecasts_long.csv.gz``) - nothing is retrained. Only the iTransformer is trained, once per (horizon, fold), with the SAME
  fold layout that the earlier run recorded, and its forecasts are appended to the stored rows. The intersected comparison
  (every model, persistence and seasonal-naive scored on identical samples) is then rebuilt so the iTransformer appears
  alongside the existing models. Copies of the files this script rewrites are kept as ``*.pre_itransformer.bak``.

Writes into ``--out-dir`` (the file names the existing dashboard already reads):
  forecasts_long.csv.gz, comparison_common_samples.{csv,json}, comparison_by_fold.csv, comparison_station_level.csv,
  comparison_sample_accounting.csv, evaluation_own_samples.json, metrics_table_own_samples.csv, run_manifest.json,
  RESULTS_GUIDE.md, units/itransformer_{h}d_fold{k}.{csv,json}, run_config_itransformer.json
and refreshes the dashboard's Pooled-tab source ``evaluation_reference/rolling_comparison_summary.json`` (numbers only;
the descriptive text already in the file is preserved).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
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
from src.preprocessing.features import add_targets, build_features  # noqa: E402
from src.preprocessing.preprocess import load_dataset, preprocess  # noqa: E402

log = get_logger("itransformer_rolling")
MODEL = "itransformer"
DEFAULT_OUT = REPO_ROOT / "experiments" / "rolling_comparison_real_v1"
DEFAULT_ITRANS_CONFIG = REPO_ROOT / "config" / "itransformer_rolling.json"
DEFAULT_SUMMARY = REPO_ROOT / "evaluation_reference" / "rolling_comparison_summary.json"
LONG_COLS = ["fold", "model", "horizon", "sensor_id", "origin_date", "target_date", "y_true", "y_model", "lo", "hi",
             "y_persistence", "y_seasonal_naive"]


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


def _backup(path: Path) -> None:
    """Keep a one-off copy of a file this run rewrites (never delete the original)."""
    if path.exists():
        bak = path.with_name(path.name + ".pre_itransformer.bak")
        if not bak.exists():
            shutil.copy2(path, bak)


def recorded_folds(out: Path) -> list[dict] | None:
    """The fold layout the earlier run used (fold_definitions.csv), if present."""
    p = out / "fold_definitions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return [{"fold": int(r["fold"]), "start": str(r["start"])[:10], "train_end": str(r["train_end"])[:10],
             "origin": str(r["origin"])[:10], "test_end": str(r["test_end"])[:10]} for _, r in d.iterrows()]


def build_dashboard_summary(comp: dict, manifest: dict, existing: dict) -> dict:
    """Pooled-tab JSON (GET /api/evaluation) rebuilt from the intersected comparison, all-models set.
    Only the numeric ``data`` / ``metrics`` blocks are replaced; title / evaluation / integration_notes are preserved.
    The single implementation lives in ``src.evaluation.rolling_comparison.dashboard_summary``."""
    return rc.dashboard_summary(comp, manifest, existing)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real"], default="real")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="the COMPLETED rolling-comparison folder to extend in place")
    ap.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 90])
    ap.add_argument("--itransformer-config", default=str(DEFAULT_ITRANS_CONFIG))
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--resume", action="store_true", help="re-use iTransformer units that already finished (other models always re-used)")
    ap.add_argument("--limit-folds", type=int, default=None, help="SMOKE TEST ONLY: use just the first K folds")
    ap.add_argument("--summary-path", default=str(DEFAULT_SUMMARY), help="dashboard Pooled-tab JSON to refresh")
    ap.add_argument("--no-summary", action="store_true", help="do not touch the dashboard summary JSON")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out = out if out.is_absolute() else REPO_ROOT / out
    long_gz = out / "forecasts_long.csv.gz"
    if not long_gz.exists():
        sys.exit(f"STOPPED: {long_gz} not found. Run scripts/run_rolling_comparison.py first (this script extends a finished run).")
    try:
        from src.models.itransformer_forecaster import load_itransformer_config  # noqa: PLC0415
    except ImportError as exc:
        sys.exit(f"STOPPED: PyTorch is required for the iTransformer ({exc}).")

    paths = Paths(a.source)
    if not paths.data_csv.exists():
        sys.exit(f"STOPPED: {paths.data_csv} not found (this script never downloads data).")
    loaded = load_itransformer_config(a.itransformer_config)
    it_cfg, it_rolling_epochs = loaded["config"], loaded["rolling_max_epochs"]

    set_seed()
    t_start = time.time()
    raw = load_dataset(paths.data_csv)
    clean, prep = preprocess(raw)
    _ = add_targets(build_features(clean))  # folds only depend on ``clean``; built for parity with the main runner
    folds = ro.make_rolling_folds(clean["date"], **ro.default_rolling_params(clean["date"].nunique()))
    if a.limit_folds:
        folds = folds[:a.limit_folds]
    horizons = sorted(set(a.horizons))

    new_fold_rows = [{"fold": f.index, "start": str(f.splits.start.date()), "train_end": str(f.splits.train_end.date()),
                      "origin": str(f.splits.val_end.date()), "test_end": str(f.test_end.date())} for f in folds]
    rec = recorded_folds(out)
    if rec is not None:
        by_index = {r["fold"]: {k: r[k] for k in ("start", "train_end", "origin", "test_end")} for r in rec}
        for row in new_fold_rows:
            mine = {k: row[k] for k in ("start", "train_end", "origin", "test_end")}
            if by_index.get(row["fold"]) != mine:
                sys.exit(f"STOPPED: recomputed fold {row['fold']} ({mine}) does not match fold_definitions.csv in --out-dir "
                         f"({by_index.get(row['fold'])}). The iTransformer must be scored on the SAME folds as the existing models; "
                         "check that the source data CSV is unchanged.")

    # ---- existing rows (all previously trained models) are RE-USED, never retrained -------------------------------
    old = pd.read_csv(long_gz, dtype={"sensor_id": str}, parse_dates=["origin_date", "target_date"])
    old["model"] = old["model"].astype(str)
    kept = old[old["model"] != MODEL].copy()
    if kept.empty:
        sys.exit(f"STOPPED: {long_gz} has no rows from other models to compare against.")
    log.info("re-using %d stored forecast rows from %d model(s): %s", len(kept), kept["model"].nunique(),
             ", ".join(sorted(kept["model"].unique())))

    (out / "units").mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    diag_all: list[dict] = []
    unit_status: list[dict] = []
    limitations: list[str] = []
    for h in horizons:
        for f in folds:
            stem = out / "units" / f"{MODEL}_{h}d_fold{f.index}"
            csv_p, js_p = stem.with_suffix(".csv"), stem.with_suffix(".json")
            if a.resume and csv_p.exists() and js_p.exists() and json.loads(js_p.read_text()).get("status") == "ok":
                frames.append(pd.read_csv(csv_p, dtype={"sensor_id": str}, parse_dates=["origin_date", "target_date"]))
                diag_all += json.loads(js_p.read_text())["diagnostics"]
                unit_status.append({"model": MODEL, "horizon": h, "fold": f.index, "status": "reused"})
                log.info("[reuse] %s %dd fold %d", MODEL, h, f.index)
                continue
            log.info("[run] %s %dd fold %d  (train_end %s, origin %s, test_end %s)", MODEL, h, f.index,
                     f.splits.train_end.date(), f.splits.val_end.date(), f.test_end.date())
            t0 = time.time()
            diag: list[dict] = []
            try:
                long = ro.rolling_itransformer(clean, [f], (h,), it_cfg, max_epochs=it_rolling_epochs, device=a.device, diagnostics=diag)
            except (ImportError, RuntimeError) as exc:
                limitations.append(f"iTransformer {h} d fold {f.index}: NOT RUN - {exc}")
                unit_status.append({"model": MODEL, "horizon": h, "fold": f.index, "status": f"NOT RUN: {exc}"})
                log.warning("iTransformer %dd fold %d NOT RUN: %s", h, f.index, exc)
                continue
            dt = round(time.time() - t0, 1)
            status = "ok" if diag and all(d.get("status") == "ok" for d in diag) and len(long) else "skipped/empty"
            long = long if len(long) else pd.DataFrame(columns=LONG_COLS)
            long.to_csv(csv_p, index=False, date_format="%Y-%m-%d")
            js_p.write_text(json.dumps({"status": status, "seconds": dt, "diagnostics": diag, "rows": int(len(long))}, indent=1, default=str))
            diag_all += diag
            unit_status.append({"model": MODEL, "horizon": h, "fold": f.index, "status": status, "rows": int(len(long)), "seconds": dt})
            if len(long):
                frames.append(long)
            else:
                limitations.append(f"iTransformer {h} d fold {f.index}: {status} ({[d.get('status') for d in diag]}).")
            log.info("[done] %s %dd fold %d: %d rows in %.1fs", MODEL, h, f.index, len(long), dt)

    if not frames:
        sys.exit("STOPPED: the iTransformer produced no forecast rows; nothing was changed.")

    merged = pd.concat([kept, *frames], ignore_index=True)
    merged = merged[LONG_COLS]
    merged["sensor_id"] = merged["sensor_id"].astype(str)
    merged["origin_date"], merged["target_date"] = pd.to_datetime(merged["origin_date"]), pd.to_datetime(merged["target_date"])
    for name in ("forecasts_long.csv.gz", "comparison_common_samples.csv", "comparison_by_fold.csv", "comparison_station_level.csv",
                 "comparison_sample_accounting.csv", "comparison_common_samples.json", "evaluation_own_samples.json",
                 "metrics_table_own_samples.csv", "run_manifest.json", "RESULTS_GUIDE.md"):
        _backup(out / name)
    merged.to_csv(long_gz, index=False, date_format="%Y-%m-%d")

    # ---- rebuild the intersected comparison (identical samples across every model) -------------------------------
    comp = rc.build_comparison(merged)
    tab = rc.comparison_table(comp)
    tab.to_csv(out / "comparison_common_samples.csv", index=False)
    rc.by_fold_table(comp).to_csv(out / "comparison_by_fold.csv", index=False)
    rc.station_table_all(comp).to_csv(out / "comparison_station_level.csv", index=False)
    rc.accounting_table(comp).to_csv(out / "comparison_sample_accounting.csv", index=False)
    from src.pipeline import _clean_json  # noqa: PLC0415 (JSON-safe NaN handling; no torch import)
    nested = {k: {kk: vv for kk, vv in v.items() if kk != "station_outcomes"} for k, v in comp["comparisons"].items()}
    (out / "comparison_common_samples.json").write_text(json.dumps(_clean_json({"comparisons": nested, "accounting": comp["accounting"]}), indent=1))
    own = ro.summarize(merged)
    (out / "evaluation_own_samples.json").write_text(json.dumps(_clean_json(own), indent=1))
    ro.metrics_table(own, SOURCE_LABELS[a.source]).to_csv(out / "metrics_table_own_samples.csv", index=False)

    # ---- notes ----------------------------------------------------------------------------------------------------
    for h in horizons:
        got = sorted(merged.loc[(merged["model"] == MODEL) & (merged["horizon"] == h), "fold"].unique())
        if got != [f.index for f in folds]:
            limitations.append(f"iTransformer {h} d produced rows for folds {got} only (of {[f.index for f in folds]}): its common-sample "
                               "comparison is restricted to those folds.")
    limitations += ["Single seed (42) per model and fold: no multi-seed variability is reported.",
                    f"The {len(folds)} rolling-origin test blocks cover target dates {folds[0].splits.val_end.date() + pd.Timedelta(days=1)} to "
                    f"{folds[-1].test_end.date()} only; folds share most of their training data, so they are not independent replications.",
                    "Pooled NSE is close to 1 for every forecaster because between-station differences dominate; use mean per-station NSE, skill and "
                    "station wins/losses.",
                    "The LSTM / XGBoost / LightGBM / PatchTST rows were RE-USED from the earlier run (not retrained); only the iTransformer is new."]
    if a.limit_folds:
        limitations.insert(0, f"SMOKE TEST: only the first {a.limit_folds} fold(s) were used - not a final result.")

    data_info = {"path": str(paths.data_csv), "sha256": _sha256(paths.data_csv), "rows": int(len(clean)),
                 "stations": int(clean["sensor_id"].nunique()), "date_min": str(clean["date"].min().date()),
                 "date_max": str(clean["date"].max().date())}
    manifest = {"source": a.source, "source_label": SOURCE_LABELS[a.source], "out_dir": str(out), "script": "scripts/run_itransformer_rolling.py",
                "models_reused": sorted(kept["model"].unique().tolist()), "model_added": MODEL, "horizons_requested": horizons,
                "folds": new_fold_rows, "data": data_info, "preprocessing": prep, "versions": _versions(),
                "itransformer_config_file": a.itransformer_config, "itransformer": asdict(it_cfg),
                "itransformer_rolling_max_epochs": it_rolling_epochs, "unit_status": unit_status, "limitations": limitations,
                "finished_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"), "wall_seconds": round(time.time() - t_start, 1)}
    (out / "run_manifest.json").write_text(json.dumps(_clean_json(manifest), indent=2, default=str))
    (out / "run_config_itransformer.json").write_text(json.dumps(
        {"source": a.source, "model": MODEL, "horizons": horizons, "itransformer_config_file": a.itransformer_config,
         "itransformer": asdict(it_cfg), "itransformer_rolling_max_epochs": it_rolling_epochs, "data_sha256": data_info["sha256"],
         "folds": new_fold_rows, "limit_folds": a.limit_folds}, indent=2, default=str))
    (out / "RESULTS_GUIDE.md").write_text(results_guide(manifest, comp), encoding="utf-8")

    # ---- dashboard Pooled-tab summary (numbers only; wording preserved) ------------------------------------------
    if not a.no_summary:
        sp = Path(a.summary_path)
        sp = sp if sp.is_absolute() else REPO_ROOT / sp
        existing = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        _backup(sp)
        sp.write_text(json.dumps(build_dashboard_summary(comp, manifest, existing), indent=2), encoding="utf-8")
        log.info("refreshed dashboard summary: %s", sp)

    print("\n=== iTRANSFORMER ADDED TO THE ROLLING-ORIGIN COMPARISON (identical samples) ===")
    if len(tab):
        cols = ["set", "horizon_days", "predictor", "n_samples", "n_stations", "RMSE_m", "MAE_m", "NSE_pooled", "NSE_mean_station",
                "stations_win", "stations_loss"]
        with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print(tab[tab["set"] == "all_models"][cols].to_string(index=False))
    print(f"\nWritten to {out}\nOpen the dashboard (AquaSense.bat) and check the 'Map & operations' and 'Pooled evaluation' tabs.")


def results_guide(manifest: dict, comp: dict) -> str:
    L = ["# iTransformer added to the rolling-origin comparison", "",
         f"Generated {manifest['finished_utc']} | data sha256 {manifest['data']['sha256'][:12]}... | rows {manifest['data']['rows']} | "
         f"stations {manifest['data']['stations']} | {manifest['data']['date_min']} to {manifest['data']['date_max']}", "",
         f"Re-used models: {', '.join(rc.LABELS.get(m, m) for m in manifest['models_reused'])} (NOT retrained). Added: iTransformer.", "",
         "Scoring: day-h end point only; a forecast belongs to a fold iff its TARGET date is in (origin, test_end]; every model, persistence and "
         "seasonal-naive are scored on the SAME (fold, station, origin) rows (all-models intersection).", "",
         "## Folds", "", "| fold | train_end | origin (val_end) | test_end |", "|---|---|---|---|"]
    L += [f"| {f['fold']} | {f['train_end']} | {f['origin']} | {f['test_end']} |" for f in manifest["folds"]]
    for h in sorted({c["horizon_days"] for c in comp["comparisons"].values()}):
        c = comp["comparisons"].get(f"all_models_{h}d")
        if not c or not c.get("n"):
            continue
        L += ["", f"## {h}-day horizon - all models ({', '.join(rc.LABELS.get(m, m) for m in c['models'])})", "",
              f"{c['n']} identical samples, {c['n_stations']} stations, folds {c['folds']}.", "",
              "| predictor | RMSE m | MAE m | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |",
              "|---|---|---|---|---|---|---|---|"]
        for name, p in c["predictors"].items():
            v = p.get("vs_persistence")
            wl = f"{v['stations_win']}/{v['stations_loss']}/{v['stations_tie']}" if v else "-"
            fb = f"{v['folds_rmse_better']}/{v['folds_compared']}" if v else "-"
            skill = f"{v['rmse_skill_pooled']:.3f}" if v else "-"
            L.append(f"| {rc.LABELS.get(name, name)} | {p['rmse']:.4f} | {p['mae']:.4f} | {p['nse_mean_station']:.3f} | "
                     f"{p['nse_median_station']:.3f} | {skill} | {wl} | {fb} |")
    L += ["", "## Notes", ""] + [f"- {x}" for x in manifest["limitations"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
