"""EXPANDED-station pipeline: download + filter a larger USGS/NASA station set, then run EVERY model on it.

    python scripts/run_expanded_pipeline.py [--config config/real_data_expanded_config.json]
        [--out-dir experiments/rolling_comparison_real_expanded_v1] [--summary-path evaluation_reference/rolling_comparison_summary_expanded.json]
        [--models lstm xgboost lightgbm patchtst itransformer] [--horizons 7 30 90] [--lstm-epochs 40]
        [--device auto|cuda|cpu] [--refresh] [--skip-fetch] [--no-main-pipeline] [--resume] [--limit-folds K]

What it does (the original 60-station 'real' dataset is never touched - every file below has an `_expanded` / `real_expanded`
name, i.e. a SEPARATE data source):
  1. DOWNLOAD + FILTER: discovery (time-series-metadata -> groundwater sites with long, currently-active records) -> cheap
     metadata filter -> per-station data-sufficiency verdict (thresholds fixed in the config BEFORE training) -> NASA POWER
     climate, written to data/real_data_expanded.csv + data/real_stations_expanded.csv + outputs/real_expanded/station_selection.json.
     All raw responses are cached under data/raw/, so re-runs only fetch what is new (``--refresh`` re-downloads everything).
  2. MAIN PIPELINE for source 'real_expanded': preprocess -> features -> LSTM + XGBoost -> anomalies -> spatial grid ->
     forecasts, so the dashboard service can serve the expanded source (its Map tab).
  3. ROLLING-ORIGIN COMPARISON for every requested model (LSTM 7 d; XGBoost / LightGBM / PatchTST / iTransformer 7/30/90),
     run by ``scripts/run_rolling_comparison.py`` into ``--out-dir`` (reference it with ``--resume`` to continue a partial run).
  4. DASHBOARD Pooled-tab summary (``--summary-path``) rebuilt from the intersected comparison, so all models show up in the
     dashboard exactly as they do for the original source.

Point the dashboard at the expanded results with AquaSense_expanded.bat (sets AQUASENSE_SOURCE / AQUASENSE_EVAL_SUMMARY /
AQUASENSE_EVAL_STATIONS).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.fetch_real_data import FetchError, RawCache, build_real_dataset, load_config  # noqa: E402
from src.config import REPO_ROOT, SOURCE_LABELS, Paths, get_logger, set_seed  # noqa: E402
from src.evaluation import rolling_comparison as rc  # noqa: E402

log = get_logger("expanded_pipeline")
SOURCE = "real_expanded"
DEFAULT_CONFIG = REPO_ROOT / "config" / "real_data_expanded_config.json"
DEFAULT_OUT = REPO_ROOT / "experiments" / "rolling_comparison_real_expanded_v1"
DEFAULT_SUMMARY = REPO_ROOT / "evaluation_reference" / "rolling_comparison_summary_expanded.json"
DEFAULT_MODELS = ["lstm", "xgboost", "lightgbm", "patchtst", "itransformer"]


def _backup(path: Path) -> None:
    """Keep a one-off copy of the dashboard summary this run rewrites (never delete the original)."""
    if path.exists():
        bak = path.with_name(path.name + ".pre_expanded.bak")
        if not bak.exists():
            shutil.copy2(path, bak)


def _step1_download(paths: Paths, cfg_path: str, refresh: bool, max_stations: int | None) -> None:
    """Build data/real_data_expanded.csv (+ stations, selection report) with the expanded discovery caps."""
    cfg = load_config(Path(cfg_path))
    log.info("[1/4] downloading + filtering stations (bbox=%s, max_candidates=%d, target_stations=%d)",
             cfg["discovery"]["bbox"], cfg["discovery"]["max_candidates"], cfg["discovery"]["target_stations"])
    try:
        df, stations, report = build_real_dataset(cfg, RawCache(refresh=refresh, offline=False), max_stations)
    except FetchError as exc:
        raise RuntimeError(f"expanded USGS/NASA download failed: {exc}\n"
                          "Check the connection, `python scripts/check_usgs_api.py`, and the USGS block in the config.") from exc
    df.to_csv(paths.data_csv, index=False, date_format="%Y-%m-%d")
    stations.to_csv(paths.stations_csv, index=False)
    (paths.outputs_dir / "station_selection.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("[1/4] stations selected=%d (rejected=%d of %d discovered) -> %s",
             report["stations_selected"], report["stations_rejected"], report["stations_discovered"], paths.data_csv)


def _step3_compare(out: Path, a: argparse.Namespace) -> None:
    """Run the existing rolling-origin comparison runner for the expanded source (all requested models)."""
    runner = REPO_ROOT / "scripts" / "run_rolling_comparison.py"
    reuse = a.resume or (out / "run_config.json").exists()
    cmd = [sys.executable, str(runner), "--source", SOURCE, "--out-dir", str(out),
           "--models", *a.models, "--horizons", *[str(h) for h in a.horizons], "--device", a.device]
    if reuse:
        cmd.append("--resume")
        if not a.resume:
            log.info("[3/4] %s already holds an earlier run - re-using its finished units (--resume) instead of refusing.", out)
    if a.limit_folds:
        cmd += ["--limit-folds", str(a.limit_folds)]
    log.info("[3/4] rolling-origin comparison: %s", " ".join(cmd[1:]))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    products = [out / "forecasts_long.csv.gz", out / "run_manifest.json", out / "comparison_common_samples.csv"]
    missing = [p.name for p in products if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"rolling-origin comparison (exit code {proc.returncode}) did not write {missing} in {out}.")
    if proc.returncode != 0:
        # On some Windows setups a native library (torch / XGBoost / OpenMP) crashes while Python shuts down, AFTER the
        # runner has written every output. The artifacts are complete, so this is not a failed comparison.
        log.warning("[3/4] the comparison process exited with code %s AFTER writing every output - a native library crash on "
                    "interpreter shutdown, not a failed run; continuing.", proc.returncode)


def _step4_summary(out: Path, summary_path: Path) -> dict:
    """Rebuild the dashboard Pooled-tab JSON from the comparison the runner just wrote."""
    manifest = json.loads((out / "run_manifest.json").read_text())
    long = pd.read_csv(out / "forecasts_long.csv.gz", dtype={"sensor_id": str}, parse_dates=["origin_date", "target_date"])
    comp = rc.build_comparison(long)
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    _backup(summary_path)
    summary_path.write_text(json.dumps(rc.dashboard_summary(comp, manifest, existing), indent=2), encoding="utf-8")
    log.info("[4/4] dashboard summary refreshed: %s", summary_path)
    return comp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="expanded discovery/sufficiency config (default: config/real_data_expanded_config.json)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="rolling-comparison folder (an earlier run in it is re-used automatically)")
    ap.add_argument("--summary-path", default=str(DEFAULT_SUMMARY), help="dashboard Pooled-tab JSON to write")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    ap.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 90])
    ap.add_argument("--lstm-epochs", type=int, default=None, help="max LSTM epochs for the main pipeline (default 60)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--max-stations", type=int, default=None, help="cap the number of stations (default: the config's target)")
    ap.add_argument("--refresh", action="store_true", help="ignore data/raw/ and re-download everything while building the dataset")
    ap.add_argument("--skip-fetch", action="store_true", help="reuse an existing data/real_data_expanded.csv (never download)")
    ap.add_argument("--no-main-pipeline", action="store_true", help="skip step 2 (the dashboard service artifacts for the expanded source)")
    ap.add_argument("--resume", action="store_true", help="always re-use finished units in --out-dir (automatic when it already has a run_config.json)")
    ap.add_argument("--limit-folds", type=int, default=None, help="SMOKE TEST ONLY: use just the first K folds")
    a = ap.parse_args()

    paths = Paths(SOURCE)
    paths.ensure_dirs()
    out = Path(a.out_dir)
    out = out if out.is_absolute() else REPO_ROOT / out
    summary_path = Path(a.summary_path)
    summary_path = summary_path if summary_path.is_absolute() else REPO_ROOT / summary_path
    set_seed()
    t_start = time.time()

    log.info("=== EXPANDED station-set pipeline (%s) ===", SOURCE_LABELS[SOURCE])
    try:
        if a.skip_fetch:
            if not paths.data_csv.exists():
                sys.exit(f"STOPPED: --skip-fetch but {paths.data_csv} does not exist.")
            log.info("[1/4] --skip-fetch: reusing %s", paths.data_csv)
        else:
            _step1_download(paths, a.config, a.refresh, a.max_stations)

        if a.no_main_pipeline:
            log.info("[2/4] --no-main-pipeline: skipping preprocess/features/LSTM/XGBoost/anomalies/spatial grid")
        else:
            from src.pipeline import run  # noqa: PLC0415  (imports torch only inside)
            log.info("[2/4] main pipeline for source '%s' (preprocess -> features -> LSTM/XGBoost -> anomalies -> spatial grid)", SOURCE)
            run(SOURCE, regenerate=False, lstm_epochs=a.lstm_epochs, fetch=False, rolling=False)

        _step3_compare(out, a)
        comp = _step4_summary(out, summary_path)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nEXPANDED PIPELINE STOPPED: {exc}", file=sys.stderr)
        return 1

    tab = rc.comparison_table(comp)
    print("\n=== EXPANDED STATION SET - ROLLING-ORIGIN COMPARISON (primary: identical samples) ===")
    if len(tab):
        cols = ["set", "horizon_days", "predictor", "n_samples", "n_stations", "RMSE_m", "MAE_m", "NSE_mean_station",
                "stations_win", "stations_loss"]
        with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print(tab[tab["set"] == "all_models"][cols].to_string(index=False))
    print(f"\nData:            {paths.data_csv}")
    print(f"Comparison:      {out}")
    print(f"Dashboard JSON:  {summary_path}")
    print(f"Wall time:       {round(time.time() - t_start, 1)} s")
    print("\nNext: run AquaSense_expanded.bat to open the dashboard on the expanded source "
          "(Map & operations + Pooled evaluation tabs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
