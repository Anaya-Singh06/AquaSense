"""Train + evaluate the additional PatchTST forecaster on the EXISTING data CSV.

    python scripts/train_patchtst.py [--source synthetic|real] [--horizons 7 [30 90]] [--no-rolling] [--no-lstm-compare]
                                     [--patchtst-config FILE] [--epochs N] [--device auto|cuda|cpu] [--resume] [--run-dir DIR]

Never regenerates/downloads data and never overwrites existing models/results. Writes (in the source's models/ and outputs/ dirs):
  models:  patchtst_{h}d.pt, patchtst_{h}d_scalers.joblib, patchtst_{h}d_meta.json   (patchtst_{h}d_resume.pt only while training)
  outputs: metrics_patchtst.json, metrics_table_patchtst.csv, evaluation_patchtst.json, forecasts_patchtst.json,
           patchtst_{h}d_actual_vs_predicted.png
Settings: config/patchtst_config.json (horizons default to the file's list, 7 only).
--run-dir DIR keeps the whole run isolated: models in DIR/models, results in DIR/outputs (default folders untouched; refuses a non-empty DIR).
--resume continues an interrupted run from patchtst_{h}d_resume.pt (same settings and data required).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_patchtst  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--horizons", type=int, nargs="+", default=None, help="forecast horizons in days (default: config file, 7)")
    ap.add_argument("--no-rolling", action="store_true", help="skip the rolling-origin evaluation (single chronological split only)")
    ap.add_argument("--no-lstm-compare", action="store_true", help="rolling-origin: do not re-run the LSTM alongside PatchTST (7-day horizon)")
    ap.add_argument("--patchtst-config", default=None, help="alternative settings JSON (default config/patchtst_config.json)")
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs (e.g. 2 for a quick check; rolling uses min(this, rolling cap))")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="auto = CUDA if available, else CPU")
    ap.add_argument("--resume", action="store_true", help="resume the single-split training from its resume checkpoint if present")
    ap.add_argument("--run-dir", default=None, help="isolated experiment folder: writes DIR/models and DIR/outputs instead of models/ and outputs/ (never overwrites)")
    a = ap.parse_args()
    try:
        m = run_patchtst(a.source, rolling=not a.no_rolling, compare_lstm=not a.no_lstm_compare, horizons=a.horizons,
                         patchtst_config=a.patchtst_config, resume=a.resume, epochs=a.epochs, device=a.device, run_dir=a.run_dir)
    except ImportError as exc:
        print(f"\nPATCHTST RUN STOPPED: PyTorch is required ({exc}). Install it as described in the README.", file=sys.stderr)
        sys.exit(2)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\nPATCHTST RUN STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
    print("\n=== PATCHTST SUMMARY (single chronological test split; full detail: metrics_patchtst.json) ===")
    for h in m["horizons"]:
        r = m[f"patchtst_{h}d"]
        e = r[f"day{h}_only"]
        print(f"patchtst_{h}d  day-{h} end point: NSE(per-sensor mean)={e['model']['nse_per_sensor_mean']:.3f} "
              f"persistence={e['persistence']['nse_per_sensor_mean']:.3f} | RMSE {e['model']['rmse']:.3f} vs {e['persistence']['rmse']:.3f} m | "
              f"MAE {e['model']['mae']:.3f} | n={e['n']} | epochs={r['epochs_run']} (best {r['best_epoch']})")
    ro_res = m.get("rolling_origin", {})
    print("rolling-origin table: metrics_table_patchtst.csv" if "summary" in ro_res else f"rolling-origin: {ro_res or 'skipped (--no-rolling)'}")


if __name__ == "__main__":
    main()
