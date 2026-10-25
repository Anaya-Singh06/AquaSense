"""One-command pipeline: generate/load data -> preprocess -> train -> evaluate -> write outputs.

Offline demo:   python scripts/run_pipeline.py --source synthetic
Real data:      python scripts/run_pipeline.py --source real      (downloads/caches USGS + NASA POWER; needs Internet the first time)
                python scripts/run_pipeline.py --source real --refresh   (re-download, ignoring data/raw/)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--keep-data", action="store_true", help="do not regenerate the synthetic CSV if it exists")
    ap.add_argument("--skip-lstm", action="store_true", help="skip the PyTorch LSTM (faster smoke runs)")
    ap.add_argument("--lstm-epochs", type=int, default=None, help="max LSTM epochs (default 60)")
    ap.add_argument("--refresh", action="store_true", help="real: ignore the data/raw cache and download again")
    ap.add_argument("--offline", action="store_true", help="real: use only cached raw files (no network)")
    ap.add_argument("--no-fetch", action="store_true", help="real: use the existing data/real_data.csv, never download")
    ap.add_argument("--no-rolling", action="store_true", help="skip the rolling-origin evaluation (faster smoke runs)")
    a = ap.parse_args()
    try:
        m = run(a.source, regenerate=not a.keep_data, skip_lstm=a.skip_lstm, lstm_epochs=a.lstm_epochs, refresh=a.refresh,
                fetch=not a.no_fetch, offline=a.offline, rolling=not a.no_rolling)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nPIPELINE STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
    print("\n=== SUMMARY (full detail: metrics.json in the outputs folder) ===")
    for k in ("lstm_7d", "xgb_30d", "xgb_90d"):
        r = m.get(k, {})
        if "model" in r:
            print(f"{k:8s} model NSE(per-sensor mean)={r['model']['nse_per_sensor_mean']:.3f} "
                  f"persistence={r['persistence']['nse_per_sensor_mean']:.3f} | RMSE {r['model']['rmse']:.3f} vs {r['persistence']['rmse']:.3f} m")
        else:
            print(f"{k:8s} {r.get('status', r)}")


if __name__ == "__main__":
    main()
