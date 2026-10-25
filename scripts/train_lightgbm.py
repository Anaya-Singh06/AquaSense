"""Train + evaluate the additional LightGBM forecaster (30 d / 90 d) on the EXISTING data CSV.

    python scripts/train_lightgbm.py [--source synthetic|real] [--no-rolling] [--no-xgb-compare] [--lgbm-config FILE]

Never regenerates/downloads data and never overwrites existing models/results. Writes (in the source's models/ and outputs/ dirs):
  models:  lgbm_30d.txt, lgbm_30d_meta.json, lgbm_90d.txt, lgbm_90d_meta.json
  outputs: metrics_lightgbm.json, metrics_table_lightgbm.csv, evaluation_lightgbm.json, forecasts_lightgbm.json,
           lgbm_30d_actual_vs_predicted.png, lgbm_90d_actual_vs_predicted.png
Hyper-parameters: config/lightgbm_config.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_lightgbm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--no-rolling", action="store_true", help="skip the rolling-origin evaluation (single chronological split only)")
    ap.add_argument("--no-xgb-compare", action="store_true", help="rolling-origin: do not re-run XGBoost alongside LightGBM")
    ap.add_argument("--lgbm-config", default=None, help="alternative hyper-parameter JSON (default config/lightgbm_config.json)")
    a = ap.parse_args()
    try:
        m = run_lightgbm(a.source, rolling=not a.no_rolling, compare_xgb=not a.no_xgb_compare, lgbm_config=a.lgbm_config)
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"\nLIGHTGBM RUN STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
    print("\n=== LIGHTGBM SUMMARY (single chronological test split; full detail: metrics_lightgbm.json) ===")
    for k in ("lgbm_30d", "lgbm_90d"):
        r = m[k]
        print(f"{k:9s} NSE(per-sensor mean)={r['model']['nse_per_sensor_mean']:.3f} persistence={r['persistence']['nse_per_sensor_mean']:.3f} | "
              f"RMSE {r['model']['rmse']:.3f} vs {r['persistence']['rmse']:.3f} m | MAE {r['model']['mae']:.3f} | n={r['n']}")
    print("rolling-origin table: metrics_table_lightgbm.csv" if "summary" in m.get("rolling_origin", {}) else f"rolling-origin: {m.get('rolling_origin')}")


if __name__ == "__main__":
    main()
