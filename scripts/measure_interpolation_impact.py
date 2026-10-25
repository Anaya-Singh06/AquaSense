"""Exact interpolation-impact accounting on the real data (no training, no download).
Usage:  python scripts/measure_interpolation_impact.py [--source real]   -> outputs/<source>/interpolation_impact.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import TARGET, Paths
from src.models.xgboost_forecaster import split_rows
from src.preprocessing.features import add_targets, build_features
from src.preprocessing.preprocess import load_dataset, preprocess
from src.preprocessing.splits import make_split_dates
from src.preprocessing.windows import make_windows

ap = argparse.ArgumentParser(); ap.add_argument("--source", default="real"); a = ap.parse_args()
p = Paths(a.source)
clean, _ = preprocess(load_dataset(p.data_csv))
feats = add_targets(build_features(clean)); sp = make_split_dates(clean["date"])
imp = clean["depth_imputed"]; have = clean[TARGET].notna()
out = {"rows_total": len(clean), "rows_with_gw_value": int(have.sum()), "observed": int(((imp == 0) & have).sum()),
       "imputed_1": int((imp == 1).sum()), "imputed_2": int((imp == 2).sum()), "imputed_total": int((imp > 0).sum()),
       "pct_interpolated": round(100 * (imp > 0).sum() / have.sum(), 3),
       "stations_affected": int(clean.loc[imp > 0, "sensor_id"].nunique()),
       "imputed_per_station": clean[imp > 0].groupby("sensor_id").size().sort_values(ascending=False).to_dict(), "xgb": {}}
for h in (30, 90):
    st: dict = {}; split_rows(feats, h, sp, stats=st); out["xgb"][f"{h}d"] = st
d = clean.sort_values(["sensor_id", "date"]).copy(); d["x"] = d[TARGET]; d["obs"] = (d["depth_imputed"] == 0).astype(int)
out["lstm"] = make_windows(d, ["x"], "x", 30, 7, sp, observed_col="obs")["stats"]
Path(p.outputs_dir, "interpolation_impact.json").write_text(json.dumps(out, indent=2, default=int)); print(json.dumps(out, indent=2, default=int)[:3000])
