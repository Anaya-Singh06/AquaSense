"""Per-station data-sufficiency report (run BEFORE training on real data).

  python scripts/data_sufficiency.py --source real        # uses data/real_data.csv
  python scripts/data_sufficiency.py --source synthetic

Writes outputs[/real]/data_sufficiency.csv and .md and prints the table and the dataset-level conclusion.
Thresholds: config/real_data_config.json -> "sufficiency" (fixed in advance; do not tune them to pass stations).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, Paths  # noqa: E402
from src.evaluation.sufficiency import overall_conclusion, sufficiency_table  # noqa: E402


def run(source: str) -> tuple[pd.DataFrame, dict]:
    paths = Paths(source)
    if not paths.data_csv.exists():
        raise SystemExit(f"{paths.data_csv} not found. Real data: python data/fetch_real_data.py")
    df = pd.read_csv(paths.data_csv, parse_dates=["date"])
    th = json.loads((REPO_ROOT / "config" / "real_data_config.json").read_text())["sufficiency"]
    tab = sufficiency_table(df, df["date"].min(), df["date"].max(), th)
    concl = overall_conclusion(tab)
    paths.ensure_dirs()
    tab.to_csv(paths.outputs_dir / "data_sufficiency.csv", index=False)
    cols = ["sensor_id", "years", "n_obs", "missing_fraction", "longest_gap_days", "std_m", "seasonal_amplitude_m",
            "test_variance", "dynamics", "persistence_trivial", "verdict"]
    md = tab[cols].round(3).to_markdown(index=False) if hasattr(tab, "to_markdown") and _has_tabulate() else tab[cols].round(3).to_string(index=False)
    (paths.outputs_dir / "data_sufficiency.md").write_text(md + "\n\n" + json.dumps(concl, indent=2) + "\n")
    (paths.outputs_dir / "data_sufficiency_summary.json").write_text(json.dumps(concl, indent=2))
    return tab, concl


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real"], default="real")
    a = ap.parse_args()
    tab, concl = run(a.source)
    show = tab[["sensor_id", "years", "n_obs", "missing_fraction", "longest_gap_days", "std_m", "seasonal_amplitude_m",
                "test_variance", "dynamics", "verdict"]].copy()
    show["missing_fraction"] = (show["missing_fraction"] * 100).round(1)
    show = show.rename(columns={"missing_fraction": "missing_%", "longest_gap_days": "longest_gap_d"})
    print(show.round(3).to_string(index=False))
    print("\nDATASET CONCLUSION:", json.dumps(concl, indent=2))


if __name__ == "__main__":
    main()
