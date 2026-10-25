import json
from pathlib import Path
import csv
import statistics


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_FILE = ROOT / "outputs" / "real" / "evaluation.json"
CSV_FILE = ROOT / "outputs" / "real" / "station_level_analysis.csv"
TXT_FILE = ROOT / "outputs" / "real" / "station_level_analysis.txt"


def pct(x):
    return f"{x * 100:.1f}%"


def analyse_model(name, data):
    rows = []

    for station_id, station in data["per_station"].items():
        model = station["model"]
        persistence = station["persistence"]

        rows.append({
            "station_id": station_id,
            "n": station["n"],
            "model_rmse": model["rmse"],
            "persistence_rmse": persistence["rmse"],
            "model_nse": model["nse_pooled"],
            "persistence_nse": persistence["nse_pooled"],
            "rmse_better": model["rmse"] < persistence["rmse"],
            "nse_better": model["nse_pooled"] > persistence["nse_pooled"],
            "rmse_improvement_pct": (
                (persistence["rmse"] - model["rmse"])
                / persistence["rmse"] * 100
            ),
            "nse_difference": (
                model["nse_pooled"] - persistence["nse_pooled"]
            ),
        })

    rmse_better = sum(r["rmse_better"] for r in rows)
    nse_better = sum(r["nse_better"] for r in rows)

    model_nse = [r["model_nse"] for r in rows]
    persistence_nse = [r["persistence_nse"] for r in rows]

    model_rmse = [r["model_rmse"] for r in rows]
    persistence_rmse = [r["persistence_rmse"] for r in rows]

    summary = {
        "model": name,
        "stations": len(rows),
        "rmse_better": rmse_better,
        "nse_better": nse_better,
        "model_mean_nse": statistics.mean(model_nse),
        "model_median_nse": statistics.median(model_nse),
        "model_min_nse": min(model_nse),
        "model_max_nse": max(model_nse),
        "persistence_mean_nse": statistics.mean(persistence_nse),
        "persistence_median_nse": statistics.median(persistence_nse),
        "model_median_rmse": statistics.median(model_rmse),
        "persistence_median_rmse": statistics.median(persistence_rmse),
    }

    return rows, summary


def main():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {EVALUATION_FILE}"
        )

    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    all_rows = []
    summaries = []

    model_map = {
        "lstm_7d": "LSTM 7-day",
        "xgboost_30d": "XGBoost 30-day",
        "xgboost_90d": "XGBoost 90-day",
    }

    for key, display_name in model_map.items():
        rows, summary = analyse_model(display_name, evaluation[key])

        for row in rows:
            row["model"] = display_name
            all_rows.append(row)

        summaries.append(summary)

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "station_id",
        "n",
        "model_rmse",
        "persistence_rmse",
        "rmse_improvement_pct",
        "model_nse",
        "persistence_nse",
        "nse_difference",
        "rmse_better",
        "nse_better",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            writer.writerow({
                field: row[field]
                for field in fieldnames
            })

    # ---------------------------------------------------------
    # Human-readable TXT
    # ---------------------------------------------------------

    with open(TXT_FILE, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("AQUASENSE — STATION-LEVEL MODEL ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "This is a post-processing analysis of the existing "
            "rolling-origin evaluation.\n"
        )
        f.write(
            "No models were retrained and no prediction files were changed.\n\n"
        )

        for s in summaries:
            f.write("-" * 70 + "\n")
            f.write(f"{s['model']}\n")
            f.write("-" * 70 + "\n")

            f.write(f"Stations evaluated: {s['stations']}\n")
            f.write(
                f"Stations where model beats persistence on RMSE: "
                f"{s['rmse_better']} / {s['stations']} "
                f"({pct(s['rmse_better'] / s['stations'])})\n"
            )
            f.write(
                f"Stations where model beats persistence on NSE:  "
                f"{s['nse_better']} / {s['stations']} "
                f"({pct(s['nse_better'] / s['stations'])})\n"
            )

            f.write("\nNSE:\n")
            f.write(f"  Model mean:       {s['model_mean_nse']:.4f}\n")
            f.write(f"  Model median:     {s['model_median_nse']:.4f}\n")
            f.write(f"  Model minimum:    {s['model_min_nse']:.4f}\n")
            f.write(f"  Model maximum:    {s['model_max_nse']:.4f}\n")
            f.write(
                f"  Persistence mean:   "
                f"{s['persistence_mean_nse']:.4f}\n"
            )
            f.write(
                f"  Persistence median: "
                f"{s['persistence_median_nse']:.4f}\n"
            )

            f.write("\nRMSE:\n")
            f.write(
                f"  Model median:       "
                f"{s['model_median_rmse']:.4f} m\n"
            )
            f.write(
                f"  Persistence median: "
                f"{s['persistence_median_rmse']:.4f} m\n"
            )

            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "For an individual station, lower RMSE and higher NSE "
            "rankings are equivalent because both depend on the same "
            "sum of squared errors while the station's observation "
            "count and observed variance remain fixed.\n\n"
        )

        f.write(
            "The difference between pooled RMSE and mean station NSE "
            "therefore comes from aggregation: pooled RMSE weights "
            "individual observations, while mean station NSE gives "
            "each station equal weight.\n\n"
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("AQUASENSE — STATION-LEVEL ANALYSIS")
    print("=" * 70)

    for s in summaries:
        print()
        print(s["model"])
        print("-" * 70)

        print(
            f"RMSE better than persistence: "
            f"{s['rmse_better']} / {s['stations']}"
        )

        print(
            f"NSE better than persistence:  "
            f"{s['nse_better']} / {s['stations']}"
        )

        print(
            f"Model NSE mean / median: "
            f"{s['model_mean_nse']:.4f} / "
            f"{s['model_median_nse']:.4f}"
        )

        print(
            f"Persistence NSE mean / median: "
            f"{s['persistence_mean_nse']:.4f} / "
            f"{s['persistence_median_nse']:.4f}"
        )

        print(
            f"Model NSE range: "
            f"{s['model_min_nse']:.4f} to "
            f"{s['model_max_nse']:.4f}"
        )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(CSV_FILE)
    print(TXT_FILE)
    print()


if __name__ == "__main__":
    main()