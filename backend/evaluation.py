"""Model Evaluation results: a read-only view of the completed offline rolling-origin evaluation.

Single source of truth: ``evaluation_reference/rolling_comparison_summary.json`` (override the location with
the AQUASENSE_EVAL_SUMMARY environment variable). Station-level PatchTST / LightGBM / iTransformer scores come from
``experiments/rolling_comparison_real_v1/comparison_station_level.csv`` (override with AQUASENSE_EVAL_STATIONS).
This module never trains or evaluates anything, never touches operational model files, and uses only the
standard library, so it works even when the forecasting service, models or ML dependencies are unavailable.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "evaluation_reference" / "rolling_comparison_summary.json"
DEFAULT_STATIONS_PATH = ROOT / "experiments" / "rolling_comparison_real_v1" / "comparison_station_level.csv"
HORIZON_ORDER = ("7d", "30d", "90d")
METRIC_KEYS = ("rmse_m", "mae_m", "mean_station_nse", "median_station_nse")
MODEL_KEYS = METRIC_KEYS + ("rmse_skill", "station_wins", "station_total", "folds_better", "folds_total")
MAP_MODELS = ("PatchTST", "LightGBM", "iTransformer")
STATION_SET = "all_models"  # same sample intersection used in the pooled summary


class EvaluationUnavailable(Exception):
    """The summary file is missing, unreadable or does not have the expected structure."""


def summary_path() -> Path:
    return Path(os.environ.get("AQUASENSE_EVAL_SUMMARY") or DEFAULT_PATH)


def _validate(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise EvaluationUnavailable("evaluation summary is not a JSON object")
    for key in ("data", "evaluation", "metrics"):
        if not isinstance(doc.get(key), dict):
            raise EvaluationUnavailable(f"evaluation summary is missing the '{key}' section")
    for h, block in doc["metrics"].items():
        if not isinstance(block, dict) or "persistence" not in block or not isinstance(block.get("models"), dict):
            raise EvaluationUnavailable(f"horizon '{h}' needs 'persistence' and 'models'")
        for k in ("sample_rows", "stations"):
            if k not in block:
                raise EvaluationUnavailable(f"horizon '{h}' is missing '{k}'")
        for k in METRIC_KEYS:
            if k not in block["persistence"]:
                raise EvaluationUnavailable(f"horizon '{h}' persistence is missing '{k}'")
        for name, m in block["models"].items():
            missing = [k for k in MODEL_KEYS if k not in m]
            if missing:
                raise EvaluationUnavailable(f"horizon '{h}' model '{name}' is missing {missing}")


def _best(models: dict[str, dict], key: str, lowest: bool) -> str | None:
    cands = {n: m[key] for n, m in models.items() if m.get(key) is not None}
    if not cands:
        return None
    return (min if lowest else max)(cands, key=cands.get)


def _derive(block: dict) -> dict[str, Any]:
    """Facts computed from the numbers (not typed in), so the narrative cannot drift from the data."""
    models, base = block["models"], block["persistence"]
    return {
        "best_rmse": _best(models, "rmse_m", True),
        "best_mae": _best(models, "mae_m", True),
        "best_mean_nse": _best(models, "mean_station_nse", False),
        "best_median_nse": _best(models, "median_station_nse", False),
        "best_station_wins": _best(models, "station_wins", False),
        # models whose pooled RMSE beats persistence but whose mean station NSE does not
        "pooled_better_station_worse": sorted(
            n for n, m in models.items()
            if m["rmse_m"] < base["rmse_m"] and m["mean_station_nse"] < base["mean_station_nse"]),
        "beat_persistence_rmse": sorted(n for n, m in models.items() if m["rmse_m"] < base["rmse_m"]),
        "mean_nse_below_zero": sorted(n for n, m in models.items() if m["mean_station_nse"] < 0),
        "persistence_mean_nse_below_zero": base["mean_station_nse"] < 0,
        "models_ranked_by_rmse": sorted(models, key=lambda n: models[n]["rmse_m"]),
    }


def stations_path() -> Path:
    return Path(os.environ.get("AQUASENSE_EVAL_STATIONS") or DEFAULT_STATIONS_PATH)


def _num(value: str | None, digits: int = 6) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _rmse_skill(model_rmse: float | None, persistence_rmse: float | None) -> float | None:
    if model_rmse is None or persistence_rmse is None or persistence_rmse <= 0:
        return None
    return round(1.0 - (model_rmse / persistence_rmse), 6)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2, 6)


def _summarise_horizon(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MAP_MODELS:
        subset = [r for r in rows if r.get("model") == name]
        if not subset:
            continue
        wins = sum(1 for r in subset if r["outcome_vs_persistence"] == "win")
        out[name] = {
            "stations": len(subset),
            "station_wins": wins,
            "station_losses": len(subset) - wins,
            "median_rmse_m": _median([r["rmse_m"] for r in subset if r["rmse_m"] is not None]),
            "median_nse": _median([r["nse"] for r in subset if r["nse"] is not None]),
        }
    both = {}
    for row in rows:
        both.setdefault(row["sensor_id"], {})[row["model"]] = row
    patch_wins = lgbm_wins = ties = 0
    for pair in both.values():
        a, b = pair.get("PatchTST"), pair.get("LightGBM")
        if not a or not b or a["rmse_m"] is None or b["rmse_m"] is None:
            continue
        if a["rmse_m"] < b["rmse_m"]:
            patch_wins += 1
        elif b["rmse_m"] < a["rmse_m"]:
            lgbm_wins += 1
        else:
            ties += 1
    out["compare"] = {"patchtst_lower_rmse": patch_wins, "lightgbm_lower_rmse": lgbm_wins, "ties": ties,
                      "stations": patch_wins + lgbm_wins + ties}
    return out


def load_station_evaluation() -> dict[str, Any]:
    """Per-station map-model scores (PatchTST / LightGBM / iTransformer) for the all_models comparison set."""
    path = stations_path()
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        raise EvaluationUnavailable(f"station evaluation not found at {path}") from None
    except OSError as exc:
        raise EvaluationUnavailable(f"station evaluation could not be read ({exc})") from None
    required = {"sensor_id", "n", "mae_model", "rmse_model", "nse_model", "mae_persistence", "rmse_persistence",
                "nse_persistence", "outcome_vs_persistence", "set", "horizon_days", "model"}
    stations: dict[str, dict[str, dict[str, Any]]] = {}
    horizon_rows: dict[str, list[dict[str, Any]]] = {h: [] for h in HORIZON_ORDER}
    try:
        with handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or required - set(reader.fieldnames):
                missing = sorted(required - set(reader.fieldnames or []))
                raise EvaluationUnavailable(f"station evaluation is missing columns {missing}")
            for raw in reader:
                if raw.get("set") != STATION_SET or raw.get("model") not in MAP_MODELS:
                    continue
                try:
                    horizon = f"{int(raw['horizon_days'])}d"
                except (TypeError, ValueError):
                    continue
                if horizon not in HORIZON_ORDER:
                    continue
                record = {
                    "model": raw["model"],
                    "sensor_id": raw["sensor_id"],
                    "n": int(float(raw["n"])),
                    "rmse_m": _num(raw["rmse_model"], 4),
                    "mae_m": _num(raw["mae_model"], 4),
                    "nse": _num(raw["nse_model"], 4),
                    "rmse_persistence": _num(raw["rmse_persistence"], 4),
                    "mae_persistence": _num(raw["mae_persistence"], 4),
                    "nse_persistence": _num(raw["nse_persistence"], 4),
                    "outcome_vs_persistence": raw["outcome_vs_persistence"],
                }
                record["rmse_skill"] = _rmse_skill(record["rmse_m"], record["rmse_persistence"])
                stations.setdefault(raw["sensor_id"], {}).setdefault(horizon, {})[raw["model"]] = {
                    k: v for k, v in record.items() if k not in ("model", "sensor_id")
                }
                horizon_rows[horizon].append(record)
    except EvaluationUnavailable:
        raise
    except (OSError, ValueError) as exc:
        raise EvaluationUnavailable(f"station evaluation could not be parsed ({exc})") from None
    if not stations:
        raise EvaluationUnavailable("station evaluation has no rows for the map models in the all_models set")
    for sid, horizons in stations.items():
        for horizon, models in horizons.items():
            patch, lgbm = models.get("PatchTST"), models.get("LightGBM")
            if not patch or not lgbm or patch["rmse_m"] is None or lgbm["rmse_m"] is None:
                continue
            delta = round(lgbm["rmse_m"] - patch["rmse_m"], 4)
            if delta > 0:
                winner = "PatchTST"
            elif delta < 0:
                winner = "LightGBM"
            else:
                winner = "tie"
            models["compare"] = {"winner": winner, "rmse_delta_m": abs(delta)}
    horizons = [h for h in HORIZON_ORDER if horizon_rows[h]]
    return {
        "kind": "completed_offline_station_evaluation",
        "kind_label": "Completed offline station scores (not live training output)",
        "source_file": path.name,
        "comparison_set": STATION_SET,
        "models": list(MAP_MODELS),
        "horizons": horizons,
        "n_stations": len(stations),
        "stations": stations,
        "summary": {h: _summarise_horizon(horizon_rows[h]) for h in horizons},
    }


def load_evaluation() -> dict[str, Any]:
    path = summary_path()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvaluationUnavailable(f"evaluation summary not found at {path}") from None
    except (OSError, ValueError) as exc:
        raise EvaluationUnavailable(f"evaluation summary could not be read ({exc})") from None
    _validate(doc)
    horizons = [h for h in HORIZON_ORDER if h in doc["metrics"]] + \
              [h for h in doc["metrics"] if h not in HORIZON_ORDER]
    return {
        **doc,
        "kind": "completed_offline_evaluation",
        "kind_label": "Completed offline evaluation results (not live training output)",
        "source_file": path.name,
        "horizons": horizons,
        "derived": {h: _derive(doc["metrics"][h]) for h in horizons},
    }
