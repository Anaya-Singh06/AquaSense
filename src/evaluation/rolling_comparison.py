"""Cross-model comparison of rolling-origin forecasts on IDENTICAL samples (numpy / pandas only; no torch, no lightgbm).

Input: the "long" frame produced by ``rolling_origin.rolling_*`` - one row per (fold, model, horizon, sensor_id, origin_date) with
``y_true``, ``y_model``, ``y_persistence`` and ``y_seasonal_naive`` (the day-h end point; persistence = depth on the origin day).

Why this module exists
  Every model has its own eligibility mask: the tabular models need 31 observed input days + an observed day-h target, the LSTM needs
  30 observed input days + ALL h target days observed, PatchTST needs 96 input days + all h target days. ``rolling_origin.summarize``
  scores each model (and its baselines) on that model's own rows, so numbers of different models are NOT comparable. Here the rows are
  intersected on (fold, sensor_id, origin_date, horizon): every model, persistence and seasonal-naive are then scored on exactly the
  same observations, the same targets, the same fold dates.

Metric definitions (all in metres, day-h end point only)
  pooled RMSE / MAE / NSE : computed once over all rows (all stations, all folds). Pooled NSE is dominated by between-station
                            differences in mean depth and is close to 1 even for poor forecasts - never read it alone.
  mean per-station NSE    : NSE computed separately for every station (all folds pooled; stations with >= ``MIN_STATION_N``
                            scored rows), then averaged. Different quantity from pooled NSE; much harsher.
  station wins/losses     : a station is a WIN if the model's RMSE on that station is strictly lower than persistence's, a LOSS if
                            strictly higher, a TIE otherwise (same rows for both).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import SOURCE_LABELS
from src.evaluation.metrics import mae, nse, rmse

MIN_STATION_N = 10
SHARED_COLUMN_TOL_M = 5e-4     # 0.5 mm: float32 round-off allowance; real misalignments are >= the 1 cm data resolution
KEY = ["fold", "sensor_id", "origin_date"]
BASELINES = ("persistence", "seasonal_naive")
SEQUENCE_MODELS = ("lstm", "patchtst", "itransformer")
TABULAR_MODELS = ("xgboost", "lightgbm")
_PRED_COL = {"persistence": "y_persistence", "seasonal_naive": "y_seasonal_naive"}


# --------------------------------------------------------------------------------------------- sample alignment
def _check_unique(g: pd.DataFrame, what: str) -> None:
    if g.duplicated(KEY).any():
        raise ValueError(f"{what}: duplicate (fold, sensor_id, origin_date) rows - a model was run twice for the same fold")


def common_samples(long: pd.DataFrame, horizon: int, models: Iterable[str] | None = None,
                   require_seasonal: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rows of ``long`` at ``horizon`` that EVERY model in ``models`` forecast (finite ``y_model``), one frame per model concatenated.

    ``require_seasonal`` (default, as in ``rolling_origin.summarize``) also drops rows with no seasonal-naive value, so that all
    three of model / persistence / seasonal-naive are always defined on the same rows. Returns (rows, accounting)."""
    h = long[long["horizon"] == horizon]
    names = sorted(h["model"].unique()) if models is None else [m for m in models if m in set(h["model"])]
    if not names:
        return h.iloc[0:0].copy(), {"models": [], "n_common": 0, "per_model": {}}
    per: dict[str, pd.DataFrame] = {}
    acct: dict[str, Any] = {}
    for m in names:
        g = h[h["model"] == m]
        _check_unique(g, f"{m} h={horizon}")
        ok = np.isfinite(g["y_model"].to_numpy(float)) & np.isfinite(g["y_true"].to_numpy(float)) & np.isfinite(g["y_persistence"].to_numpy(float))
        n_total = int(len(g))
        g = g[ok]
        n_finite = int(len(g))
        if require_seasonal:
            g = g[np.isfinite(g["y_seasonal_naive"].to_numpy(float))]
        per[m] = g.set_index(KEY)
        acct[m] = {"n_own_rows": n_total, "n_own_after_finite_filter": n_finite, "n_own_with_seasonal_naive": int(len(g)),
                   "stations_own": int(g["sensor_id"].nunique()), "folds_present": sorted(int(x) for x in g["fold"].unique())}
    idx = None
    for g in per.values():
        idx = g.index if idx is None else idx.intersection(g.index)
    idx = idx.sort_values() if idx is not None else pd.MultiIndex.from_arrays([[], [], []], names=KEY)
    parts = []
    for m, g in per.items():
        sub = g.loc[idx].reset_index()
        parts.append(sub)
        acct[m]["n_common"] = int(len(sub))
        acct[m]["fraction_of_own_kept"] = float(len(sub) / acct[m]["n_own_with_seasonal_naive"]) if acct[m]["n_own_with_seasonal_naive"] else float("nan")
    out = pd.concat(parts, ignore_index=True) if parts else h.iloc[0:0].copy()
    if len(names) > 1 and len(out):
        # Identical-row guarantee. Truth and baselines must agree across models on the SAME (fold, station, origin); float32 sequence
        # models carry ~1e-5 m round-off, so the shared columns are then taken from ONE reference model (a float64 tabular model if
        # present) and every model is scored against exactly those numbers.
        ref_name = next((m for m in names if m in TABULAR_MODELS), names[0])
        ref = out[out["model"] == ref_name].set_index(KEY).sort_index()
        for m in names:
            if m == ref_name:
                continue
            o = out[out["model"] == m].set_index(KEY).sort_index()
            for col in ("y_true", "y_persistence", "y_seasonal_naive"):
                if not np.allclose(ref[col].to_numpy(float), o[col].to_numpy(float), rtol=0, atol=SHARED_COLUMN_TOL_M, equal_nan=True):
                    raise ValueError(f"h={horizon}: '{col}' differs between {ref_name} and {m} on the SAME (fold, station, origin) by more than "
                                     f"{SHARED_COLUMN_TOL_M} m - models are not being scored against the same targets/baselines")
        shared = ref[["y_true", "y_persistence", "y_seasonal_naive"]]
        keyed = out.set_index(KEY)
        for col in ("y_true", "y_persistence", "y_seasonal_naive"):
            keyed[col] = shared[col].reindex(keyed.index).to_numpy()
        out = keyed.reset_index()
    return out, {"models": names, "n_common": int(len(idx)), "per_model": acct}


# --------------------------------------------------------------------------------------------- metrics
def _block(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "nse_pooled": nse(y, p)}


def station_table(g: pd.DataFrame, col: str, min_n: int = MIN_STATION_N) -> pd.DataFrame:
    """Per-station n / MAE / RMSE / NSE of prediction column ``col`` (stations with fewer than ``min_n`` rows are omitted)."""
    rows = []
    for sid, s in g.groupby("sensor_id", sort=True):
        if len(s) < min_n:
            continue
        rows.append({"sensor_id": sid, "n": int(len(s)), **_block(s["y_true"].to_numpy(float), s[col].to_numpy(float))})
    return pd.DataFrame(rows, columns=["sensor_id", "n", "mae", "rmse", "nse_pooled"])


def score_predictor(g: pd.DataFrame, col: str, min_n: int = MIN_STATION_N) -> dict[str, Any]:
    """Pooled and per-station summary for one prediction column of a frame in which every row is one sample."""
    st = station_table(g, col, min_n)
    out: dict[str, Any] = {"n": int(len(g)), "n_stations": int(g["sensor_id"].nunique()), "n_stations_scored": int(len(st)),
                           "n_folds": int(g["fold"].nunique()), **_block(g["y_true"].to_numpy(float), g[col].to_numpy(float))}
    nse_st = st["nse_pooled"].to_numpy(float) if len(st) else np.array([])
    out["nse_mean_station"] = float(np.nanmean(nse_st)) if np.isfinite(nse_st).any() else float("nan")
    out["nse_median_station"] = float(np.nanmedian(nse_st)) if np.isfinite(nse_st).any() else float("nan")
    out["rmse_median_station"] = float(np.nanmedian(st["rmse"])) if len(st) else float("nan")
    return out


def station_outcomes(g: pd.DataFrame, min_n: int = MIN_STATION_N) -> pd.DataFrame:
    """Per-station model-vs-persistence comparison on the SAME rows (``g`` holds ONE model's common rows)."""
    m, p, s = station_table(g, "y_model", min_n), station_table(g, "y_persistence", min_n), station_table(g, "y_seasonal_naive", min_n)
    t = m.rename(columns={"mae": "mae_model", "rmse": "rmse_model", "nse_pooled": "nse_model"}).merge(
        p.rename(columns={"mae": "mae_persistence", "rmse": "rmse_persistence", "nse_pooled": "nse_persistence"}).drop(columns="n"), on="sensor_id").merge(
        s.rename(columns={"mae": "mae_seasonal_naive", "rmse": "rmse_seasonal_naive", "nse_pooled": "nse_seasonal_naive"}).drop(columns="n"),
        on="sensor_id", how="left")
    d = t["rmse_model"] - t["rmse_persistence"]
    t["outcome_vs_persistence"] = np.where(d < 0, "win", np.where(d > 0, "loss", "tie"))
    return t


def _fold_block(g: pd.DataFrame, col: str, min_n: int = MIN_STATION_N) -> dict[str, Any]:
    r = score_predictor(g, col, min_n)
    return {k: r[k] for k in ("n", "n_stations", "mae", "rmse", "nse_pooled", "nse_mean_station", "nse_median_station", "n_stations_scored")}


def compare_horizon(rows: pd.DataFrame, horizon: int, set_name: str, models: list[str]) -> dict[str, Any]:
    """Full comparison of ``models`` + baselines on the common rows ``rows`` (output of ``common_samples``)."""
    first = rows[rows["model"] == models[0]]
    res: dict[str, Any] = {"set": set_name, "horizon_days": int(horizon), "models": models, "n": int(len(first)),
                           "n_stations": int(first["sensor_id"].nunique()), "n_folds": int(first["fold"].nunique()),
                           "folds": sorted(int(f) for f in first["fold"].unique()), "predictors": {}, "by_fold": {}, "station_outcomes": {}}
    if first.empty:
        return res
    res["predictors"]["persistence"] = {"role": "baseline", **score_predictor(first, "y_persistence")}
    res["predictors"]["seasonal_naive"] = {"role": "baseline", **score_predictor(first, "y_seasonal_naive")}
    pers_pool = res["predictors"]["persistence"]
    folds = sorted(first["fold"].unique())
    for f in folds:
        gf = first[first["fold"] == f]
        res["by_fold"][int(f)] = {"persistence": _fold_block(gf, "y_persistence"), "seasonal_naive": _fold_block(gf, "y_seasonal_naive")}
    for m in models:
        g = rows[rows["model"] == m]
        sc = {"role": "model", **score_predictor(g, "y_model")}
        so = station_outcomes(g)
        n_win, n_loss, n_tie = (int((so["outcome_vs_persistence"] == k).sum()) for k in ("win", "loss", "tie"))
        fold_rmse_better = 0
        for f in folds:
            gf = g[g["fold"] == f]
            blk = _fold_block(gf, "y_model")
            blk["rmse_beats_persistence"] = bool(blk["rmse"] < res["by_fold"][int(f)]["persistence"]["rmse"])
            fold_rmse_better += int(blk["rmse_beats_persistence"])
            res["by_fold"][int(f)][m] = blk
        sc["vs_persistence"] = {
            "rmse_skill_pooled": float(1 - sc["rmse"] / pers_pool["rmse"]) if pers_pool["rmse"] > 0 else float("nan"),
            "mae_skill_pooled": float(1 - sc["mae"] / pers_pool["mae"]) if pers_pool["mae"] > 0 else float("nan"),
            "stations_compared": int(len(so)), "stations_win": n_win, "stations_loss": n_loss, "stations_tie": n_tie,
            "folds_rmse_better": int(fold_rmse_better), "folds_compared": int(len(folds)),
            "delta_nse_mean_station": float(sc["nse_mean_station"] - pers_pool["nse_mean_station"])}
        res["predictors"][m] = sc
        res["station_outcomes"][m] = so
    return res


def build_comparison(long: pd.DataFrame, horizons: Iterable[int] | None = None, require_seasonal: bool = True) -> dict[str, Any]:
    """All common-sample comparisons. For every horizon:
    * ``all_models``     - intersection over every model that produced rows at that horizon (the PRIMARY comparison);
    * ``tabular_only``   - XGBoost/LightGBM intersection, only when sequence models shrank ``all_models`` (a much larger sample).
    """
    out: dict[str, Any] = {"comparisons": {}, "accounting": {}}
    if long.empty:
        return out
    for h in sorted(int(x) for x in (horizons if horizons is not None else long["horizon"].unique())):
        present = sorted(long.loc[long["horizon"] == h, "model"].unique())
        if not present:
            continue
        sets = [("all_models", present)]
        tab = [m for m in present if m in TABULAR_MODELS]
        if len(tab) >= 2 and len(tab) < len(present):
            sets.append(("tabular_only", tab))
        for name, ms in sets:
            rows, acct = common_samples(long, h, ms, require_seasonal)
            out["accounting"][f"{name}_{h}d"] = acct
            if rows.empty:
                out["comparisons"][f"{name}_{h}d"] = {"set": name, "horizon_days": h, "models": ms, "n": 0, "note": "no common samples"}
                continue
            out["comparisons"][f"{name}_{h}d"] = compare_horizon(rows, h, name, ms)
    return out


# --------------------------------------------------------------------------------------------- flat tables
LABELS = {"lstm": "LSTM", "xgboost": "XGBoost", "lightgbm": "LightGBM", "patchtst": "PatchTST", "itransformer": "iTransformer",
          "persistence": "Persistence", "seasonal_naive": "Seasonal naive"}


def comparison_table(comp: dict[str, Any]) -> pd.DataFrame:
    """One row per (set, horizon, predictor): pooled and per-station metrics, skill and station wins/losses vs persistence."""
    rows = []
    for key, c in comp["comparisons"].items():
        for name, p in c.get("predictors", {}).items():
            v = p.get("vs_persistence", {})
            rows.append({"set": c["set"], "horizon_days": c["horizon_days"], "predictor": LABELS.get(name, name), "role": p["role"],
                         "n_samples": p["n"], "n_stations": p["n_stations"], "n_stations_scored": p["n_stations_scored"], "n_folds": p["n_folds"],
                         "RMSE_m": p["rmse"], "MAE_m": p["mae"], "NSE_pooled": p["nse_pooled"], "NSE_mean_station": p["nse_mean_station"],
                         "NSE_median_station": p["nse_median_station"], "RMSE_skill_vs_persistence": v.get("rmse_skill_pooled", np.nan),
                         "MAE_skill_vs_persistence": v.get("mae_skill_pooled", np.nan), "stations_win": v.get("stations_win", np.nan),
                         "stations_loss": v.get("stations_loss", np.nan), "stations_tie": v.get("stations_tie", np.nan),
                         "folds_rmse_better": v.get("folds_rmse_better", np.nan), "folds_compared": v.get("folds_compared", np.nan)})
    return pd.DataFrame(rows)


def by_fold_table(comp: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, c in comp["comparisons"].items():
        for f, blocks in c.get("by_fold", {}).items():
            for name, b in blocks.items():
                rows.append({"set": c["set"], "horizon_days": c["horizon_days"], "fold": int(f), "predictor": LABELS.get(name, name),
                             "n_samples": b["n"], "n_stations": b["n_stations"], "n_stations_scored": b["n_stations_scored"],
                             "RMSE_m": b["rmse"], "MAE_m": b["mae"], "NSE_pooled": b["nse_pooled"], "NSE_mean_station": b["nse_mean_station"],
                             "NSE_median_station": b["nse_median_station"], "rmse_beats_persistence": b.get("rmse_beats_persistence", np.nan)})
    return pd.DataFrame(rows)


def station_table_all(comp: dict[str, Any]) -> pd.DataFrame:
    parts = []
    for key, c in comp["comparisons"].items():
        for m, so in c.get("station_outcomes", {}).items():
            parts.append(so.assign(set=c["set"], horizon_days=c["horizon_days"], model=LABELS.get(m, m)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def accounting_table(comp: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, a in comp["accounting"].items():
        set_name, h = key.rsplit("_", 1)[0], int(key.rsplit("_", 1)[1][:-1])
        for m, v in a["per_model"].items():
            rows.append({"set": set_name, "horizon_days": h, "model": LABELS.get(m, m), "n_common_all_models_in_set": a["n_common"],
                         "n_own_rows": v["n_own_rows"], "n_own_finite": v["n_own_after_finite_filter"],
                         "n_own_with_seasonal_naive": v["n_own_with_seasonal_naive"], "stations_own": v["stations_own"],
                         "folds_present": ",".join(map(str, v["folds_present"])), "n_common": v["n_common"],
                         "fraction_of_own_kept": v["fraction_of_own_kept"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- dashboard (Pooled tab)
DEFAULT_EVALUATION = {"method": "five-fold expanding-window rolling-origin", "scoring": "forecast endpoint at each horizon",
                      "test_period": "the last five rolling-origin test blocks", "seed_policy": "one seed per model/fold",
                      "warning": "folds share training history; not independent replications"}
DEFAULT_INTEGRATION_NOTES = [
    "This summary is a reference for the dashboard, not a substitute for the original per-fold experiment outputs.",
    "Do not fabricate fold-level values or label the summary as live metrics.",
    "If actual experiment artifacts are present in the project, prefer parsing those files; otherwise use this clearly labelled static evaluation summary.",
    "Clearly distinguish pooled metrics from station-level metrics.",
    "Do not claim anomaly detections prove illegal extraction or other causes.",
]


def dashboard_summary(comp: dict[str, Any], manifest: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dashboard 'Pooled evaluation' JSON (GET /api/evaluation) built from a rolling-origin comparison.

    Uses the PRIMARY ``all_models_*`` sets (every model scored on identical samples). Only the numeric ``data`` and
    ``metrics`` blocks are (re)computed; ``title`` / ``source`` / ``evaluation`` / ``integration_notes`` are preserved
    from ``existing`` or filled with defaults. ``manifest`` must carry ``source``/``source_label`` and a ``data`` block
    with ``stations``, ``rows``, ``date_min`` and ``date_max`` (the shape of ``run_manifest.json``)."""
    doc = dict(existing or {})
    label = manifest.get("source_label") or SOURCE_LABELS.get(manifest.get("source", ""), "")
    doc.setdefault("title", "AquaSense rolling-origin evaluation summary")
    doc.setdefault("source", f"Real USGS groundwater observations + NASA POWER environmental estimates ({label})" if label
                   else "Real USGS groundwater observations + NASA POWER environmental estimates")
    doc.setdefault("evaluation", DEFAULT_EVALUATION)
    doc.setdefault("integration_notes", DEFAULT_INTEGRATION_NOTES)
    data = manifest["data"]
    doc["data"] = {**doc.get("data", {}), "stations_selected": data["stations"], "rows": data["rows"],
                   "date_start": data["date_min"], "date_end": data["date_max"]}
    metrics: dict[str, dict] = {}
    for key, c in comp["comparisons"].items():
        if not key.startswith("all_models_") or not c.get("n"):
            continue
        h = int(c["horizon_days"])
        preds = c["predictors"]
        pers = preds["persistence"]
        block: dict = {"sample_rows": int(c["n"]), "stations": int(c["n_stations"]),
                       "persistence": {"rmse_m": round(pers["rmse"], 4), "mae_m": round(pers["mae"], 4),
                                       "mean_station_nse": round(pers["nse_mean_station"], 4),
                                       "median_station_nse": round(pers["nse_median_station"], 4)},
                       "models": {}}
        for name, p in preds.items():
            if p.get("role") != "model":
                continue
            v = p.get("vs_persistence", {})
            block["models"][LABELS.get(name, name)] = {
                "rmse_m": round(p["rmse"], 4), "mae_m": round(p["mae"], 4),
                "mean_station_nse": round(p["nse_mean_station"], 4), "median_station_nse": round(p["nse_median_station"], 4),
                "rmse_skill": round(v.get("rmse_skill_pooled", float("nan")), 4),
                "station_wins": int(v.get("stations_win", 0)), "station_total": int(v.get("stations_compared", 0)),
                "folds_better": int(v.get("folds_rmse_better", 0)), "folds_total": int(v.get("folds_compared", 0))}
        metrics[f"{h}d"] = block
    doc["metrics"] = metrics
    return doc
