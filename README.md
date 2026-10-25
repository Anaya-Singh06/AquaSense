# AquaSense

Groundwater intelligence prototype: forecast well depth, flag unusual drawdown, and map conditions across USGS monitoring wells with NASA POWER climate.

This repository is the **final B.Tech version**. The dashboard default for a full demo is the **expanded** real-data run (200 wells). A smaller 60-well run and a synthetic demo remain in the same code path.

> Depth is always **metres below ground level** (`groundwater_depth_m_bgl`). Larger values mean a deeper water table (more drawdown).

## What it does

- Pulls daily groundwater depth from the USGS Water Data API (parameter 72019) and daily climate from NASA POWER (rain, temperature, land evaporation, root-zone wetness).
- Cleans records, scores data sufficiency, and trains several forecasters on the same chronological splits.
- Scores models with **expanding-window rolling origin** against persistence (and seasonal-naive in the training pipeline).
- Serves a React dashboard: operational map + forecasts + anomaly feed, and a **Model evaluation** tab from saved metrics (not live training).

## Honest limits

- This is a prototype, not an operational water-management system.
- NASA POWER values are **gridded estimates**, not sensors on the well.
- An Isolation Forest + hydrology-rule flag is a **statistical outlier**, not proof of illegal extraction.
- Map classes (10 m / 20 m bgl) and IDW interpolation are demonstration methods, not regulatory products.
- Skill is uneven across wells. Stable wells make persistence hard to beat. Mean-station NSE can be poor even when pooled RMSE improves.
- Large data, model weights, and per-fold experiment dumps stay **on your machine** (see `.gitignore`). GitHub holds code, configs, tests, and evaluation summaries.

## Repository layout

| Path | Role |
|---|---|
| `src/` | Pipeline, models (LSTM, XGBoost, LightGBM, PatchTST, iTransformer, anomaly), rolling-origin evaluation, IDW |
| `data/` | Fetch/generate scripts; `dwlr_data.csv` + `stations.csv` for the synthetic demo |
| `config/` | USGS/NASA discovery, sufficiency thresholds, model hyperparameters |
| `scripts/` | Pipeline entry points |
| `backend/` | FastAPI: operational artifacts + `/api/evaluation` |
| `frontend/` | Vite + React + Leaflet + Recharts dashboard |
| `tests/` | Unit tests (leakage, metrics, USGS client, evaluation payload, models) |
| `evaluation_reference/` | Frozen rolling-origin summaries used by the evaluation tab |
| `example_outputs_synthetic/` | Small sandbox outputs (not real-groundwater evidence) |

Local-only (not in Git): `.venv/`, `frontend/node_modules/`, `data/raw/`, merged real CSVs, `models/` weights, `outputs/` plots, `experiments/**/units/`.

## Models

| Model | Horizons used | Notes |
|---|---|---|
| Persistence | 7 / 30 / 90 d | Baseline: last observation |
| Seasonal naive | training pipeline | Same calendar window, previous year |
| LSTM | 7 d | Sequence model, PyTorch |
| PatchTST | 7 / 30 / 90 d | Channel-independent patched Transformer |
| iTransformer | 7 / 30 / 90 d | Inverted Transformer |
| XGBoost / LightGBM | 7 / 30 / 90 d | Direct change-in-depth models |
| Isolation Forest + rules | — | Potential abnormal behaviour flags |

Comparison runs score every model on the **intersection** of eligible `(fold, station, origin)` rows. Per-model tables that use each model's own rows are not comparable.

## Results (expanded real set)

Source: USGS daily groundwater + NASA POWER, **200 selected wells**, 1,226,200 daily rows, 2010-01-01 to 2026-10-14. Method: five-fold expanding-window rolling origin, endpoint scoring. Numbers from `evaluation_reference/rolling_comparison_summary_expanded.json`.

**7-day** (~139k samples, 193 stations). Persistence RMSE 0.409 m.

| Model | RMSE (m) | MAE (m) | Median station NSE | RMSE skill vs persistence | Stations beating persistence |
|---|---:|---:|---:|---:|---:|
| PatchTST | 0.367 | 0.159 | 0.942 | 10.4% | 148 / 192 |
| LightGBM | 0.367 | 0.156 | 0.942 | 10.3% | 133 / 192 |
| XGBoost | 0.369 | 0.157 | 0.943 | 9.9% | 132 / 192 |
| iTransformer | 0.375 | 0.161 | 0.941 | 8.3% | 125 / 192 |
| LSTM | 0.402 | 0.181 | 0.929 | 1.7% | 96 / 192 |

**30-day** (~136k samples, 192 stations). Persistence RMSE 0.885 m. Tree models have the lowest pooled RMSE but **negative mean-station NSE** (a few wells dominate the average). PatchTST keeps the best mean-station NSE among learned models.

| Model | RMSE (m) | Mean station NSE | Median station NSE | RMSE skill |
|---|---:|---:|---:|---:|
| LightGBM | 0.768 | −0.16 | 0.621 | 13.2% |
| XGBoost | 0.770 | −0.11 | 0.626 | 12.9% |
| PatchTST | 0.786 | 0.41 | 0.633 | 11.2% |
| iTransformer | 0.795 | 0.29 | 0.606 | 10.2% |

**90-day** (~127k samples, 191 stations). Persistence RMSE 1.69 m. Median-station NSE is **negative for every model**, including persistence. Pooled RMSE still improves (LightGBM skill 20.5%) because of between-station scale. Do not read 90-day forecasts as reliable well-by-well skill.

A smaller 60-well comparison is summarised in `evaluation_reference/rolling_comparison_summary.json` and `evaluation_reference/Results_Current.txt`.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

Optional: copy `frontend/.env.example` to `frontend/.env.local` and add a Carto key for nicer map tiles.

## Run the dashboard (uses artifacts already on disk)

Expanded 200-well demo:

```text
AquaSense_expanded.bat
```

Original 60-well real demo: `AquaSense.bat`  
Requires a completed pipeline for that source (`outputs/` and `models/` on disk).

Or by hand:

```powershell
$env:AQUASENSE_SOURCE="real_expanded"
$env:AQUASENSE_EVAL_SUMMARY="$PWD\evaluation_reference\rolling_comparison_summary_expanded.json"
uvicorn backend.main:app --reload --port 8000
# second terminal:
cd frontend; npm run dev
```

Open http://localhost:5173 — **Operational dashboard** and **Model evaluation**.

## Rebuild data and models

```powershell
python -m unittest discover -s tests -t .
python scripts/check_usgs_api.py                          # live USGS probe
python scripts/run_pipeline.py --source synthetic         # offline demo
python scripts/run_pipeline.py --source real              # ~60 wells
python scripts/run_expanded_pipeline.py                   # up to 200 wells
python scripts/run_rolling_comparison.py --source real_expanded
python scripts/run_itransformer_rolling.py
```

Discovery, sufficiency thresholds, and model hyperparameters live under `config/`. Sufficiency rules were fixed **before** training; do not retune them to pass more wells.

Set `USGS_API_KEY` if you hit HTTP 429. Never commit the key.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Source + whether artifacts loaded |
| GET | `/api/sensors` | Stations, latest depth, anomaly ring |
| GET | `/api/forecast/{sensor_id}` | Observed series + model forecasts |
| GET | `/api/anomalies` | Recent flags with explanations |
| GET | `/api/spatial-grid` | IDW GeoJSON / raster payload |
| GET | `/api/data-summary` | Counts, date range, source badge |
| GET | `/api/evaluation` | Frozen rolling-origin summary for the evaluation tab |

`AQUASENSE_SOURCE` is `synthetic` \| `real` \| `real_expanded`.  
`AQUASENSE_EVAL_SUMMARY` points at a JSON file under `evaluation_reference/`.

## Tests

```powershell
python -m unittest discover -s tests -t .
# or
pytest
```

Torch / XGBoost / LightGBM tests skip if those packages are missing.

## Data sources

- [USGS Water Data API](https://api.waterdata.usgs.gov/ogcapi/) — daily groundwater depth (ft converted to m).
- [NASA POWER](https://power.larc.nasa.gov/) — daily climate at the well coordinates.

Cite USGS and NASA POWER if you publish figures from a local run.
