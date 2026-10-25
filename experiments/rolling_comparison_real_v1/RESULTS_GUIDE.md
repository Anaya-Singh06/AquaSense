# iTransformer added to the rolling-origin comparison

Generated 2026-10-23 19:31 UTC | data sha256 1a4faa110c87... | rows 366660 | stations 60 | 2010-01-01 to 2026-09-24

Re-used models: LightGBM, LSTM, PatchTST, XGBoost (NOT retrained). Added: iTransformer.

Scoring: day-h end point only; a forecast belongs to a fold iff its TARGET date is in (origin, test_end]; every model, persistence and seasonal-naive are scored on the SAME (fold, station, origin) rows (all-models intersection).

## Folds

| fold | train_end | origin (val_end) | test_end |
|---|---|---|---|
| 0 | 2023-10-10 | 2024-04-07 | 2024-10-04 |
| 1 | 2024-04-07 | 2024-10-04 | 2025-04-02 |
| 2 | 2024-10-04 | 2025-04-02 | 2025-09-29 |
| 3 | 2025-04-02 | 2025-09-29 | 2026-03-28 |
| 4 | 2025-09-29 | 2026-03-28 | 2026-09-24 |

## 7-day horizon - all models (iTransformer, LightGBM, LSTM, PatchTST, XGBoost)

40586 identical samples, 57 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|
| Persistence | 0.3996 | 0.1812 | 0.819 | 0.936 | - | - | - |
| Seasonal naive | 4.6659 | 1.2324 | -1.682 | -0.439 | - | - | - |
| iTransformer | 0.3361 | 0.1610 | 0.823 | 0.940 | 0.159 | 32/25/0 | 5/5 |
| LightGBM | 0.3466 | 0.1613 | 0.813 | 0.942 | 0.133 | 38/19/0 | 5/5 |
| LSTM | 0.3360 | 0.1682 | 0.779 | 0.941 | 0.159 | 31/26/0 | 5/5 |
| PatchTST | 0.3340 | 0.1584 | 0.840 | 0.948 | 0.164 | 41/16/0 | 5/5 |
| XGBoost | 0.3459 | 0.1616 | 0.797 | 0.942 | 0.134 | 39/18/0 | 5/5 |

## 30-day horizon - all models (iTransformer, LightGBM, PatchTST, XGBoost)

39438 identical samples, 57 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|
| Persistence | 0.9309 | 0.4575 | 0.474 | 0.639 | - | - | - |
| Seasonal naive | 4.7266 | 1.2418 | -1.711 | -0.419 | - | - | - |
| iTransformer | 0.8966 | 0.4393 | 0.377 | 0.631 | 0.037 | 27/30/0 | 4/5 |
| LightGBM | 0.8313 | 0.4133 | 0.150 | 0.685 | 0.107 | 36/21/0 | 5/5 |
| PatchTST | 0.8735 | 0.4275 | 0.441 | 0.699 | 0.062 | 30/27/0 | 5/5 |
| XGBoost | 0.8280 | 0.4127 | 0.156 | 0.682 | 0.111 | 36/21/0 | 5/5 |

## 90-day horizon - all models (iTransformer, LightGBM, PatchTST, XGBoost)

36563 identical samples, 56 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|
| Persistence | 1.7267 | 0.9260 | -0.664 | -0.604 | - | - | - |
| Seasonal naive | 4.8921 | 1.2658 | -2.010 | -0.572 | - | - | - |
| iTransformer | 1.6687 | 0.8639 | -0.942 | -0.061 | 0.034 | 32/24/0 | 4/5 |
| LightGBM | 1.4045 | 0.7888 | -3.544 | -0.104 | 0.187 | 30/26/0 | 5/5 |
| PatchTST | 1.6887 | 0.8644 | -1.223 | -0.050 | 0.022 | 34/22/0 | 4/5 |
| XGBoost | 1.4108 | 0.7923 | -3.519 | -0.067 | 0.183 | 32/24/0 | 5/5 |

## Notes

- Single seed (42) per model and fold: no multi-seed variability is reported.
- The 5 rolling-origin test blocks cover target dates 2024-04-08 to 2026-09-24 only; folds share most of their training data, so they are not independent replications.
- Pooled NSE is close to 1 for every forecaster because between-station differences dominate; use mean per-station NSE, skill and station wins/losses.
- The LSTM / XGBoost / LightGBM / PatchTST rows were RE-USED from the earlier run (not retrained); only the iTransformer is new.
