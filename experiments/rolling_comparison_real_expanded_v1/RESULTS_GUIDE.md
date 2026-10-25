# Rolling-origin comparison - USGS + NASA POWER (expanded station set)

Generated 2026-10-25 08:06 UTC | data sha256 baf68b0370b8... | rows 1226200 | stations 200 | 2010-01-01 to 2026-10-14

Scoring: day-h end point only; a forecast belongs to a fold iff its TARGET date is in (origin, test_end]; every model, persistence and seasonal-naive are scored on the SAME rows (intersection of eligible rows). These numbers are NOT comparable with the single-split 'full-horizon' RMSE/MAE (which average all h days); the matching single-split quantity is the 'day-h only' block.

## Folds

| fold | train_end | origin (val_end) | test_end |
|---|---|---|---|
| 0 | 2023-10-30 | 2024-04-27 | 2024-10-24 |
| 1 | 2024-04-27 | 2024-10-24 | 2025-04-22 |
| 2 | 2024-10-24 | 2025-04-22 | 2025-10-19 |
| 3 | 2025-04-22 | 2025-10-19 | 2026-04-17 |
| 4 | 2025-10-19 | 2026-04-17 | 2026-10-14 |

## 7-day horizon - set `all_models` (iTransformer, LightGBM, LSTM, PatchTST, XGBoost)

139030 identical samples, 193 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.4091 | 0.1780 | 0.999821 | 0.827 | 0.932 | - | - | - |
| Seasonal naive | 2.6428 | 0.8602 | 0.992537 | -0.888 | -0.424 | - | - | - |
| iTransformer | 0.3750 | 0.1608 | 0.999850 | 0.810 | 0.941 | 0.083 | 125/67/0 | 5/5 |
| LightGBM | 0.3671 | 0.1563 | 0.999856 | 0.788 | 0.942 | 0.103 | 133/59/0 | 5/5 |
| LSTM | 0.4021 | 0.1808 | 0.999827 | 0.569 | 0.929 | 0.017 | 96/96/0 | 5/5 |
| PatchTST | 0.3668 | 0.1585 | 0.999856 | 0.837 | 0.942 | 0.104 | 148/44/0 | 5/5 |
| XGBoost | 0.3688 | 0.1567 | 0.999855 | 0.789 | 0.943 | 0.099 | 132/60/0 | 5/5 |

## 7-day horizon - set `tabular_only` (LightGBM, XGBoost)

151200 identical samples, 194 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.4081 | 0.1786 | 0.999828 | 0.832 | 0.936 | - | - | - |
| Seasonal naive | 2.5677 | 0.8539 | 0.993207 | -0.810 | -0.423 | - | - | - |
| LightGBM | 0.3671 | 0.1575 | 0.999861 | 0.793 | 0.941 | 0.100 | 137/57/0 | 5/5 |
| XGBoost | 0.3689 | 0.1580 | 0.999860 | 0.793 | 0.941 | 0.096 | 135/59/0 | 5/5 |

## 30-day horizon - set `all_models` (iTransformer, LightGBM, PatchTST, XGBoost)

135554 identical samples, 192 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.8848 | 0.4590 | 0.999150 | 0.436 | 0.578 | - | - | - |
| Seasonal naive | 2.6697 | 0.8632 | 0.992264 | -1.056 | -0.418 | - | - | - |
| iTransformer | 0.7947 | 0.4055 | 0.999314 | 0.294 | 0.606 | 0.102 | 107/85/0 | 5/5 |
| LightGBM | 0.7682 | 0.3967 | 0.999359 | -0.163 | 0.621 | 0.132 | 115/77/0 | 5/5 |
| PatchTST | 0.7855 | 0.4020 | 0.999330 | 0.411 | 0.633 | 0.112 | 124/68/0 | 5/5 |
| XGBoost | 0.7703 | 0.3963 | 0.999356 | -0.115 | 0.626 | 0.129 | 118/74/0 | 5/5 |

## 30-day horizon - set `tabular_only` (LightGBM, XGBoost)

150202 identical samples, 194 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.8818 | 0.4590 | 0.999187 | 0.445 | 0.586 | - | - | - |
| Seasonal naive | 2.5752 | 0.8551 | 0.993067 | -0.809 | -0.416 | - | - | - |
| LightGBM | 0.7697 | 0.4001 | 0.999381 | -0.255 | 0.618 | 0.127 | 117/77/0 | 5/5 |
| XGBoost | 0.7711 | 0.3996 | 0.999378 | -0.195 | 0.629 | 0.126 | 119/75/0 | 5/5 |

## 90-day horizon - set `all_models` (iTransformer, LightGBM, PatchTST, XGBoost)

127114 identical samples, 191 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|---|
| Persistence | 1.6945 | 0.9481 | 0.996725 | -0.813 | -0.796 | - | - | - |
| Seasonal naive | 2.7389 | 0.8698 | 0.991445 | -1.284 | -0.441 | - | - | - |
| iTransformer | 1.5908 | 0.8486 | 0.997114 | -1.895 | -0.332 | 0.061 | 99/92/0 | 4/5 |
| LightGBM | 1.3468 | 0.7974 | 0.997931 | -6.313 | -0.314 | 0.205 | 107/84/0 | 5/5 |
| PatchTST | 1.4703 | 0.8169 | 0.997534 | -2.270 | -0.257 | 0.132 | 105/86/0 | 5/5 |
| XGBoost | 1.3581 | 0.7969 | 0.997897 | -6.372 | -0.322 | 0.199 | 107/84/0 | 5/5 |

## 90-day horizon - set `tabular_only` (LightGBM, XGBoost)

149269 identical samples, 194 stations, folds [0, 1, 2, 3, 4].

| predictor | RMSE m | MAE m | NSE pooled | NSE mean per-station | NSE median per-station | RMSE skill vs persistence | stations win/loss/tie | folds better |
|---|---|---|---|---|---|---|---|---|
| Persistence | 1.6919 | 0.9465 | 0.996954 | -0.654 | -0.748 | - | - | - |
| Seasonal naive | 2.5889 | 0.8592 | 0.992870 | -0.837 | -0.429 | - | - | - |
| LightGBM | 1.3749 | 0.8089 | 0.997989 | -6.800 | -0.317 | 0.187 | 109/85/0 | 5/5 |
| XGBoost | 1.3847 | 0.8087 | 0.997960 | -7.057 | -0.330 | 0.182 | 109/85/0 | 5/5 |

## Not evaluated / limitations

- LSTM at 30 d: NOT EVALUATED - LSTM architecture is fixed to a 7-day output.
- LSTM at 90 d: NOT EVALUATED - LSTM architecture is fixed to a 7-day output.
- Ridge: not found anywhere in the project source, so not evaluated (nothing was invented).
- Random Forest: not found anywhere in the project source, so not evaluated (nothing was invented).
- Extra Trees: not found anywhere in the project source, so not evaluated (nothing was invented).
- HistGradientBoosting (as a model in its own right): not found anywhere in the project source, so not evaluated (nothing was invented).
- Single seed (42) per model and fold: no multi-seed variability is reported.
- The 5 rolling-origin test blocks cover target dates 2024-04-28 to 2026-10-14 only; folds share most of their training data, so they are not independent replications.
- Pooled NSE is close to 1 for every forecaster because between-station differences dominate; use mean per-station NSE, skill and station wins/losses.
