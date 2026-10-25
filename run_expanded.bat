@echo off
setlocal

REM ============================================================
REM AquaSense - EXPANDED station set (separate pipeline)
REM Downloads + filters a LARGER USGS/NASA station set, then
REM trains and compares EVERY model on it:
REM   LSTM (7d), XGBoost / LightGBM / PatchTST / iTransformer (7/30/90).
REM The original 60-station 'real' results are NOT touched.
REM ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ==========================================
echo  AquaSense - expanded pipeline (all models)
echo ==========================================
echo.
echo Step 1/4  download + filter stations (cached under data\raw\)
echo Step 2/4  main pipeline (preprocess / features / LSTM / XGBoost / anomalies / map)
echo Step 3/4  rolling-origin comparison (LSTM, XGBoost, LightGBM, PatchTST, iTransformer)
echo Step 4/4  refresh the dashboard summary
echo.
echo This can take a while on the first run: the new stations are downloaded once,
echo then cached. A re-run re-uses the finished units already in the output folder.
echo.

"%ROOT%.venv\Scripts\python.exe" scripts\run_expanded_pipeline.py ^
  --config config\real_data_expanded_config.json ^
  --out-dir experiments\rolling_comparison_real_expanded_v1 ^
  --summary-path evaluation_reference\rolling_comparison_summary_expanded.json ^
  --models lstm xgboost lightgbm patchtst itransformer ^
  --horizons 7 30 90 ^
  --device auto
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" goto failed

echo Expanded results written and the dashboard summary refreshed.
echo Now open the dashboard with AquaSense_expanded.bat and check the
echo Map and Pooled evaluation tabs.
goto done

:failed
echo Expanded pipeline FAILED. Exit code %RC% - see the messages above.
echo Nothing outside the expanded files was changed.
echo Re-running this file is safe: finished work is re-used, not repeated.

:done
echo.
pause
endlocal
