@echo off
setlocal

REM ============================================================
REM AquaSense - iTransformer ONLY (separate runner)
REM Adds the iTransformer to the existing rolling-origin
REM comparison. It RE-USES the already-trained LSTM / XGBoost /
REM LightGBM / PatchTST results and trains ONLY the iTransformer.
REM The dashboard launcher (AquaSense.bat) is left untouched.
REM ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ==========================================
echo   AquaSense - add iTransformer (rolling)
echo ==========================================
echo.
echo Re-using existing model results (no retraining of LSTM / XGBoost / LightGBM / PatchTST).
echo Training the iTransformer for each horizon and fold...
echo.

"%ROOT%.venv\Scripts\python.exe" scripts\run_itransformer_rolling.py --source real --out-dir experiments\rolling_comparison_real_v1 --horizons 7 30 90 --device auto
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
  echo iTransformer run FAILED (exit code %RC%). Nothing else was changed; see the messages above.
) else (
  echo iTransformer results written and the dashboard summary refreshed.
  echo Now open the dashboard with AquaSense.bat and check the
  echo 'Map ^& operations' and 'Pooled evaluation' tabs.
)
echo.
pause
endlocal
