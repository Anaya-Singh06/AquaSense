@echo off
setlocal

REM ============================================================
REM AquaSense - dashboard on the EXPANDED station set
REM Same dashboard as AquaSense.bat, but the source and the
REM evaluation files point at the expanded pipeline results.
REM Requires run_expanded.bat to have finished first.
REM ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "AQUASENSE_SOURCE=real_expanded"
set "AQUASENSE_EVAL_SUMMARY=%ROOT%evaluation_reference\rolling_comparison_summary_expanded.json"
set "AQUASENSE_EVAL_STATIONS=%ROOT%experiments\rolling_comparison_real_expanded_v1\comparison_station_level.csv"

echo.
echo ==========================================
echo      AquaSense Demo Launcher (EXPANDED)
echo ==========================================
echo.

REM --- Start backend ---
echo [1/3] Starting FastAPI backend on port 8000 (source=real_expanded)...
start "AquaSense Backend" /D "%ROOT%" "%ROOT%.venv\Scripts\python.exe" -m uvicorn backend.main:app --port 8000

REM --- Give backend a moment to start ---
timeout /t 5 /nobreak >nul

REM --- Start frontend ---
echo [2/3] Starting React/Vite frontend...
start "AquaSense Frontend" /D "%ROOT%frontend" cmd /k npm run dev

REM --- Give Vite a moment to start ---
timeout /t 7 /nobreak >nul

REM --- Open browser ---
echo [3/3] Opening AquaSense dashboard...
start "" "http://localhost:5173"

echo.
echo ==========================================
echo AquaSense (expanded) startup commands sent.
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo Source:   real_expanded
echo.
echo Keep both server windows open.
echo ==========================================
echo.
pause

endlocal
