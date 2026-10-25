@echo off
setlocal

REM ============================================================
REM AquaSense - One-click demo launcher
REM Starts FastAPI backend + Vite frontend, then opens browser.
REM ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "AQUASENSE_SOURCE=real"

echo.
echo ==========================================
echo        AquaSense Demo Launcher
echo ==========================================
echo.

REM --- Start backend ---
echo [1/3] Starting FastAPI backend on port 8000...
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
echo AquaSense startup commands sent.
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Keep both server windows open.
echo ==========================================
echo.
pause

endlocal
