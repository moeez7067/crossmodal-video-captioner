@echo off
REM ============================================================
REM  Start the backend API (FastAPI on http://localhost:8000)
REM  Double-click this file, or run it from a terminal.
REM ============================================================
cd /d "D:\nlp project 2\multimodal-video-intelligence-main"
echo Starting backend API on http://localhost:8000 ...
echo (First video you process will download ~1.4 GB of models - one time.)
echo.
".\venv\Scripts\python.exe" run.py
echo.
echo Backend stopped. Press any key to close this window.
pause >nul
