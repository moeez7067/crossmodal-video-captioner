@echo off
REM ============================================================
REM  Start the frontend (opens http://localhost:3000)
REM  Double-click this file, or run it from a terminal.
REM  Make sure the backend (run_backend.cmd) is running too.
REM ============================================================
cd /d "D:\nlp project 2\multimodal-video-intelligence-main\frontend"
echo Starting frontend on http://localhost:3000 ...
echo.
call npm start
echo.
echo Frontend stopped. Press any key to close this window.
pause >nul
