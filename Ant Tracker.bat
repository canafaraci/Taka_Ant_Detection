@echo off
rem Double-click to open the ant tracking app (Windows).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The environment is not set up yet. Run setup_windows.ps1 first ^(see README.md^).
  pause
  exit /b 1
)
".venv\Scripts\python.exe" FINAL\pipeline.py app
if errorlevel 1 pause
