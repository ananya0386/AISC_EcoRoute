@echo off
REM One-time setup: creates a virtual environment in .venv and installs dependencies.
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment. Is Python 3.10+ installed and on PATH?
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Setup complete. Start the app with run_app.bat
pause
