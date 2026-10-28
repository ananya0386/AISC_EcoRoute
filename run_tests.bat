@echo off
REM Runs the unit test suite.
cd /d "%~dp0"
.venv\Scripts\python.exe -m pytest -v
pause
