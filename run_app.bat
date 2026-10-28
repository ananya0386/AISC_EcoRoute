@echo off
REM Starts the EcoRoute Streamlit app at http://localhost:8501
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py
pause
