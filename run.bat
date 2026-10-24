@echo off
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting Magic Eraser (Streamlit)...
echo Open http://localhost:8501 in your browser.
"%PY%" -u -m streamlit run streamlit_app.py --server.port 8501

endlocal
