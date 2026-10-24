@echo off
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting DeepFill v2 API (FastAPI)...
echo Swagger docs: http://127.0.0.1:8000/docs
"%PY%" -u app.py

endlocal
