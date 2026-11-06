@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 (echo [ERROR] Could not create virtual environment. Install Python 3.11 or newer.& pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
echo [INFO] Installing API dependencies. First run may take a few minutes...
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
if errorlevel 1 (echo [ERROR] Dependency installation failed. See output above.& pause & exit /b 1)
echo [INFO] Starting Flight Delay Prediction API...
echo [INFO] Open http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
