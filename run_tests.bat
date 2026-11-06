@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run run_api.bat first to create the environment.& pause & exit /b 1)
call ".venv\Scripts\activate.bat"
python -m pip install pytest httpx
python -m pytest tests -q
pause
