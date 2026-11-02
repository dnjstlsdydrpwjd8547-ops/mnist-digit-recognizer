@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
where python >nul 2>nul || set "PY=%USERPROFILE%\anaconda3\python.exe"
echo [1/2] Installing packages...
"%PY%" -m pip install -q -r requirements.txt
echo [2/2] Server: http://127.0.0.1:8000  (stop: Ctrl+C)
start "" http://127.0.0.1:8000
"%PY%" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
