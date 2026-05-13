@echo off
cd /d %~dp0

echo Starting server...

start "" http://127.0.0.1:8000

venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000

pause