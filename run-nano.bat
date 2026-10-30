@echo off
cd /d "%~dp0"
.venv\Scripts\python chatterbox_turbo_server.py --host 0.0.0.0 --port 8880 --nano
pause
