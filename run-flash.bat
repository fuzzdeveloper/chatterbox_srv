@echo off
cd /d "%~dp0"
.venv\Scripts\python chatterbox_flash_server.py --host 0.0.0.0 --port 8880
pause
