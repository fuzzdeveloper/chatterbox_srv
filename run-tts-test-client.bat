@echo off
cd /d "%~dp0"
cd tts-test-client
set TTS_SERVER_URL=http://127.0.0.1:8880/
python tts_client.py 7860
