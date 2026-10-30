@echo off
echo Setting up venv suitable for running chatterbox on CUDA
echo Note: this was tested against Python 3.13.0
cd /d "%~dp0"

python -m venv .venv
IF %ERRORLEVEL% NEQ 0 (
    echo Error running python -m venv - is python installed?
    exit /b %ERRORLEVEL%
)

rem lets just install the chatterbox from git, let them put whatever they think they need
git clone https://github.com/resemble-ai/Perth.git
.venv\scripts\pip install -e Perth/
git clone https://github.com/resemble-ai/chatterbox.git
.venv\scripts\pip install -e chatterbox/
git clone https://github.com/resemble-ai/chatterbox-flash.git
.venv\scripts\pip install -e chatterbox-flash/

rem now we replace the torch libs with cuda ones while listening to python whinge about them being "incompatible"
.venv\Scripts\pip uninstall touch torchaudio 
.venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

pause
