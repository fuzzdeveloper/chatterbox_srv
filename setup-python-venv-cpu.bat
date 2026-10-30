@echo off
echo Setting up venv suitable for running chatterbox on CPU
echo Note: this was tested against Python 3.13.0
cd /d "%~dp0"

python -m venv .venv
IF %ERRORLEVEL% NEQ 0 (
    echo Error running python -m venv - is python installed?
    exit /b %ERRORLEVEL%
)

rem cpu specific torch libs
.venv\scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

rem now we install deps that are ACTUALLY NEEDED while trying not to screw up the above or bring in a pile of cuda stuff
.venv\scripts\pip install numpy
.venv\scripts\pip install librosa
.venv\scripts\pip install huggingface_hub
.venv\scripts\pip install safetensors
.venv\scripts\pip install transformers
.venv\scripts\pip install einops
.venv\scripts\pip install s3tokenizer --no-deps
.venv\scripts\pip install onnx
.venv\scripts\pip install pre-commit
.venv\scripts\pip install omegaconf
.venv\scripts\pip install conformer
.venv\scripts\pip install diffusers --no-deps
.venv\scripts\pip install pillow
.venv\scripts\pip install pyloudnorm

rem these come from github because pip ones are out of date
git clone https://github.com/resemble-ai/Perth.git
.venv\scripts\pip install --no-deps -e Perth/
git clone https://github.com/resemble-ai/chatterbox.git
.venv\scripts\pip install --no-deps -e chatterbox/
git clone https://github.com/resemble-ai/chatterbox-flash.git
.venv\scripts\pip install --no-deps -e chatterbox-flash/
