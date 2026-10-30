# Chatterbox TTS Servers

Local text-to-speech (TTS) server implementations with an **OpenAI-compatible speech API**, using the Chatterbox Nano, Turbo, Full, and Flash models. A browser-based test client is included for trying voices, adjusting synthesis settings, and playing or downloading generated audio.

## Setup

On Windows, install **Python** and **Git**, and make sure both are available on your `PATH`. The setup scripts note that they were tested with Python 3.13.0. Internet access is needed to install dependencies and download model weights on first use.

From the project directory, run **one** of the following batch files:

- **Run models on CPU:** `setup-python-venv-cpu.bat`
- **Run models on an NVIDIA GPU with CUDA:** `setup-python-venv-cuda.bat` (requires a compatible GPU and NVIDIA driver; the script installs CUDA 12.6 PyTorch wheels - works with GTX1080TI and should work with newer).

Both scripts create a local `.venv` and clone/install the upstream [Perth](https://github.com/resemble-ai/Perth), [Chatterbox](https://github.com/resemble-ai/chatterbox), and [Chatterbox Flash](https://github.com/resemble-ai/chatterbox-flash) projects. They are alternative setups for the same environment, not steps to run one after the other.

The server launchers use `.venv` directly, so you do not need to activate it manually. Servers automatically select CUDA when available, otherwise CPU. To explicitly select a device, use the direct Python commands below with `--device cpu` or `--device cuda`.

## Custom voices

Create a `voices` subdirectory beside the server scripts and put your reference recordings there as **WAV files**:

```text
voices/
  narrator.wav
  another_voice.wav
```

Use recordings **at least 10–15 seconds long**, with clear, clean speech, little background noise, and no music or overlapping speakers. Choose a sample with the **enunciation and cadence you want in the synthesized speech**.

The filename without `.wav` becomes the API voice name: for example, `voices/narrator.wav` is selected as `narrator`. Custom voices are cloned and cached on first use. The `default` voice uses the built-in reference voice; unknown voice names also fall back to it. You can use a different reference directory with `--voices PATH`.

## Models and how to run them

All four batch launchers serve the API on **port 8880**, so run only one at a time unless you assign different ports manually. Wait for the server to finish downloading/loading its model and print its listening address before making requests.

### 1. Nano

**Chatterbox Nano** is the smallest option (110M parameters), aimed at CPU/on-device inference and tight memory or latency budgets. It shares Turbo's architecture, single-step speech decoder, and support for expressive tags such as `[laugh]` and `[cough]`. Can run real-time on AMD 7950x but doesn't clone voices particularly well (not with non-US accent anyway). Note: does not support exaggeration or cfgweight params from client. 

```bat
run-nano.bat
```

Nano uses the Turbo server with the `--nano` flag. To launch it explicitly on CPU:

```bat
.venv\Scripts\python chatterbox_turbo_server.py --nano --device cpu --port 8880
```

### 2. Turbo

**Chatterbox Turbo** is a streamlined 350M-parameter English model designed for low-latency speech generation. It uses a single-step speech decoder and supports expressive tags such as `[laugh]`, `[chuckle]`, and `[cough]`. Can run real-time on nVidia GTX1080TI and is far more capable than nano at voice cloning. Note: does not support exaggeration or cfgweight params from client (though there's less need for them with this than full/flash models).

```bat
run-turbo.bat
```

Direct launch, for example on CUDA:

```bat
.venv\Scripts\python chatterbox_turbo_server.py --device cuda --port 8880
```

### 3. Full

**Full** runs the original 500M-parameter English Chatterbox model. It offers **exaggeration** (emotion intensity) and **CFG weight** (guidance/pacing) controls for tuning delivery (though I find you often NEED to tweak these to get results you can 1-shot with turbo). This is the original English model, not the separate multilingual model.

```bat
run-full.bat
```

Direct launch:

```bat
.venv\Scripts\python chatterbox_full_server.py --device cuda --port 8880
```

### 4. Flash

**Chatterbox Flash** replaces the original model's autoregressive speech-token decoder with a **block-diffusion decoder**, generating token blocks rather than decoding strictly one token at a time. It reuses the original vocoder and voice encoder, and exposes temperature, exaggeration, guidance, and diffusion-step settings. Actual performance depends on your hardware and inference backend (on GTX1080TI it's not quite realtime speed). This seems to have cadence/pacing issues that others don't (often talks way too fast and this can't be fixed via param tweaking).

```bat
run-flash.bat
```

Direct launch:

```bat
.venv\Scripts\python chatterbox_flash_server.py --device cuda --port 8880
```

Flash can also run on CPU with `--device cpu`. Its `--backend` option supports `auto`, `torch`, `flashinfer`, and `mlx`; optional accelerated backends require their own dependencies and compatible hardware. The setup scripts do not explicitly install these optional backends. Flash's guidance defaults and behavior differ from Full's, so the same settings will not produce the same results.

For all servers, run the corresponding Python command with `--help` to see available options. Add `--host 0.0.0.0` to a direct launch to allow connections from other machines.

## Run the test client

1. Start one of the model servers above and leave it running.
2. In another terminal, run:

   ```bat
   run-tts-test-client.bat
   ```

3. Open **http://localhost:7860/** in your browser.
4. Select a voice, enter text, and click **Speak**. You can adjust the sliders and download the generated audio. Use the refresh button to reload the voice list after adding WAV files.

The launcher uses `python` from your `PATH` and points the client at `http://127.0.0.1:8880/`. The client uses Python's standard library to serve the UI and proxy API requests, avoiding browser CORS issues. Exaggeration and CFG sliders have no effect on Nano or Turbo.

To use a different server address or UI port, launch the client manually from the project directory (Windows Command Prompt):

```bat
set TTS_SERVER_URL=http://127.0.0.1:8881/
.venv\Scripts\python tts-test-client\tts_client.py 7860
```

For slower CPU inference, try short text first: the client currently has a 30-second upstream request timeout.

## API

The default local API base URL is **`http://127.0.0.1:8880/v1`**.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/speech` | Synthesize text and return audio. |
| `GET /v1/audio/voices` | List `default` and available custom voice names. |
| `GET /health` | Check that the server is responding. |

Example from Windows Command Prompt:

```bat
curl.exe http://127.0.0.1:8880/v1/audio/speech -H "Content-Type: application/json" -d "{\"input\":\"Hello, this is a test.\",\"voice\":\"default\",\"response_format\":\"wav\"}" --output speech.wav
```

Replace `default` with a custom voice name such as `narrator`. Responses are WAV by default; `response_format: "pcm"` returns raw signed 16-bit PCM. MP3 and other OpenAI output formats are not implemented. Each server runs one selected model; a request's `model` field does not switch models.

Optional request controls include `temperature` on all servers, plus `exaggeration` and `cfgweight` on Full and Flash. This is a speech-focused OpenAI-compatible API, not an implementation of the complete OpenAI API.

**Network access:** the batch launchers bind the servers to `0.0.0.0`, and the test client also listens on all interfaces. There is no built-in authentication; keep them on a trusted network or restrict access with your firewall.
