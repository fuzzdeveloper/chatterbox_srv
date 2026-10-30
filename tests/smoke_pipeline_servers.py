"""Opt-in HTTP checks for all three servers, serial and pipelined.

Run from the repository root with .venv/Scripts/python tests/smoke_pipeline_servers.py.
Loads one CUDA model at a time; uses cached weights and cloned reference voices.
"""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "benchmarks" / "pipeline_servers"


def request(base, payload):
    req = urllib.request.Request(base + "/v1/audio/speech", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    began = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        body = response.read()
        mime = response.headers.get("Content-Type")
    if payload.get("response_format") == "pcm":
        assert mime == "audio/pcm" and len(body) > 0 and len(body) % 2 == 0
        assert any(body)
        duration = len(body) / 2 / 24000
    else:
        assert mime == "audio/wav"
        with wave.open(io.BytesIO(body), "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
            assert wav.getframerate() == 24000 and wav.getnframes() > 0
            assert any(wav.readframes(wav.getnframes()))
            duration = wav.getnframes() / wav.getframerate()
    return dict(voice=payload["voice"], audio_secs=duration, wall_secs=time.perf_counter() - began,
                format=payload.get("response_format", "wav"))


def check(kind, pipeline):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    label = f"{kind}_{'pipeline' if pipeline else 'serial'}"
    log_path = DEST / f"{label}_http.log"
    command = [sys.executable, "-u", f"chatterbox_{kind}_server.py", "--host", "127.0.0.1",
               "--port", str(port), "--device", "cuda", "--profile-stages"]
    if pipeline:
        command.append("--pipeline")
    if kind == "flash":
        # Portable to the GTX 1080 Ti; FlashInfer/BF16 are not tested here.
        command += ["--dtype", "fp32", "--backend", "torch", "--num-steps", "3"]
    base = f"http://127.0.0.1:{port}"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        assert json.loads(response.read()) == {"status": "ok"}
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Startup: {log_path}")
                    time.sleep(.25)
            with urllib.request.urlopen(base + "/v1/audio/voices", timeout=5) as response:
                assert "adam" in json.loads(response.read())["voices"]
            for invalid in ({"input": ""}, {"input": "hello", "temperature": "invalid"}):
                try:
                    request(base, invalid)
                    raise AssertionError("Expected HTTP 400")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
            warm = request(base, dict(input="This is the first sentence.", voice="default"))
            voices = ("default", "adam", "af_heart", "adam", "missing-voice", "af_heart")
            texts = ("Hello there, how are you today?", "The robot walked across the room.",
                     "This sentence uses another speaker.", "A warm voice is ready to speak again.",
                     "An unknown speaker uses the default voice.", "That is the final sentence in the queue.")
            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(request, base, dict(
                    input=text, voice=voice, temperature=.65 if i % 2 else .8,
                    exaggeration=.3 if i % 2 else .7, cfgweight=.2 if i % 2 else .8,
                    response_format="pcm" if i % 2 else "wav"))
                           for i, (text, voice) in enumerate(zip(texts, voices))]
                results = [future.result(timeout=240) for future in futures]
            wall = time.perf_counter() - began
            # A new request after a drained queue must also work as an idle burst.
            final = request(base, dict(input="We are ready for a new conversation.", voice="adam"))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    contents = log_path.read_text(encoding="utf-8")
    assert contents.count("  timing:") == 8, log_path
    for stage in ("queue=", "synthesis=", "t3=", "flow=", "vocoder=", "encode=", "voice_prepare="):
        assert stage in contents, (log_path, stage)
    assert "Traceback" not in contents, log_path
    assert ("CUDA sentence pipeline enabled" in contents) == pipeline
    result = dict(kind=kind, pipeline=pipeline, concurrent_batch_secs=wall,
                  first=warm, concurrent_requests=results, after_idle=final)
    (DEST / f"{label}_http.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS {label}: first request, six concurrent requests with cold/cached/default voices, "
          f"post-idle request, WAV/PCM, parameter overrides, validation errors", flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for kind in ("full", "turbo", "flash"):
        for pipeline in (False, True):
            check(kind, pipeline)


if __name__ == "__main__":
    main()
