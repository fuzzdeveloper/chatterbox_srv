#!/usr/bin/env python3
"""
Minimal OpenAI-compatible text-to-speech server using Chatterbox Turbo (nano variant) on CPU.

    .env\\Scripts\\python chatterbox_server.py --host 0.0.0.0 --port 8880

Endpoints:
    POST /v1/audio/speech   {"input": "...", "voice": "adam", "response_format": "wav"}
    GET  /v1/audio/voices   list of voice names ("default" + *.wav files in ./voices)
    GET  /health

Voices: each <name>.wav in the voices directory (must be > 5 s long) is cloned on first use and
cached. "default", or any unknown voice, uses the model's built-in voice.
Only WAV output is produced ("pcm" returns raw 16-bit mono samples). "speed" is ignored.
Sampling is set server-wide via --temperature, --top-p, --top-k, --repetition-penalty.
(exaggeration / cfg_weight are not supported by the Turbo/nano model.)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE = "default"


def parse_args():
    p = argparse.ArgumentParser(description="OpenAI-compatible Chatterbox nano TTS server")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (0.0.0.0 = all)")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda", "mps"], help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--nano", type=bool, default=False)
    p.add_argument("--voices", default=os.path.join(HERE, "voices"), help="directory of voice .wav files")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="default voice name")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (higher = more varied)")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff (lower = steadier)")
    p.add_argument("--top-k", type=int, default=1000, help="top-k sampling cutoff (lower = steadier)")
    p.add_argument("--repetition-penalty", type=float, default=1.2, help="penalise repeated tokens/stutters")
    return p.parse_args()


def list_voices(voices_dir):
    """Map voice name -> wav path, rescanned on each call so new files are picked up."""
    if not os.path.isdir(voices_dir):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(voices_dir, f)
        for f in sorted(os.listdir(voices_dir))
        if f.lower().endswith(".wav")
    }


def to_pcm16(samples):
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def to_wav(samples, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_pcm16(samples))
    return buf.getvalue()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is False")
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=args.nano)
    # Workaround: with NumPy 2, norm_loudness() upcasts the reference wav to float64, which then
    # breaks voice cloning ("expected scalar type Double but found Float"). Cast back to float32.
    _norm = model.norm_loudness
    model.norm_loudness = lambda wav, sr, **kw: np.asarray(_norm(wav, sr, **kw), dtype=np.float32)
    default_conds = model.conds  # built-in voice; prepare_conditionals() overwrites model.conds
    cache = {}  # voice name -> (wav mtime, Conditionals)

    def conds_for(voice):
        """Return (resolved voice name, Conditionals). Only called by the synthesis worker."""
        path = list_voices(args.voices).get(voice)
        if voice == DEFAULT_VOICE or path is None:
            return DEFAULT_VOICE, default_conds
        mtime = os.path.getmtime(path)
        cached = cache.get(voice)
        if cached and cached[0] == mtime:
            return voice, cached[1]
        try:
            model.prepare_conditionals(path)
        except Exception as e:  # noqa: BLE001  (e.g. clip shorter than 5 s)
            print(f"failed to clone voice '{voice}' ({e}); using default")
            return DEFAULT_VOICE, default_conds
        cache[voice] = (mtime, model.conds)
        print(f"cloned voice '{voice}' from {path}")
        return voice, model.conds

    @torch.inference_mode()
    def synthesize(text, voice, temperature):
        # Enter inference mode in this thread: it is thread-local. Turbo's inner
        # inference methods do not cover its CPU watermarker, which otherwise
        # builds unnecessary autograd graphs and causes substantial RAM growth.
        voice, conds = conds_for(voice)
        model.conds = conds
        wav = model.generate(text, temperature=temperature, top_p=args.top_p, top_k=args.top_k,
                             repetition_penalty=args.repetition_penalty)
        return voice, wav

    # Serialize model access on a persistent thread, rather than on each fresh
    # HTTP thread, so native CPU inference buffers can be reused between requests.
    synthesis_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status, body, ctype):
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, obj):
            self.send_body(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"status": "ok"})
            if self.path in ("/v1/audio/voices", "/v1/voices"):
                return self.send_json(200, {"voices": [DEFAULT_VOICE] + list(list_voices(args.voices))})
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/audio/speech", "/audio/speech"):
                return self.send_json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            except ValueError:
                return self.send_json(400, {"error": "invalid JSON"})
            text = str(req.get("input") or "").strip()
            if not text:
                return self.send_json(400, {"error": "empty input"})
            voice = str(req.get("voice") or args.voice)
            try:  # optional per-request overrides; fall back to command-line values
                temperature = float(args.temperature if req.get("temperature") is None else req["temperature"])
            except (TypeError, ValueError):
                return self.send_json(400, {"error": "temperature, exaggeration and cfgweight must be numbers"})
            started = time.perf_counter()
            try:
                voice, wav = synthesis_worker.submit(synthesize, text, voice, temperature).result()
            except Exception as e:  # noqa: BLE001
                return self.send_json(500, {"error": str(e)})
            samples = wav.squeeze(0).numpy()
            rate = model.sr
            took = time.perf_counter() - started
            audio_secs = len(samples) / rate
            print(f"[{voice}] {len(text):4d} chars -> {audio_secs:5.1f}s audio in {took:4.2f}s "
                  f"(RTF {took / max(audio_secs, 0.01):.2f})")
            if req.get("response_format") == "pcm":
                return self.send_body(200, to_pcm16(samples), "audio/pcm")
            self.send_body(200, to_wav(samples, rate), "audio/wav")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    n = len(list_voices(args.voices))
    print(f"Chatterbox nano TTS listening on http://{args.host}:{args.port}/v1/audio/speech "
          f"(voices: default + {n} from {args.voices})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        synthesis_worker.shutdown(wait=True)


if __name__ == "__main__":
    main()
