#!/usr/bin/env python3
"""
Minimal OpenAI-compatible text-to-speech server using the full Chatterbox model (CPU or CUDA).

    .env\\Scripts\\python chatterbox_full_server.py --host 0.0.0.0 --port 8880 --device cuda

Endpoints:
    POST /v1/audio/speech   {"input": "...", "voice": "adam", "response_format": "wav",
                             "exaggeration": 0.5, "cfgweight": 0.5}   (last two optional)
    GET  /v1/audio/voices   list of voice names ("default" + *.wav files in ./voices)
    GET  /health

Voices: each <name>.wav in the voices directory (must be > 5 s long) is cloned on first use and
cached. "default", or any unknown voice, uses the model's built-in voice.
Only WAV output is produced ("pcm" returns raw 16-bit mono samples). "speed" is ignored.
Generation is set server-wide via --exaggeration, --cfg-weight, --temperature, --min-p, --top-p,
--repetition-penalty.
"""

import argparse
import io
import json
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS
import chatterbox.tts as _cb_tts
from chatterbox.models.s3tokenizer import SPEECH_VOCAB_SIZE


def _drop_invalid_tokens(x):
    """Safer replacement for chatterbox's drop_invalid_tokens().

    The original crashes with "only integer tensors of a single element can be converted to an
    index" when the model samples the start-of-speech token more than once (seen mostly on CUDA).
    Here: cut at the first EOS, then drop every special token (SOS/EOS/anything out of vocab).
    """
    x = x.reshape(-1)
    eos = (x == SPEECH_VOCAB_SIZE + 1).nonzero(as_tuple=True)[0]
    if len(eos):
        x = x[: eos[0].item()]
    return x[x < SPEECH_VOCAB_SIZE]


_cb_tts.drop_invalid_tokens = _drop_invalid_tokens  # tts.py looks this name up at call time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE = "default"


def parse_args():
    p = argparse.ArgumentParser(description="OpenAI-compatible Chatterbox TTS server")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (0.0.0.0 = all)")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda", "mps"], help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--voices", default=os.path.join(HERE, "voices"), help="directory of voice .wav files")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="default voice name")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument("--exaggeration", type=float, default=0.5, help="emotion intensity (0.5 = neutral)")
    p.add_argument("--cfg-weight", type=float, default=0.5, help="classifier-free guidance weight / pacing")
    p.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (higher = more varied)")
    p.add_argument("--min-p", type=float, default=0.05, help="min-p sampling cutoff")
    p.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling cutoff (lower = steadier)")
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
    model = ChatterboxTTS.from_pretrained(device=args.device)
    print(f"model loaded on {model.device}")
    default_conds = model.conds  # built-in voice; prepare_conditionals() overwrites model.conds
    cache = {}  # voice name -> (wav mtime, Conditionals)
    lock = threading.Lock()  # one synthesis at a time (model state is shared)

    def conds_for(voice):
        """Return (resolved voice name, Conditionals). Caller must hold the lock."""
        path = list_voices(args.voices).get(voice)
        if voice == DEFAULT_VOICE or path is None:
            return DEFAULT_VOICE, default_conds
        mtime = os.path.getmtime(path)
        cached = cache.get(voice)
        if cached and cached[0] == mtime:
            return voice, cached[1]
        try:
            model.prepare_conditionals(path, exaggeration=args.exaggeration)
        except Exception as e:  # noqa: BLE001  (e.g. clip shorter than 5 s)
            print(f"failed to clone voice '{voice}' ({e}); using default")
            return DEFAULT_VOICE, default_conds
        cache[voice] = (mtime, model.conds)
        print(f"cloned voice '{voice}' from {path}")
        return voice, model.conds

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
                exaggeration = float(args.exaggeration if req.get("exaggeration") is None else req["exaggeration"])
                cfg_weight = float(args.cfg_weight if req.get("cfgweight") is None else req["cfgweight"])
            except (TypeError, ValueError):
                return self.send_json(400, {"error": "temperature, exaggeration and cfgweight must be numbers"})
            started = time.perf_counter()
            try:
                with lock:
                    voice, conds = conds_for(voice)
                    model.conds = conds
                    wav = model.generate(
                        text,
                        exaggeration=exaggeration,
                        cfg_weight=max(cfg_weight, 1e-6),  # 0 crashes chatterbox (t3 always runs CFG); 1e-6 ~= off
                        temperature=temperature,
                        min_p=args.min_p,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    )
            except Exception as e:  # noqa: BLE001
                return self.send_json(500, {"error": str(e)})
            samples = wav.squeeze(0).numpy()
            rate = model.sr
            took = time.perf_counter() - started
            audio_secs = len(samples) / rate
            print(f"[{voice} ex={exaggeration:g} cfg={cfg_weight:g}] {len(text):4d} chars -> {audio_secs:5.1f}s audio in {took:4.2f}s "
                  f"(RTF {took / max(audio_secs, 0.01):.2f})")
            if req.get("response_format") == "pcm":
                return self.send_body(200, to_pcm16(samples), "audio/pcm")
            self.send_body(200, to_wav(samples, rate), "audio/wav")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    n = len(list_voices(args.voices))
    print(f"Chatterbox TTS ({args.device}) listening on http://{args.host}:{args.port}/v1/audio/speech "
          f"(voices: default + {n} from {args.voices})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
