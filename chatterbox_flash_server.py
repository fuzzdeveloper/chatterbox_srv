#!/usr/bin/env python3
"""
Minimal OpenAI-compatible text-to-speech server using Chatterbox-Flash (block-diffusion T3).

    .venv\\Scripts\\python chatterbox_flash_server.py --host 0.0.0.0 --port 8881 --device cuda

Endpoints:
    POST /v1/audio/speech   {"input": "...", "voice": "adam", "response_format": "wav",
                             "temperature": 0.6, "exaggeration": 0.5, "cfgweight": 1.0}
                            (last three optional; "cfg_weight" / "cfg_scale" also accepted)
    GET  /v1/audio/voices   list of voice names ("default" + *.wav files in ./voices)
    GET  /health

Voices: each <name>.wav in the voices directory (must be > 5 s long) is cloned on first use and
cached. "default", or any unknown voice, uses the built-in voice of the full Chatterbox model
(conds.pt from ResembleAI/chatterbox; Chatterbox-Flash itself ships no built-in voice).
Only WAV output is produced ("pcm" returns raw 16-bit mono samples). "speed" is ignored.

Notes on Chatterbox-Flash parameters vs. the full model:
  * cfg_weight is called cfg_scale. Same formula (cond + w * (cond - uncond)), but the default
    is 1.0, and 0 is valid: it disables CFG entirely (halves the forward batch -> faster).
  * exaggeration is the T3 "emotion_adv" conditioning value; it's applied per request here.
  * temperature default is 0.6. Additional knobs: --num-steps, --position-temperature,
    --time-shift-tau, --n-cfm-timesteps (server-wide).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import io
import json
import os
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from chatterbox_flash import ChatterboxFlashTTS, Conditionals

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE = "default"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser(description="OpenAI-compatible Chatterbox-Flash TTS server")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (0.0.0.0 = all)")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda", "mps"], help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--dtype", choices=list(DTYPES), default=None,
                   help="T3 compute dtype (default: bf16 on cuda, fp16 otherwise)")
    p.add_argument("--backend", default="auto", choices=["auto", "flashinfer", "torch", "mlx"],
                   help="T3 inference engine")
    p.add_argument("--voices", default=os.path.join(HERE, "voices"), help="directory of voice .wav files")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="default voice name")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument("--exaggeration", type=float, default=0.5, help="emotion intensity (0.5 = neutral)")
    p.add_argument("--cfg-weight", type=float, default=1.0,
                   help="classifier-free guidance scale (Flash's cfg_scale; 0 = CFG off)")
    p.add_argument("--temperature", type=float, default=0.6, help="sampling temperature (higher = more varied)")
    p.add_argument("--num-steps", type=int, default=10, help="max denoising steps per block")
    p.add_argument("--position-temperature", type=float, default=5.0,
                   help="Gumbel noise on unmask-position ranking (0 = deterministic)")
    p.add_argument("--time-shift-tau", type=float, default=0.1, help="early-decoding aggressiveness (0 = off)")
    p.add_argument("--n-cfm-timesteps", type=int, default=2, help="S3Gen vocoder flow-matching steps")
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


def load_builtin_conds(model):
    """Borrow the full Chatterbox model's built-in voice (same Conditionals layout)."""
    try:
        path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="conds.pt")
        conds = Conditionals.load(path, map_location=model.device).to(model.device)
    except Exception as e:  # noqa: BLE001
        print(f"could not load built-in voice ({e}); 'default' voice unavailable")
        return None
    t3_dtype = next(model.t3.parameters()).dtype
    conds.t3.to(device=model.device, dtype=t3_dtype)  # long tensors (prompt tokens) keep their dtype
    return conds


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is False")
    dtype = DTYPES[args.dtype or ("bf16" if args.device == "cuda" else "fp16")]
    use_cuda_graph = args.device == "cuda"

    model = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=args.device, dtype=dtype)
    print(f"model loaded on {model.device} ({dtype})")
    t3_dtype = next(model.t3.parameters()).dtype
    default_conds = load_builtin_conds(model)
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
            conds = model.prepare_conditionals(path, exaggeration=args.exaggeration)
        except Exception as e:  # noqa: BLE001  (e.g. clip shorter than 5 s)
            print(f"failed to clone voice '{voice}' ({e}); using default")
            return DEFAULT_VOICE, default_conds
        cache[voice] = (mtime, conds)
        print(f"cloned voice '{voice}' from {path}")
        return voice, conds

    def with_exaggeration(conds, exaggeration):
        """Copy of conds with emotion_adv set (cached conds are left untouched)."""
        emotion = torch.full((1, 1, 1), exaggeration, dtype=t3_dtype, device=model.device)
        return Conditionals(dataclasses.replace(conds.t3, emotion_adv=emotion), conds.gen)

    @torch.inference_mode()
    def synthesize(text, voice, temperature, exaggeration, cfg_weight):
        # Inference mode is thread-local; cover voice preparation and the entire
        # generation pipeline, including any CPU watermarking, in this worker.
        voice, conds = conds_for(voice)
        if conds is None:
            return voice, None
        model.conds = with_exaggeration(conds, exaggeration)
        wav = model.generate(
            text,
            temperature=temperature,
            cfg_scale=max(cfg_weight, 0.0),
            num_steps=args.num_steps,
            position_temperature=args.position_temperature,
            time_shift_tau=args.time_shift_tau,
            n_cfm_timesteps=args.n_cfm_timesteps,
            use_cuda_graph=use_cuda_graph,
            backend=args.backend,
        )
        return voice, wav

    # Serialize model access on a persistent thread so native CPU inference
    # buffers can be reused instead of being allocated on fresh HTTP threads.
    synthesis_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def opt_float(req, keys, default):
        for k in keys:
            if req.get(k) is not None:
                return float(req[k])
        return float(default)

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
                temperature = opt_float(req, ("temperature",), args.temperature)
                exaggeration = opt_float(req, ("exaggeration",), args.exaggeration)
                cfg_weight = opt_float(req, ("cfgweight", "cfg_weight", "cfg_scale"), args.cfg_weight)
            except (TypeError, ValueError):
                return self.send_json(400, {"error": "temperature, exaggeration and cfgweight must be numbers"})
            started = time.perf_counter()
            try:
                voice, wav = synthesis_worker.submit(
                    synthesize, text, voice, temperature, exaggeration, cfg_weight
                ).result()
            except Exception as e:  # noqa: BLE001
                return self.send_json(500, {"error": str(e)})
            if wav is None:
                return self.send_json(400, {"error": "no built-in voice available; use a cloned voice"})
            samples = wav.reshape(-1).float().numpy()
            rate = model.sr
            took = time.perf_counter() - started
            audio_secs = len(samples) / rate
            print(f"[{voice} t={temperature:g} ex={exaggeration:g} cfg={cfg_weight:g}] {len(text):4d} chars -> "
                  f"{audio_secs:5.1f}s audio in {took:4.2f}s (RTF {took / max(audio_secs, 0.01):.2f})")
            if req.get("response_format") == "pcm":
                return self.send_body(200, to_pcm16(samples), "audio/pcm")
            self.send_body(200, to_wav(samples, rate), "audio/wav")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    n = len(list_voices(args.voices))
    print(f"Chatterbox-Flash TTS ({args.device}) listening on http://{args.host}:{args.port}/v1/audio/speech "
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
