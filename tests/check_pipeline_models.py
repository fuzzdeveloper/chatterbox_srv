"""Opt-in real-model equivalence checks; not run by unittest discovery.

Run from the repository root:
  .venv/Scripts/python tests/check_pipeline_models.py --kind full
Each process loads one model; no simultaneous LLM workload is started.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

# Direct script execution puts tests/, rather than the repository, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from chatterbox_pipeline import SynthesisWorker
from chatterbox_full_timing import ModelStageProfiler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("full", "turbo", "flash"), required=True)
    p.add_argument("--nano", action="store_true", help="requires locally available Nano weights")
    args = p.parse_args()
    if args.kind == "full":
        from chatterbox_full_server import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device="cuda")
        options = [dict(temperature=.8, exaggeration=.5, cfg_weight=.5),
                   dict(temperature=.65, exaggeration=.7, cfg_weight=.2, top_p=.9, min_p=.03)]
    elif args.kind == "turbo":
        from chatterbox_turbo_server import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=args.nano)
        norm = model.norm_loudness
        model.norm_loudness = lambda wav, sr, **kw: np.asarray(norm(wav, sr, **kw), dtype=np.float32)
        options = [dict(temperature=.8, top_p=.95, top_k=1000, repetition_penalty=1.2),
                   dict(temperature=.65, top_p=.85, top_k=100, repetition_penalty=1.3)]
    else:
        from chatterbox_flash_server import ChatterboxFlashTTS, load_builtin_conds
        model = ChatterboxFlashTTS.from_pretrained(device="cuda", dtype=torch.float32)
        model.conds = load_builtin_conds(model)
        options = [dict(temperature=.6, cfg_scale=0, num_steps=3, backend="torch", use_cuda_graph=False),
                   dict(temperature=.7, cfg_scale=.8, num_steps=3, backend="torch", use_cuda_graph=False,
                        position_temperature=2.0, time_shift_tau=.15, n_cfm_timesteps=3)]
    default_conds = model.conds
    with torch.inference_mode():
        prepared = model.prepare_conditionals("voices/adam.wav")
    adam = prepared if prepared is not None else model.conds
    profiler = ModelStageProfiler(model)
    report = []
    for index, conds in enumerate((default_conds, adam)):
        settings = options[index]
        text = "Hello there, this is a pipeline test." if index == 0 else "The next sentence uses a different voice."
        original_decoder = model.s3gen.inference
        captured = []

        def capture_tokens(*positional, **named):
            captured.append(named["speech_tokens"].detach().cpu())
            return original_decoder(*positional, **named)

        def fresh_conds():
            return type(conds)(replace(conds.t3), dict(conds.gen))

        def generate(worker):
            # Requests are awaited individually here: seeding cannot race a
            # decoder. Concurrent scheduling is tested separately over HTTP.
            torch.manual_seed(1234 + index)
            model.conds = fresh_conds()
            return "default" if index == 0 else "adam", worker.generate(text, **settings)

        model.s3gen.inference = capture_tokens
        try:
            serial = SynthesisWorker(model, args.kind, profiler=profiler)
            try:
                expected = serial.submit(generate, serial).result(timeout=180)
            finally:
                serial.shutdown()
        finally:
            model.s3gen.inference = original_decoder
        pipeline = SynthesisWorker(model, args.kind, pipeline=True, profiler=profiler)
        try:
            # Capture adapter output independently from the reference hook.
            original_generate = pipeline.generate
            token_result = []

            def capture_pipeline(*positional, **named):
                job = original_generate(*positional, **named)
                token_result.append(job.kwargs["speech_tokens"].detach().cpu())
                return job

            pipeline.generate = capture_pipeline
            actual = pipeline.submit(generate, pipeline).result(timeout=180)
        finally:
            pipeline.shutdown()
        assert torch.equal(captured[0], token_result[0]), "Speech token mismatch"
        a, b = expected.wav.reshape(-1), actual.wav.reshape(-1)
        assert a.shape == b.shape, (a.shape, b.shape)
        delta = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"Waveform mismatch, max delta {delta}"
        assert torch.isfinite(b).all() and torch.any(b != 0)
        for stage in ("t3", "flow", "vocoder"):
            assert stage in actual.stages, stage
        result = dict(kind=args.kind, nano=args.nano, voice=actual.voice, samples=b.numel(),
                      tokens=captured[0].numel(), max_waveform_delta=delta,
                      serial_secs=expected.synthesis_secs, pipeline_secs=actual.synthesis_secs,
                      settings=settings, pipeline_stages=actual.stages)
        report.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "pipeline_stages"}), flush=True)
    dest = Path("benchmarks/pipeline_servers")
    dest.mkdir(parents=True, exist_ok=True)
    label = "nano" if args.nano else args.kind
    (dest / f"{label}_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    profiler.close()


if __name__ == "__main__":
    main()
