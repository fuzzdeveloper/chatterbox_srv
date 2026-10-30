#!/usr/bin/env python3
r"""Compare serial full Chatterbox with T3/S3Gen overlap on ONE CUDA GPU.

Example:
  .venv\Scripts\python benchmark_full_pipeline.py --voice adam --rounds 3

This standalone experiment does not change server scheduling or model sources.
Serial and pipeline modes use identical T3 seeds, checked token hashes,
persistent workers, and the same split-generation code. Only scheduling changes.
Use --include-original to also check the unsplit server baseline. A warm-up
also checks the split code against the original model.generate waveform.
CUDA events are asynchronous: no synchronization between measured sub-stages.
All sentences are available at time zero; completion times include queueing.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the server's invalid-token fix, so the benchmark matches the server.
from chatterbox_full_server import _drop_invalid_tokens, to_wav
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox_full_timing import ModelStageProfiler


TEXTS = [
    "Hello there! I hope you are having a wonderful day.",
    "The little robot looked out of the window and wondered where all the birds had gone.",
    "That is a good question. Let me think about it for a moment.",
    "We can prepare the next sentence while the previous one is being converted into audio.",
]


@contextmanager
def independent_sampling():
    """Keep T3 sampling independent of concurrent decoder RNG use.

    This process-local patch is ONLY for the standalone benchmark. All other
    torch.multinomial callers are unchanged. No seed resets occur concurrently.
    """
    original = torch.multinomial
    local = threading.local()

    def sample(*args, **kwargs):
        generator = getattr(local, "generator", None)
        if generator is not None and "generator" not in kwargs:
            kwargs["generator"] = generator
        return original(*args, **kwargs)

    torch.multinomial = sample
    try:
        yield local
    finally:
        torch.multinomial = original


def generate_tokens(model, text, args):
    """Token half of local chatterbox/tts.py::generate, with the server fix.

    Keep defaults, CFG batching, conditioning, padding, and filtering identical
    to upstream. Checked against model.generate before any measurements.
    """
    if args.exaggeration != model.conds.t3.emotion_adv[0, 0, 0]:
        cond = model.conds.t3
        model.conds.t3 = T3Cond(
            speaker_emb=cond.speaker_emb,
            cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
            emotion_adv=args.exaggeration * torch.ones(1, 1, 1),
        ).to(device=model.device)
    text_tokens = model.tokenizer.text_to_tokens(punc_norm(text)).to(model.device)
    cfg = max(args.cfg_weight, 1e-6)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    text_tokens = F.pad(text_tokens, (1, 0), value=model.t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=model.t3.hp.stop_text_token)
    tokens = model.t3.inference(
        t3_cond=model.conds.t3, text_tokens=text_tokens, max_new_tokens=1000,
        temperature=args.temperature, cfg_weight=cfg, repetition_penalty=1.2,
        min_p=0.05, top_p=1.0,
    )[0]
    tokens = _drop_invalid_tokens(tokens)
    return tokens[tokens < 6561].to(model.device)


def validate_split(model, args, sampling):
    text = "This is a short warm up sentence."
    kwargs = dict(exaggeration=args.exaggeration, cfg_weight=max(args.cfg_weight, 1e-6),
                  temperature=args.temperature)
    captured = []
    original_decode = model.s3gen.inference

    def capture_tokens(*positional, **named):
        captured.append(named["speech_tokens"].detach().cpu())
        return original_decode(*positional, **named)

    torch.manual_seed(args.seed)
    sampling.generator = torch.Generator(device=model.device).manual_seed(args.seed)
    model.s3gen.inference = capture_tokens
    try:
        with torch.inference_mode():
            original = model.generate(text, **kwargs)
    finally:
        model.s3gen.inference = original_decode
    torch.manual_seed(args.seed)
    sampling.generator = torch.Generator(device=model.device).manual_seed(args.seed)
    with torch.inference_mode():
        tokens = generate_tokens(model, text, args)
        split, _ = model.s3gen.inference(speech_tokens=tokens, ref_dict=dict(model.conds.gen))
        split = split.detach().cpu()
    sampling.generator = None
    if not torch.equal(captured[0], tokens.cpu()):
        raise RuntimeError("Split generation tokens differ from model.generate; aborting benchmark")
    # CUDA convolution/reduction roundoff can vary by a few float32 ULPs.
    # Absolute tolerance is below one 16-bit PCM quantization step (1/32767).
    if original.shape != split.shape or not torch.allclose(original, split, atol=1e-5, rtol=1e-5):
        delta = (original - split).abs().max().item() if original.shape == split.shape else None
        raise RuntimeError(f"Split waveform differs from model.generate: shapes "
                           f"{tuple(original.shape)} / {tuple(split.shape)}, max delta={delta}")
    print(f"Split-generation validation passed ({original.numel() / model.sr:.2f}s audio)", flush=True)


def run_round(model, profiler, workers, streams, sampling, args, texts, mode, round_no):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    origin = torch.cuda.Event(enable_timing=True)
    origin.record()
    for stream in streams:
        stream.wait_event(origin)
    started = time.perf_counter()

    @torch.inference_mode()
    def produce(index, text):
        sampling.generator = torch.Generator(device=model.device).manual_seed(args.seed + index)
        try:
            with torch.cuda.stream(streams[0]), profiler.capture() as trace:
                with trace.measure("tokens"):
                    tokens = generate_tokens(model, text, args)
                ready = torch.cuda.Event()
                ready.record()
                return tokens, dict(model.conds.gen), ready, trace
        finally:
            sampling.generator = None

    def summarize(index, wav, tokens, stages, completed):
        token_cpu = tokens.cpu().numpy()
        samples = wav.squeeze(0).numpy()
        if not np.isfinite(samples).all() or not np.any(samples):
            raise RuntimeError("Non-finite or silent waveform")
        if args.save_audio and round_no == 0:
            dest = Path(args.output).parent / "audio"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"{mode}_{index:02}.wav").write_bytes(to_wav(samples, model.sr))
        return dict(index=index, text=texts[index], tokens=len(token_cpu),
                    token_hash=hashlib.sha256(token_cpu.tobytes()).hexdigest(),
                    audio_secs=len(samples) / model.sr, completed_secs=completed,
                    peak_amplitude=float(np.max(np.abs(samples))), stages=stages)

    @torch.inference_mode()
    def original_generate(index, text):
        # Control baseline: the unsplit model.generate on one persistent worker,
        # just like the server. Capture its final tokens to verify equal work.
        original_decode = model.s3gen.inference
        captured = []

        def capture_tokens(*positional, **named):
            captured.append(named["speech_tokens"])
            return original_decode(*positional, **named)

        sampling.generator = torch.Generator(device=model.device).manual_seed(args.seed + index)
        model.s3gen.inference = capture_tokens
        try:
            with torch.cuda.stream(streams[0]), profiler.capture() as trace:
                wav = model.generate(text, temperature=args.temperature,
                                     exaggeration=args.exaggeration,
                                     cfg_weight=max(args.cfg_weight, 1e-6))
                completed = time.perf_counter() - started
            return summarize(index, wav, captured[0], trace.resolve(origin), completed)
        finally:
            model.s3gen.inference = original_decode
            sampling.generator = None

    @torch.inference_mode()
    def decode(index, job):
        tokens, refs, ready, token_trace = job
        with torch.cuda.stream(streams[1]), profiler.capture() as trace:
            streams[1].wait_event(ready)
            # Keep tokens alive AND tell the caching allocator about their use
            # on the consumer stream. Model/reference tensors live for all rounds.
            tokens.record_stream(streams[1])
            wav, _ = model.s3gen.inference(speech_tokens=tokens, ref_dict=refs)
            with trace.measure("copy_to_cpu"):
                wav = wav.detach().cpu()
            completed = time.perf_counter() - started
        stages = token_trace.resolve(origin)
        stages.update(trace.resolve(origin))
        return summarize(index, wav, tokens, stages, completed)

    pending = deque()
    results = []
    for index, text in enumerate(texts):
        if mode == "original":
            results.append(workers[0].submit(original_generate, index, text).result())
            continue
        # Bounded lookahead: at most two sentences are in the pipeline.
        if len(pending) >= (2 if mode == "pipeline" else 1):
            results.append(pending.popleft().result())
        job = workers[0].submit(produce, index, text).result()
        pending.append(workers[1].submit(decode, index, job))
    while pending:
        results.append(pending.popleft().result())
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    audio = sum(r["audio_secs"] for r in results)
    overlap_ms = 0.0
    for prev, current in zip(results, results[1:]):
        a, b = prev["stages"]["decode"], current["stages"]["t3"]
        overlap_ms += max(0, min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"]))
    result = dict(mode=mode, round=round_no, wall_secs=wall, audio_secs=audio,
                  rtf=wall / audio, first_audio_secs=results[0]["completed_secs"],
                  overlap_secs=overlap_ms / 1000,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  requests=results)
    mean_stage = lambda name: statistics.mean(r["stages"][name]["elapsed_ms"] for r in results) / 1000
    label = "warm-up" if round_no < 0 else f"round {round_no + 1}"
    print(f"{mode:8} {label}: {wall:.3f}s for {audio:.2f}s audio; "
          f"RTF={wall / audio:.3f}; first={result['first_audio_secs']:.3f}s; "
          f"mean T3={mean_stage('t3'):.3f}s flow={mean_stage('flow'):.3f}s "
          f"vocoder={mean_stage('vocoder'):.3f}s; overlap={overlap_ms / 1000:.3f}s", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default="default")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--sentences", type=int, default=4, help="repeat built-in texts if more than four")
    p.add_argument("--texts", help="optional UTF-8 file: one sentence per line (overrides --sentences)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--output", default="benchmarks/full_pipeline/results.json")
    p.add_argument("--save-audio", action="store_true")
    p.add_argument("--include-original", action="store_true",
                   help="also benchmark unsplit model.generate as a control baseline")
    args = p.parse_args()
    if args.rounds < 1 or args.sentences < 1:
        p.error("--rounds and --sentences must be positive")
    if not torch.cuda.is_available():
        p.error("This experiment requires CUDA")
    if args.threads:
        torch.set_num_threads(args.threads)
    texts = ([line.strip() for line in Path(args.texts).read_text(encoding="utf-8").splitlines()
              if line.strip()] if args.texts else [TEXTS[i % len(TEXTS)] for i in range(args.sentences)])
    if not texts:
        p.error("No sentences to test")
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cpu_threads=torch.get_num_threads(), arguments=vars(args), runs=[])
    print(f"GPU: {report['gpu']}; PyTorch: {torch.__version__}; CPU threads: {torch.get_num_threads()}", flush=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    if args.voice != "default":
        voice = Path(__file__).parent / "voices" / f"{args.voice}.wav"
        with torch.inference_mode():
            model.prepare_conditionals(str(voice), exaggeration=args.exaggeration)
    profiler = ModelStageProfiler(model)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    with independent_sampling() as sampling, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="t3") as producer, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3gen") as decoder:
        validate_split(model, args, sampling)
        workers = (producer, decoder)
        # Warm both persistent worker threads and streams before measurement.
        for mode in (("original", "serial", "pipeline") if args.include_original else ("serial", "pipeline")):
            run_round(model, profiler, workers, streams, sampling, args, texts[:2], mode, -1)
        expected = None
        for round_no in range(args.rounds):
            # Alternate order to reduce thermal / clock / background-load bias.
            modes = ["original", "serial", "pipeline"] if args.include_original else ["serial", "pipeline"]
            if round_no % 2:
                modes.reverse()
            for mode in modes:
                result = run_round(model, profiler, workers, streams, sampling, args, texts, mode, round_no)
                hashes = [r["token_hash"] for r in result["requests"]]
                if expected is not None and hashes != expected:
                    raise RuntimeError("Token sequences changed across runs; comparison is not controlled")
                expected = hashes
                report["runs"].append(result)
                dest = Path(args.output)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    profiler.close()
    serial = statistics.median(r["wall_secs"] for r in report["runs"] if r["mode"] == "serial")
    pipeline = statistics.median(r["wall_secs"] for r in report["runs"] if r["mode"] == "pipeline")
    report["summary"] = dict(serial_median_secs=serial, pipeline_median_secs=pipeline,
                             speedup=serial / pipeline, time_saved_percent=100 * (1 - pipeline / serial))
    if args.include_original:
        original = statistics.median(r["wall_secs"] for r in report["runs"] if r["mode"] == "original")
        report["summary"].update(original_median_secs=original,
                                 speedup_vs_original=original / pipeline,
                                 time_saved_vs_original_percent=100 * (1 - pipeline / original))
        print(f"Against unsplit model.generate: {original / pipeline:.3f}x; "
              f"time saved: {100 * (1 - pipeline / original):.1f}%")
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Median speedup: {serial / pipeline:.3f}x; time saved: {100 * (1 - pipeline / serial):.1f}%")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
