"""Opt-in, thread-local stage timing for full, Turbo/Nano, and Flash Chatterbox.

CUDA events measure elapsed time on the current stream (including launch gaps),
not just kernel execution time. No stage-boundary synchronization is introduced.
Resolve a trace only after the request has finished. CPU timings use perf_counter.
"""

from contextlib import contextmanager
from functools import wraps
import threading
import time

import torch


class StageTimings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.records = []

    @contextmanager
    def measure(self, name):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        wall_start = time.perf_counter()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            if self.cuda:
                end.record()
                self.records.append((name, wall_ms, start, end))
            else:
                self.records.append((name, wall_ms, None, None))

    def resolve(self, origin=None):
        """Return milliseconds; nested stages must not be added together."""
        result = {}
        for name, wall_ms, start, end in self.records:
            if end is not None:
                end.synchronize()
            result[name] = {
                "wall_ms": wall_ms,
                "elapsed_ms": start.elapsed_time(end) if start is not None else wall_ms,
            }
            if origin is not None and start is not None:
                result[name]["start_ms"] = origin.elapsed_time(start)
                result[name]["end_ms"] = origin.elapsed_time(end)
        return result


class ModelStageProfiler:
    """Install once on a model; only threads inside capture() collect timings.

    Does not make the model thread-safe; scheduling is handled by SynthesisWorker.
    decode includes flow/vocoder; generate includes all its nested stages.
    """

    def __init__(self, model):
        self.device = model.device
        self.local = threading.local()
        self.originals = []
        for obj, attr, name in (
            (model, "prepare_conditionals", "voice_prepare"),
            (model.tokenizer, "text_to_tokens", "tokenize"),
            (model.t3, "inference", "t3"),
            (model.t3, "inference_turbo", "t3"),
            (model.t3, "generate", "t3"),
            (model.s3gen, "flow_inference", "flow"),
            (model.s3gen, "hift_inference", "vocoder"),
            (model.s3gen, "inference", "decode"),
            (model, "generate", "generate"),
        ):
            if hasattr(obj, attr):
                self._wrap(obj, attr, name)

    def _wrap(self, obj, attr, name):
        original = getattr(obj, attr)

        @wraps(original)
        def measured(*args, **kwargs):
            trace = getattr(self.local, "trace", None)
            if trace is None:
                return original(*args, **kwargs)
            with trace.measure(name):
                return original(*args, **kwargs)

        self.originals.append((obj, attr, original))
        setattr(obj, attr, measured)

    @contextmanager
    def capture(self):
        previous = getattr(self.local, "trace", None)
        trace = StageTimings(self.device)
        self.local.trace = trace
        try:
            yield trace
        finally:
            self.local.trace = previous

    def close(self):
        for obj, attr, original in reversed(self.originals):
            setattr(obj, attr, original)
        self.originals.clear()


def format_stage_timings(stages):
    # Deliberately omit nested decode/generate totals from the stage breakdown.
    names = ("voice_prepare", "tokenize", "t3", "flow", "vocoder")
    return " ".join(
        f"{name}={stages[name]['wall_ms' if name in ('tokenize', 'voice_prepare') else 'elapsed_ms'] / 1000:.3f}s"
        for name in names if name in stages
    )
