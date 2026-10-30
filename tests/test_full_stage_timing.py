import threading
import time
from types import SimpleNamespace
import unittest

import torch

from chatterbox_full_timing import ModelStageProfiler, StageTimings, format_stage_timings


def fake_model():
    def work():
        time.sleep(0.001)
        return 42

    model = SimpleNamespace(device="cpu", prepare_conditionals=work)
    model.tokenizer = SimpleNamespace(text_to_tokens=work)
    model.t3 = SimpleNamespace(inference=work)
    model.s3gen = SimpleNamespace(flow_inference=work, hift_inference=work)

    def decode():
        model.s3gen.flow_inference()
        return model.s3gen.hift_inference()

    def generate():
        model.tokenizer.text_to_tokens()
        model.t3.inference()
        return model.s3gen.inference()

    model.s3gen.inference = decode
    model.generate = generate
    return model


class StageTimingTests(unittest.TestCase):
    def test_cpu_nested_stages_and_return_values(self):
        model = fake_model()
        original = model.generate
        profiler = ModelStageProfiler(model)
        try:
            self.assertEqual(model.generate(), 42)  # capture is opt-in
            with profiler.capture() as trace:
                self.assertEqual(model.generate(), 42)
            result = trace.resolve()
            self.assertEqual(set(result), {"tokenize", "t3", "flow", "vocoder", "decode", "generate"})
            self.assertGreater(result["generate"]["elapsed_ms"], result["t3"]["elapsed_ms"])
            self.assertGreater(result["decode"]["elapsed_ms"], result["flow"]["elapsed_ms"])
            self.assertNotIn("generate=", format_stage_timings(result))
            self.assertNotIn("decode=", format_stage_timings(result))
        finally:
            profiler.close()
        self.assertIs(model.generate, original)

    def test_capture_is_thread_local(self):
        model = fake_model()
        profiler = ModelStageProfiler(model)
        try:
            with profiler.capture() as trace:
                worker = threading.Thread(target=model.generate)
                worker.start()
                worker.join()
            self.assertEqual(trace.resolve(), {})
        finally:
            profiler.close()

    def test_nested_capture_restores_outer_trace(self):
        model = fake_model()
        profiler = ModelStageProfiler(model)
        try:
            with profiler.capture() as outer:
                with profiler.capture() as inner:
                    model.t3.inference()
                model.s3gen.hift_inference()
            self.assertEqual(set(inner.resolve()), {"t3"})
            self.assertEqual(set(outer.resolve()), {"vocoder"})
        finally:
            profiler.close()

    def test_exception_preserved_and_capture_restored(self):
        model = fake_model()
        profiler = ModelStageProfiler(model)
        try:
            with self.assertRaisesRegex(ValueError, "expected"):
                with profiler.capture() as trace:
                    with trace.measure("failed"):
                        raise ValueError("expected")
            self.assertIn("failed", trace.resolve())
            model.generate()
            self.assertEqual(set(trace.resolve()), {"failed"})
        finally:
            profiler.close()

    def test_cpu_heavy_stages_format_wall_time(self):
        result = {name: {"elapsed_ms": 1, "wall_ms": 2000}
                  for name in ("tokenize", "voice_prepare", "t3")}
        formatted = format_stage_timings(result)
        self.assertIn("tokenize=2.000s", formatted)
        self.assertIn("voice_prepare=2.000s", formatted)
        self.assertIn("t3=0.001s", formatted)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_events_on_nondefault_stream(self):
        stream = torch.cuda.Stream()
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        stream.wait_event(origin)
        trace = StageTimings("cuda")
        with torch.cuda.stream(stream), trace.measure("gpu"):
            value = torch.ones(1024, device="cuda") * 2
        result = trace.resolve(origin)["gpu"]
        self.assertGreaterEqual(result["elapsed_ms"], 0)
        self.assertGreaterEqual(result["end_ms"], result["start_ms"])
        self.assertEqual(value.sum().item(), 2048)


if __name__ == "__main__":
    unittest.main()
