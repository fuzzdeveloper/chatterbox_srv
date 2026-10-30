from concurrent.futures import CancelledError
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from chatterbox_pipeline import SynthesisWorker, _DecodeJob


class SerialWorkerTests(unittest.TestCase):
    def test_default_path_uses_original_generate_and_propagates_errors(self):
        model = SimpleNamespace(device="cpu", generate=lambda text, **kw: (text, kw))
        worker = SynthesisWorker(model, "full")
        try:
            def synth(text):
                return "voice", worker.generate(text, temperature=0.7)

            result = worker.submit(synth, "hello").result(timeout=5)
            self.assertEqual(result.wav, ("hello", {"temperature": 0.7}))
            self.assertEqual(result.voice, "voice")
            self.assertGreaterEqual(result.queue_secs, 0)
            self.assertIsNone(result.stages)
            with self.assertRaisesRegex(ValueError, "bad request"):
                worker.submit(lambda: (_ for _ in ()).throw(ValueError("bad request"))).result(timeout=5)
            self.assertEqual(worker.submit(synth, "still works").result(timeout=5).voice, "voice")
        finally:
            worker.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            worker.submit(synth, "closed")

    def test_cancel_pending_and_shutdown_drains(self):
        gate = threading.Event()
        worker = SynthesisWorker(SimpleNamespace(device="cpu"), "full")
        first = worker.submit(lambda: (gate.wait(5), None))
        cancelled = worker.submit(lambda: ("unused", None))
        self.assertTrue(cancelled.cancel())
        gate.set()
        worker.shutdown()
        first.result(timeout=5)
        with self.assertRaises(CancelledError):
            cancelled.result()
        self.assertEqual(worker.outstanding, 0)

    def test_rejects_non_cuda_pipeline(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            SynthesisWorker(SimpleNamespace(device="cpu"), "full", pipeline=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaPipelineTests(unittest.TestCase):
    def make_worker(self, decode):
        model = SimpleNamespace(device="cuda", s3gen=SimpleNamespace(inference=decode))
        return SynthesisWorker(model, "test", pipeline=True)

    def test_first_protected_then_overlap_bounded_and_ordered(self):
        events = []
        second_decoding = threading.Event()
        third_producing = threading.Event()
        active = 0
        peak = 0
        lock = threading.Lock()

        def produce(model, kind, text, options):
            nonlocal active, peak
            index = int(text)
            with lock:
                active += 1
                peak = max(peak, active)
                events.append(("produce", index))
            if index == 2:
                self.assertTrue(second_decoding.wait(5))
                third_producing.set()
            return _DecodeJob(dict(speech_tokens=torch.tensor([index], device="cuda"), ref_dict={}))

        def decode(speech_tokens, ref_dict):
            nonlocal active
            index = int(speech_tokens.item())
            if index == 1:
                second_decoding.set()
                self.assertTrue(third_producing.wait(5))
            with lock:
                events.append(("decoded", index))
                active -= 1
            return torch.ones(1, 16, device="cuda") * (index + 1), None

        worker = self.make_worker(decode)
        try:
            with patch("chatterbox_pipeline._generate_tokens", produce):
                def synth(index):
                    return f"voice{index}", worker.generate(str(index))

                futures = [worker.submit(synth, i) for i in range(5)]
                results = [f.result(timeout=15) for f in futures]
            self.assertLess(events.index(("decoded", 0)), events.index(("produce", 1)))
            self.assertEqual(peak, 2)
            self.assertEqual([r.voice for r in results], [f"voice{i}" for i in range(5)])
            self.assertEqual([r.wav[0, 0].item() for r in results], [1, 2, 3, 4, 5])
            self.assertTrue(all(not r.wav.is_cuda for r in results))
        finally:
            worker.shutdown()
        self.assertEqual(worker.outstanding, 0)

    def test_producer_decoder_failures_and_none_do_not_stall(self):
        def produce(model, kind, text, options):
            if text == "producer error":
                raise ValueError("producer failed")
            return _DecodeJob(dict(speech_tokens=torch.tensor([int(text)], device="cuda"), ref_dict={}))

        def decode(speech_tokens, ref_dict):
            if speech_tokens.item() == 1:
                raise ValueError("decoder failed")
            return torch.ones(1, 4, device="cuda"), None

        worker = self.make_worker(decode)
        try:
            with patch("chatterbox_pipeline._generate_tokens", produce):
                def synth(text):
                    return "voice", worker.generate(text)

                futures = [worker.submit(synth, text) for text in ("producer error", "1", "2", "3")]
                for future, message in zip(futures[:2], ("producer failed", "decoder failed")):
                    with self.assertRaisesRegex(ValueError, message):
                        future.result(timeout=10)
                for future in futures[2:]:
                    self.assertEqual(future.result(timeout=10).wav.numel(), 4)
                self.assertIsNone(worker.submit(lambda: ("default", None)).result(timeout=10).wav)
        finally:
            worker.shutdown()
        self.assertEqual(worker.outstanding, 0)

    def test_cold_conditioning_waits_for_decoder(self):
        decoding = threading.Event()
        allow_finish = threading.Event()
        clone_started = threading.Event()
        events = []

        def produce(model, kind, text, options):
            return _DecodeJob(dict(speech_tokens=torch.ones(1, device="cuda", dtype=torch.long), ref_dict={}))

        def decode(**kwargs):
            decoding.set()
            self.assertTrue(allow_finish.wait(5))
            events.append("decoded")
            return torch.ones(1, 4, device="cuda"), None

        worker = self.make_worker(decode)
        try:
            with patch("chatterbox_pipeline._generate_tokens", produce):
                first = worker.submit(lambda: ("voice", worker.generate("first")))
                self.assertTrue(decoding.wait(5))

                def clone():
                    with worker.conditioning():
                        events.append("cloned")
                        clone_started.set()

                thread = threading.Thread(target=clone)
                thread.start()
                self.assertFalse(clone_started.wait(0.05))
                allow_finish.set()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                first.result(timeout=5)
                self.assertEqual(events, ["decoded", "cloned"])
        finally:
            allow_finish.set()
            worker.shutdown()


if __name__ == "__main__":
    unittest.main()
