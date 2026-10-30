"""Optional bounded T3 -> S3Gen CUDA pipeline shared by the three HTTP servers.

Only one request uses each network at a time. Cold voice preparation takes the
same lock as decoding because it also uses S3Gen. The first request of an idle
burst is decoded before allowing lookahead, to protect first-audio latency.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import threading
import time

import torch
import torch.nn.functional as F


def add_pipeline_arguments(parser):
    parser.add_argument("--pipeline", action="store_true",
                        help="CUDA only: overlap next-request T3 with current-request audio decoding")


def validate_pipeline_arguments(parser, args):
    if args.pipeline and args.device != "cuda":
        parser.error("--pipeline requires --device cuda (CPU/MPS overlap is not supported)")
    if args.pipeline and getattr(args, "backend", None) == "mlx":
        parser.error("--pipeline requires a CUDA backend (auto, torch, or flashinfer), not mlx")
    return args


@dataclass
class SynthesisResult:
    voice: str
    wav: object
    queue_secs: float
    synthesis_secs: float
    stages: object = None


@dataclass
class _DecodeJob:
    kwargs: dict


def _generate_tokens(model, kind, text, options):
    """Mirror the token half of the installed local generate implementations.

    This is used ONLY with --pipeline. Ordinary requests still call upstream
    generate unchanged. Equivalence tests should be rerun after model upgrades.
    No request's conditioning is read from model.conds by the decoder thread.
    """
    if model.conds is None:
        raise ValueError("Conditioning not prepared; select an available voice")
    cond = replace(model.conds.t3)
    refs = dict(model.conds.gen)  # S3Gen casts/mutates its reference dictionary.
    if kind == "full":
        from chatterbox import tts
        from chatterbox.models.t3.modules.cond_enc import T3Cond

        exaggeration = options.get("exaggeration", 0.5)
        if exaggeration != cond.emotion_adv[0, 0, 0]:
            cond = T3Cond(
                speaker_emb=cond.speaker_emb,
                cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=model.device)
        cfg = options.get("cfg_weight", 0.5)
        text_tokens = model.tokenizer.text_to_tokens(tts.punc_norm(text)).to(model.device)
        if cfg > 0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        text_tokens = F.pad(text_tokens, (1, 0), value=model.t3.hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=model.t3.hp.stop_text_token)
        tokens = model.t3.inference(
            t3_cond=cond, text_tokens=text_tokens, max_new_tokens=1000,
            temperature=options.get("temperature", 0.8), cfg_weight=cfg,
            repetition_penalty=options.get("repetition_penalty", 1.2),
            min_p=options.get("min_p", 0.05), top_p=options.get("top_p", 1.0),
        )[0]
        tokens = tts.drop_invalid_tokens(tokens)
        tokens = tokens[tokens < 6561].to(model.device)
        decode_options = {}
    elif kind == "turbo":
        from chatterbox.tts_turbo import punc_norm
        from chatterbox.models.s3gen.const import S3GEN_SIL

        text_tokens = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True)
        tokens = model.t3.inference_turbo(
            t3_cond=cond, text_tokens=text_tokens.input_ids.to(model.device),
            temperature=options.get("temperature", 0.8), top_k=options.get("top_k", 1000),
            top_p=options.get("top_p", 0.95), repetition_penalty=options.get("repetition_penalty", 1.2),
        )
        tokens = tokens[tokens < 6561].to(model.device)
        silence = torch.tensor([S3GEN_SIL] * 3, device=model.device, dtype=torch.long)
        tokens = torch.cat([tokens, silence])
        decode_options = {"n_cfm_timesteps": 2}
    elif kind == "flash":
        from chatterbox_flash.tts import _speech_len_for_text_tokens

        text_tokens = model._encode_text(text)
        n_tokens = int(text_tokens.size(1))
        tokens = model.t3.generate(
            t3_cond=cond, text_tokens=text_tokens,
            text_token_lens=torch.tensor([n_tokens], device=model.device),
            total_speech_len=_speech_len_for_text_tokens(n_tokens),
            num_steps=options.get("num_steps", 10), temperature=options.get("temperature", 0.6),
            time_shift_tau=options.get("time_shift_tau", 0.1),
            cfg_scale=options.get("cfg_scale", 1.0),
            position_temperature=options.get("position_temperature", 5.0),
            # Graph capture can synchronize the whole device and may not run
            # safely while the decoder is executing on another thread/stream.
            use_cuda_graph=False, backend=options.get("backend", "auto"), batch_size=1,
        )
        tokens = model._trim_to_eos(tokens[0]).to(model.device)
        decode_options = {"n_cfm_timesteps": options.get("n_cfm_timesteps", 2)}
    else:
        raise ValueError(f"Unknown pipeline model kind: {kind}")
    return _DecodeJob(dict(speech_tokens=tokens, ref_dict=refs, **decode_options))


class SynthesisWorker:
    """Executor-like facade with one producer and, optionally, one decoder.

    synthesize callbacks must return (resolved_voice, worker.generate(...)).
    Never wait for another submitted request inside a synthesis callback.
    Pending HTTP requests may queue, but at most TWO requests enter GPU stages.
    """

    def __init__(self, model, kind, *, pipeline=False, profiler=None):
        self.model = model
        self.kind = kind
        self.pipeline = pipeline
        self.profiler = profiler
        self.device = torch.device(model.device)
        if pipeline and self.device.type != "cuda":
            raise ValueError("--pipeline requires a CUDA model")
        self.lock = threading.Lock()
        self.decoder_lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(2)
        self.outstanding = 0
        self.closed = False
        self.local = threading.local()
        if pipeline:
            # Model/built-in-voice initialization occurred on the main stream.
            torch.cuda.synchronize(self.device)
            self.producer_stream = torch.cuda.Stream(device=self.device)
            self.decoder_stream = torch.cuda.Stream(device=self.device)
        self.producer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.decoder = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")
                        if pipeline else None)

    def generate(self, text, **options):
        if not self.pipeline:
            return self.model.generate(text, **options)
        if not getattr(self.local, "producing", False):
            raise RuntimeError("Pipeline generation must run inside the synthesis worker")
        return _generate_tokens(self.model, self.kind, text, options)

    @contextmanager
    def conditioning(self):
        """Use ONLY around cold voice preparation, not around cache lookups."""
        if not self.pipeline:
            yield
            return
        with self.decoder_lock:
            try:
                yield
            finally:
                # Even a failed clone may have enqueued work using S3Gen.
                torch.cuda.current_stream(self.device).synchronize()

    def submit(self, fn, *args, submitted=None):
        future = Future()
        submitted = time.perf_counter() if submitted is None else submitted
        with self.lock:
            if self.closed:
                raise RuntimeError("Synthesis worker is shutting down")
            protect_first = self.outstanding == 0
            self.outstanding += 1
            try:
                self.producer.submit(self._produce, future, fn, args, submitted, protect_first)
            except BaseException:
                self.outstanding -= 1
                raise
        return future

    def _finished(self, future, *, result=None, error=None):
        with self.lock:
            self.outstanding -= 1
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    @torch.inference_mode()
    def _produce(self, future, fn, args, submitted, protect_first):
        if not future.set_running_or_notify_cancel():
            with self.lock:
                self.outstanding -= 1
            return
        acquired = False
        handed_off = False
        try:
            if self.pipeline:
                self.slots.acquire()
                acquired = True
            began = time.perf_counter()
            stream_context = torch.cuda.stream(self.producer_stream) if self.pipeline else nullcontext()
            with stream_context, (self.profiler.capture() if self.profiler else nullcontext()) as trace:
                self.local.producing = True
                try:
                    voice, audio = fn(*args)
                finally:
                    self.local.producing = False
                if self.pipeline:
                    ready = torch.cuda.Event()
                    ready.record(self.producer_stream)
            if not self.pipeline or not isinstance(audio, _DecodeJob):
                # Flash's unavailable-built-in-voice case returns None.
                stages = trace.resolve() if trace else None
                self._finished(future, result=SynthesisResult(
                    voice, audio, began - submitted, time.perf_counter() - began, stages))
                return
            decoded = self.decoder.submit(self._decode, future, voice, audio, ready, trace, began, submitted)
            handed_off = True
            if protect_first:
                # Also applies to a new burst after all earlier requests finish.
                decoded.result()
        except BaseException as error:
            if not handed_off:
                if self.pipeline:
                    try:
                        self.producer_stream.synchronize()
                    except BaseException as sync_error:
                        error = sync_error
                self._finished(future, error=error)
        finally:
            if acquired and not handed_off:
                self.slots.release()

    @torch.inference_mode()
    def _decode(self, future, voice, job, ready, token_trace, began, submitted):
        try:
            with torch.cuda.stream(self.decoder_stream):
                self.decoder_stream.wait_event(ready)
                for tensor in [job.kwargs["speech_tokens"], *job.kwargs["ref_dict"].values()]:
                    if torch.is_tensor(tensor) and tensor.is_cuda:
                        tensor.record_stream(self.decoder_stream)
                with (self.profiler.capture() if self.profiler else nullcontext()) as trace:
                    with self.decoder_lock:
                        try:
                            wav, _ = self.model.s3gen.inference(**job.kwargs)
                            wav = wav.detach().cpu()
                        finally:
                            self.decoder_stream.synchronize()
            stages = token_trace.resolve() if token_trace else None
            if trace:
                stages.update(trace.resolve())
            self._finished(future, result=SynthesisResult(
                voice, wav, began - submitted, time.perf_counter() - began, stages))
        except BaseException as error:
            self._finished(future, error=error)
        finally:
            self.slots.release()

    def shutdown(self, wait=True):
        if not wait:
            raise ValueError("SynthesisWorker requires a draining shutdown (wait=True)")
        with self.lock:
            self.closed = True
        # Producers must finish scheduling their decoders before decoder shutdown.
        self.producer.shutdown(wait=True)
        if self.decoder is not None:
            self.decoder.shutdown(wait=True)
