# Optional CUDA sentence pipelining

All three servers accept `--pipeline`, disabled by default:

```bat
.venv\Scripts\python chatterbox_full_server.py --device cuda --port 8882 --pipeline
.venv\Scripts\python chatterbox_turbo_server.py --device cuda --port 8880 --pipeline
.venv\Scripts\python chatterbox_flash_server.py --device cuda --port 8881 --pipeline
```

Keep any other settings you normally use. For example, on the GTX 1080 Ti the
Flash tests used `--dtype fp32 --backend torch`. The Turbo adapter also supports
`--nano` through the same token-generation architecture, although actual Nano
weights were not available locally and were not downloaded for these tests.

Add **`--profile-stages`** to any of the three commands to log queue wait,
synthesis wall time, T3, flow matching, vocoder, and WAV/PCM encoding timings.
Full and Flash also report tokenizer timing; Turbo tokenization is included in
the overall synthesis time rather than listed separately.

Without `--pipeline`, all servers still use one persistent synthesis thread and
call the original model's `generate` method. The HTTP API is unchanged. CPU and
MPS continue to work in that serialized mode; requesting `--pipeline` on them
is rejected explicitly rather than silently pretending to overlap work.

## Client requirements

Send later sentences **before waiting for the current request's response**.
Simply buffering text locally but sending one HTTP request at a time leaves no
work for the server to overlap. Keep responses associated with their original
sentence and play them in sentence order. HTTP arrival order across separate
connections is not guaranteed.

Two or more outstanding requests are enough to permit steady-state overlap;
there is no need to flood the server with a whole conversation. Only two
requests at a time enter the GPU stages; further requests wait in the synthesis
queue. This bounds GPU lookahead, not the total number of pending HTTP requests.

## Scheduling and safeguards

`chatterbox_pipeline.py` provides a shared scheduler and model-specific token
adapters:

- One persistent T3 worker and one persistent S3Gen worker, on separate CUDA
  streams. No concurrent calls to the same network and no duplicate model weights.
- T3 for the next request can overlap S3Gen for the previous one.
- The first request after the queue becomes idle is **fully decoded before
  allowing lookahead**, preserving first-audio responsiveness. Protection is
  server-wide for each idle burst, not keyed to a user/session.
- Requests pass their own speech tokens and a separate reference dictionary to
  decoding. The decoder never reads the mutable `model.conds` of a later request.
- Cold cloning or recloning after a reference WAV changes is mutually exclusive
  with decoding, because cloning also uses S3Gen. Cache hits need no such lock.
- CUDA readiness events and allocator stream-lifetime tracking protect tensors
  handed between workers. Conditioning and error paths finish their outstanding
  stream work before releasing shared network access.
- Inference mode covers both persistent worker threads. Errors propagate to the
  relevant request and release pipeline slots, allowing subsequent requests to
  run. Shutdown drains producers before closing the decoder executor.
- **Flash CUDA graphs are disabled in pipeline mode.** Capturing or recapturing
  them while another thread/stream is using the GPU is unsafe; capture can also
  synchronize the whole device. The ordinary nonpipeline graph setting is
  unchanged. `--backend mlx` is rejected with `--pipeline`.

The adapters mirror the token-generation halves of the installed local model
implementations, including Full's invalid-token fix, Turbo's trailing silence,
and Flash's normalization, budget, and EOS trimming. After upgrading those
implementations, rerun the model equivalence checks below. They intentionally
compare against the original `generate` path instead of only testing the
adapters against themselves.

## Performance expectations

This is an experimental throughput option, **not an unconditional speedup**.
Both stages share GPU resources and the Python runtime; individual stage times
can increase during overlap. The earlier controlled Full-model experiments on
the GTX 1080 Ti showed roughly 7–12% less batch generation time. That benchmark
used eager overlap even for the first sentence; the production scheduler now
protects the first sentence, trading a little throughput for responsiveness.
See `benchmarks/full_pipeline/README.md` for those measurements and caveats.

Do not assume the Full-model gain applies to Turbo or Flash: their relative
T3/decoder costs differ, and disabling Flash graphs can change its trade-off
on supported GPUs. Compare with and without the flag on your actual workload,
particularly if an LLM is also generating on the same GPU. Small queues, cold
cloning, or only one outstanding request can erase any throughput gain. Random
sampling also means concurrent scheduling can change audio between otherwise
identical unseeded requests.

## Validation performed

On the local GTX 1080 Ti, PyTorch 2.14.1+cu126:

- **12 unit/timing tests passed**, including actual CUDA stream handoff,
  first-request protection, two-request bound, overlap, errors in either stage,
  missing-voice results, cancellation, draining shutdown, and clone/decode
  exclusion.
- Seeded **Full, Turbo, and Flash** comparisons passed against each model's
  original `generate` path, with built-in and Adam voices and varied sampling /
  CFG / flow settings. Speech tokens matched exactly; maximum waveform errors
  were approximately 0.0000003–0.0000022 (float32 roundoff).
- All three real HTTP servers passed in **both serialized and pipelined modes**:
  eight requests per configuration, including a first request, six concurrent
  requests, and a request after draining. Checks included cold and cached Adam /
  Heart voices, default / unknown voices, parameter overrides, WAV / PCM,
  invalid-input errors, and stage timing logs.
- These are correctness checks, not a controlled Turbo/Flash throughput study.
  They do not establish subjective audio quality. Flash used FP32 / torch;
  FlashInfer and newer-GPU BF16 paths were not tested. Nano was not tested with
  real weights.

Raw logs and JSON are in `benchmarks/pipeline_servers/`. Reproduce from the
repository root (one real model is loaded at a time):

```bat
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python tests/check_pipeline_models.py --kind full
.venv\Scripts\python tests/check_pipeline_models.py --kind turbo
.venv\Scripts\python tests/check_pipeline_models.py --kind flash
.venv\Scripts\python tests/smoke_pipeline_servers.py
```

GPU sub-stage times use asynchronous CUDA events and include host launch gaps
and contention visible on that stream. `synthesis` is wall time from the start
of producer work until CPU audio is ready; in pipeline mode it can include a
wait for the decoder. `queue` is the preceding wait for producer admission.
Do not add nested profiler totals or assume summed stage times capture every
transfer, text cleanup operation, and Python overhead.
