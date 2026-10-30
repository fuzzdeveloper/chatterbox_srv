# Full Chatterbox: single-GPU pipeline experiment

Tested 2026-10-29 on this machine:

- NVIDIA GeForce GTX 1080 Ti (11 GiB, Windows/WDDM).
- PyTorch `2.14.1+cu126`, six PyTorch CPU threads.
- Local `chatterbox/src/chatterbox/tts.py`, the same implementation imported by the server.
- Default settings: temperature 0.8, exaggeration 0.5, CFG 0.5,
  min-p 0.05, top-p 1.0, repetition penalty 1.2.

## Conclusion

**Overlapping T3 for sentence N+1 with S3Gen for sentence N helps modestly on
this single GPU: about 7–12% less total generation time in these workloads.**
It is not contention-free. Both stages get slower, but the overlap more than
compensates. The first sentence can arrive later in this eager-overlap benchmark.
The servers still serialize generation by default, but now offer an optional
`--pipeline` mode. That production mode protects the first request of an idle
burst before enabling overlap; see `../../PIPELINING.md` for usage and tests.

These workloads still have an RTF above 1 after pipelining, so they cannot sustain
indefinite uninterrupted playback at the generated speaking rate. Buffering can
hide temporary delays, not a sustained generation deficit. Actual results vary
with text, voice, GPU clocks, and other workloads. We did not drive a concurrent
LLM on the GPU during these tests; performance with that additional contention
has not been established.

## Results

Each row uses three measured runs per mode, after warming the model, both
persistent worker threads, and both CUDA streams. Values are median batch wall
times, not averages of per-request HTTP times.

| Workload | Baseline | Pipeline | Time saved | Throughput speedup |
| --- | ---: | ---: | ---: | ---: |
| Adam, 4 sentences, **original unsplit model.generate** | 16.75 s | 14.76 s | 11.9% | 1.135x |
| Adam, 8 mixed-length sentences, serial split | 39.38 s | 35.69 s | 9.4% | 1.103x |
| Built-in voice, 4 sentences, serial split | 15.91 s | 14.73 s | 7.4% | 1.080x |

The original-control run also measured serial split generation at 16.93 s,
close to the original unsplit path at 16.75 s. This checks that the split
benchmark did not introduce a substantially inflated baseline. An earlier Adam
4-sentence run independently showed 10.9% less time (`results.json`).

### Stage timings: Adam, 8-sentence workload

These are mean stage times across 24 sentences per mode, not medians.

| Stage | Serial | Pipeline | Change |
| --- | ---: | ---: | ---: |
| T3, text to speech tokens | 3.978 s | 4.321 s | +8.6% |
| Flow matching, tokens to mel | 0.908 s | 1.074 s | +18.2% |
| Vocoder, mel to waveform | 0.054 s | 0.064 s | +19.1% |

This explains why the gain is smaller than ideal, contention-free overlap.
T3 is the dominant cost; the vocoder alone is too small to hide much work.
CUDA timelines showed approximately eight seconds of T3/previous-sentence
decoder range overlap per eight-sentence batch. This is overlap of stage
intervals, not a claim that every kernel in those intervals ran simultaneously.

For this workload, median time to the first audio increased from 3.90 s to
4.09 s, while batch RTF improved from 1.285 to 1.165. For the built-in voice,
first-audio time increased from 3.27 s to 3.97 s. The subsequently implemented
production pipeline deliberately avoids lookahead until the first sentence is
decoded, trading some throughput for first-audio latency. Its scheduling differs
from this eager-overlap benchmark, so these numbers are not an exact benchmark
of the new server flag.

## Controls and limitations

- Pipeline lookahead is bounded to two sentences: one T3 worker, one decoder
  worker, on separate CUDA streams. There are never two concurrent T3 calls or
  two concurrent S3Gen calls. The model weights are not duplicated.
- A CUDA event transfers readiness between streams; token tensors are kept
  alive and registered with the consumer stream's allocator lifetime tracking.
- Both modes use identical per-sentence T3 sampling seeds, independent of
  decoder RNG use. Final speech-token hashes are checked across **all** measured
  modes and runs. The benchmark aborts if the workload changes.
- Before measuring, split generation is checked against the original generate
  function: tokens must match exactly, and waveforms within float32 roundoff.
  The absolute waveform tolerance is below one 16-bit PCM quantization step.
- Stage timing uses asynchronous CUDA events, not device synchronization
  between stages. Timings include host launch gaps and contention visible on
  that stream, rather than reporting pure kernel execution time.
- The mode order is reversed on alternate rounds to reduce ordering bias.
  This is a practical local benchmark, not a fully isolated laboratory test.
- All sentences are available at batch start. There is no network, text arrival
  delay, playback, or concurrently driven LLM in the throughput benchmark.
- Warm voice conditioning is used. Cold cloning is excluded from throughput
  comparisons, but is measured separately by the server's timing flag.
- Waveforms are checked for finite, nonzero samples; this is not a subjective
  audio-quality assessment. Example WAVs from the initial Adam run are in
  `audio/`.

## Enable timings in the existing server

From the repository root:

```bat
.venv\Scripts\python chatterbox_full_server.py --device cuda --port 8882 --profile-stages
```

An additional line follows each normal request log. For example, from the HTTP
smoke test (the second request waited behind a first-use voice clone):

```text
timing: queue=5.367s synthesis=3.012s tokenize=0.000s t3=2.194s flow=0.772s vocoder=0.043s encode=0.0003s
```

- `queue`: time before the single synthesis worker starts the request.
- `synthesis`: wall time for voice lookup/preparation plus complete generation,
  including copying the waveform to CPU; it excludes queueing and encoding.
- `voice_prepare`: cold voice-cloning wall time, present only on cache misses.
- `tokenize`: tokenizer CPU wall time.
- `t3`, `flow`, `vocoder`: CUDA-event elapsed time on CUDA, wall time on CPU.
- `encode`: WAV/PCM conversion wall time, excluding socket transmission.

The listed sub-stages do not account for every tiny operation: punctuation
normalization, padding, token filtering, transfers, and Python overhead also
contribute to synthesis time. The profiler retains nested `decode` and
`generate` totals internally; don't add those to their constituent stages.
Profiling is opt-in. The normal server still serializes model access and its
HTTP API is unchanged. Concurrent HTTP requests were tested with profiling
both on and off, including WAV, PCM, empty-input errors, voice caching, and
queue timing (`server_*.log`, `http_smoke.json`). The HTTP smoke timings are
functional checks, not a controlled profiling-overhead comparison.

## Reproduce benchmarks

```bat
.venv\Scripts\python benchmark_full_pipeline.py --voice adam --rounds 3 --include-original --output benchmarks/full_pipeline/original_control.json
.venv\Scripts\python benchmark_full_pipeline.py --voice adam --texts benchmarks/full_pipeline/sentences.txt --rounds 3 --output benchmarks/full_pipeline/adam_long.json
.venv\Scripts\python benchmark_full_pipeline.py --voice default --rounds 3 --output benchmarks/full_pipeline/default.json
.venv\Scripts\python -m unittest discover -s tests -v
```

Optional `--save-audio` writes the first measured round to `audio/` beside the
output JSON. Raw JSON contains per-request completion times, token hashes,
stage start/end times relative to a common CUDA event, audio durations, and GPU
peak allocated/reserved memory. Logs retain warm-up and measured-run output.

The benchmark's token-generation helper mirrors the local upstream generate
implementation. If that implementation changes, review the helper; the startup
equivalence check is intended to catch divergence. The independent sampling
patch exists only inside the standalone benchmark process, never in the server.
