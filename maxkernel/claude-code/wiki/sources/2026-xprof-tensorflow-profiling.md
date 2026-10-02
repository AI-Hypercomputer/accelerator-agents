---
title: "XProf docs: Optimize TensorFlow performance using XProf"
type: source
tags: [docs, profiler, tensorflow, keras, profile-capture, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

TensorFlow-specific XProf capture guide. Documents four profiling APIs (Keras `TensorBoard` callback, `tf.profiler.experimental.start`/`stop`, the `tf.profiler.experimental.Profile` context manager, and sampling mode via `tf.profiler.experimental.server.start` + `client.trace(...)`), a custom-training-loop step-annotation pattern using `tf.profiler.experimental.Trace(name, step_num=..., _r=1)`, and a table of which APIs support local / remote / multi-worker / which hardware platforms. Includes the important warning that profiling too many steps can OOM, and the correct placement of the dataset iterator relative to the step trace context.

## Overview

TensorFlow offers four profiling APIs, each with different reach:

1. **Keras TensorBoard callback** — `tf.keras.callbacks.TensorBoard(log_dir, profile_batch='start, end')`. Local-only, CPU/GPU.
2. **`tf.profiler.experimental.start(logdir)` / `stop()`** — programmatic start/stop. Local-only, CPU/GPU.
3. **`with tf.profiler.experimental.Profile(logdir):`** — context manager. Local-only, CPU/GPU.
4. **Sampling mode (`server.start(port)` + `client.trace(url, logdir, duration_ms)`)** — gRPC-based; **the only API that supports remote, multi-worker, and TPU**.

For custom training loops, wrap each step with `tf.profiler.experimental.Trace('train', step_num=step, _r=1)` — the `_r=1` kwarg tells XProf to treat the event as a step event, which in turn enables step-based analysis in the trace viewer and input-pipeline analyzer.

## Key claims

- **Profile no more than ~10 steps.** Running the profiler too long can OOM. Skip the first few batches to avoid initialization artifacts.
- **Include the dataset iterator inside the `tf.profiler.experimental.Trace` context**, not outside. The anti-pattern `for step, train_data in enumerate(dataset): with Trace(...): train_step(train_data)` produces inaccurate input-pipeline analysis because the data-fetch time is excluded from the step span.
- **`_r=1`** is the magic kwarg that makes `tf.profiler.experimental.Trace` events register as XProf step events (without it, step-based analysis does not kick in).
- **Only `tf.profiler.experimental.client.trace` supports TPU**. The Keras callback, start/stop API, and context manager are CPU/GPU-only.
- Multi-worker profiling uses a comma-separated list of gRPC URLs: `'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,...'`.
- The doc's example port for the in-process server is **6009** (other frameworks' docs use 9999 / 9012; all are arbitrary).
- Traces can be written to GCS directly (`gs://your_tb_logdir`) from `client.trace`.

## Key data points

### API capability matrix

| API | Local | Remote | Multiple workers | Hardware |
|---|---|---|---|---|
| `tf.keras.callbacks.TensorBoard` | Supported | Not supported | Not supported | CPU, GPU |
| `tf.profiler.experimental.start/stop` | Supported | Not supported | Not supported | CPU, GPU |
| `tf.profiler.experimental.client.trace` | Supported | Supported | Supported | **CPU, GPU, TPU** |
| `tf.profiler.experimental.Profile` (context manager) | Supported | Not supported | Not supported | CPU, GPU |

### Capture snippets

Keras callback — profile batches 10..15:
```python
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, profile_batch='10, 15')
model.fit(train_data, steps_per_epoch=20, epochs=5,
          callbacks=[tb_callback])
```

Function API:
```python
tf.profiler.experimental.start('logdir')
# ... train the model ...
tf.profiler.experimental.stop()
```

Context manager:
```python
with tf.profiler.experimental.Profile('logdir'):
    # ... train the model ...
    pass
```

Sampling (single host):
```python
tf.profiler.experimental.server.start(6009)
# ... model runs ...
tf.profiler.experimental.client.trace(
    'grpc://localhost:6009', 'gs://your_tb_logdir', 2000)  # 2 s
```

Sampling (multiple workers):
```python
tf.profiler.experimental.client.trace(
    'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
    'gs://your_tb_logdir',
    2000)
```

### Custom training loop annotation (correct pattern)

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)   # inside: input pipeline measured
        train_step(train_data)
```

### Anti-pattern (do not use)

```python
for step, train_data in enumerate(dataset):          # data-fetch outside Trace
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)                       # input analysis is wrong
```

### `Trace` arguments

| Arg | Role |
|---|---|
| `name` (positional) | Prefix for the step label in the trace viewer |
| `step_num=<int>` | Appended to the step name; step index used by XProf |
| `_r=1` | Marks the event as a step event (required for step-based analysis) |

## Techniques referenced

- **Keras-integrated profile capture** via the TensorBoard callback (`profile_batch='start, end'`).
- **Programmatic capture** via `tf.profiler.experimental.start/stop` or the `Profile` context manager.
- **Sampling-mode / on-demand capture** via `server.start(port)` + `client.trace(url, logdir, duration_ms)` — the only TF API with TPU and multi-worker support.
- **Multi-worker profiling** via comma-separated gRPC URLs.
- **Step annotation** via `tf.profiler.experimental.Trace(name, step_num=..., _r=1)` for custom training loops.
- **Direct GCS logdir** — `gs://...` accepted as the `logdir` argument.
- **Capture-Profile UI dialog** options — service URLs / TPU names, duration, tracer levels, retry count.

## Gaps & caveats

- **10-step cap for profiler runtime** — exceeding this risks OOM. Relevant for long training loops.
- **Skip initialization batches** (first few) to avoid lopsided timing.
- **Dataset iterator placement pitfall** — excluding `next(dataset)` from the `Trace` context makes input-pipeline analysis inaccurate. Easy to get wrong in refactors.
- **Only `client.trace` supports TPU.** If your experiment is on TPU, you cannot use the Keras callback, the `start/stop` API, or the `Profile` context manager. This is the most load-bearing caveat of the doc for TPU-focused research.
- **HPA / autoscaling caveats** not covered here (they appear in the Kubernetes deployment doc but apply to TF too).
- The sampling example uses port `6009` for the in-process gRPC server and the worker default `8466`; neither is a protocol default — both are examples.
- **No `ProfileOptions` surface** documented here (unlike the JAX doc). The Capture Profile UI exposes host/device/Python tracer levels but the TF-side programmatic option surface is not enumerated.
- **No TPU env-var caveats** (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`) — orthogonal but relevant.
- **No guidance on `tf.function` vs eager** performance implications in the profile — the Capture Profile UI exposes a host-tracer level but the TF doc does not elaborate.
- Doc is mostly stable TF 2 APIs; some links point out to the TF 2 profiling tutorial and TF Dev Summit 2020 talks.

## Connections

- `profile-capture` — framework-agnostic capture concept.
- `tf-profiler-trace` — `tf.profiler.experimental.Trace` with `_r=1` step annotation.
- `tf-profiler-client-trace` — sampling-mode / on-demand capture (only TPU-capable TF API).
- `keras-tensorboard-callback` — `tf.keras.callbacks.TensorBoard(profile_batch=...)`.
- `input-pipeline-analysis` — step-span accuracy depends on iterator placement.
- `multi-worker-profile-capture` — comma-separated gRPC URLs.
- `xprof-sessions` — multi-capture runs as separate sessions.

## See also

- [xprof](../codebases/xprof.md)
- [XProf docs: Capturing profiles](2026-xprof-capturing-profiles.md)
- [XProf docs: JAX profiling](2026-xprof-jax-profiling.md)
- [XProf docs: PyTorch/XLA profiling](2026-xprof-pytorch-xla-profiling.md)
- [XProf docs: Docker deployment](2026-xprof-docker-deployment.md)
- [XProf docs: Kubernetes deployment](2026-xprof-kubernetes-deployment.md)

## Sources

- `raw/code/xprof/docs/tensorflow_profiling.md`
