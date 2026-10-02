---
title: "XProf docs: Profiling PyTorch/XLA workloads"
type: source
tags: [docs, profiler, pytorch-xla, tpu, profile-capture]
created: 2026-04-22
updated: 2026-04-22
---

PyTorch/XLA-specific XProf capture guide. Covers the `torch_xla.debug.profiler` module (aliased `xp`): `xp.start_server(port)` to start the in-process profiler gRPC server, `xp.start_trace(dir)` / `xp.stop_trace()` to bracket a programmatic capture, and `xp.Trace(name)` context manager to add human-readable labels to the trace. Ends with the standard `xprof --port=8791 <logdir>` launch. Doc is short and example-driven; it's the minimum an experiment-runner needs to instrument a PyTorch/XLA training script on TPU.

## Overview

Three steps, mirroring the JAX flow:

1. **Start the profiler server** — `xp.start_server(<port>)` near the top of `__main__`.
2. **Bracket the trace** — `xp.start_trace(<log_dir>)` and `xp.stop_trace()` around the code to profile (typically the training loop).
3. **Add labels** — wrap logical regions (forward, backward, optimizer step, data prep) with `with xp.Trace('name'):` so the XProf trace viewer shows meaningful blocks instead of just low-level PyTorch/XLA internals.

Visualize with `xprof --port=8791 <log_dir>` and open `http://localhost:8791/`. For Google Cloud workloads the doc points at `cloud-diagnostics-xprof`.

## Key claims

- The canonical module is `torch_xla.debug.profiler`, conventionally imported as `xp`.
- `xp.start_server(<port>)` is separate from `xp.start_trace(...)`: the server enables on-demand capture from the XProf UI; the trace bracket does programmatic capture. In the complete example both are used in the same script.
- `xp.Trace('label')` is a **context manager**, and traces can be nested (e.g. an outer `train_step_...` block containing finer-grained blocks for data prep / forward / backward / optimizer). Custom labels show up as named blocks in the trace viewer.
- `torch_xla.step()` can be used around a training step (as shown in the example) — the labels live inside that `with` context.
- Example port for `xp.start_server` in the doc is **9012** (contrast: JAX doc uses 9999; TF uses 6009). Any free port works; must match whatever the UI capture dialog is pointed at for on-demand mode.
- Standalone viewer: `xprof --port=8791 /root/logs/` → `http://localhost:8791/`.

## Key data points

### Minimal programmatic capture

```python
import torch_xla.debug.profiler as xp

log_dir = '/root/logs/'
xp.start_trace(log_dir)
# ... training loop ...
xp.stop_trace()
```

### Full example (from `mnist_xla.py`)

```python
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

if __name__ == '__main__':
    server = xp.start_server(9012)      # on-demand capture endpoint
    xp.start_trace('/root/logs/')       # programmatic capture starts
    train_mnist()                       # annotated with xp.Trace(...) blocks
    xp.stop_trace()
```

### Custom labels (nested)

```python
def forward(self, x):
    with xp.Trace('forward'):
        # ... module forward ...
        return F.log_softmax(x, dim=1)

for batch_idx, (data, target) in enumerate(train_loader):
    with torch_xla.step():
        with xp.Trace('train_step_data_prep_and_forward'):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)

        with xp.Trace('train_step_loss_and_backward'):
            loss = loss_fn(output, target)
            loss.backward()

        with xp.Trace('train_step_optimizer_step_host'):
            optimizer.step()
```

### API summary

| Call | Purpose |
|---|---|
| `xp.start_server(port)` | Start in-process profiler gRPC server (enables on-demand capture from XProf UI) |
| `xp.start_trace(log_dir)` | Begin programmatic capture; writes `.xplane.pb` under `<log_dir>/plugins/profile/<session>/` |
| `xp.stop_trace()` | End programmatic capture; flushes trace files |
| `with xp.Trace(name):` | Add a named block to the trace timeline (nestable) |

### Ports used in examples

| Purpose | Doc's example |
|---|---|
| `xp.start_server` (in-process profiler gRPC) | 9012 |
| XProf UI / standalone viewer | 8791 |

## Techniques referenced

- **Programmatic capture** via `xp.start_trace` / `xp.stop_trace`.
- **In-process gRPC profiler server** (`xp.start_server(port)`) for on-demand capture from the XProf UI.
- **Custom trace annotations** via nested `xp.Trace(name)` context managers — essential because raw PyTorch/XLA traces are dominated by low-level internals and are hard to navigate without labels.
- **Step boundary demarcation** via `torch_xla.step()` (PyTorch/XLA's step-scoping context).
- **Standalone XProf UI** launch (`xprof --port=8791 <logdir>`).
- **`cloud-diagnostics-xprof`** recommended for GCE/GKE workloads.

## Gaps & caveats

- Doc does not show an **on-demand capture flow** end-to-end for PyTorch/XLA (only `start_server` is mentioned; the UI-side step is implicit — same as JAX: open the UI, hit "CAPTURE PROFILE", enter `localhost:<port>`).
- **No `block_until_ready` equivalent** is called out. PyTorch/XLA execution is likewise lazy; practically `torch_xla.step()` or `xm.mark_step()` acts as the sync boundary, and `xp.stop_trace` should follow a completed step. The doc does not spell this out.
- **No `ProfileOptions` guidance** — unlike the JAX doc, the PyTorch/XLA guide does not describe tracer levels, TPU trace mode, sparse-core tracing, or GPU CUPTI knobs. Options exist but the user has to find them elsewhere.
- **No multi-host TPU coordination guidance** (SPMD / multi-controller). Assumption is each host writes its own trace under the same logdir.
- **No env-var caveats** (`XLA_FLAGS`, `LIBTPU_INIT_ARGS`, `PJRT_DEVICE`, etc.) — orthogonal to profiling but commonly required for reproducible TPU runs.
- **No guidance on trace size** — nested `xp.Trace` labels multiply event count; no recommendation on scope or how many steps to profile (the TensorFlow doc recommends ≤10 steps).
- Port `9012` is an example; there's no default. Users must align the UI-side capture target port with whatever they passed to `start_server`.
- `xp.start_server` returns a server handle; the doc shows `server = xp.start_server(9012)` but never references `server` again. Lifecycle management (stopping it) is not discussed.
- Standard pitfall (not in doc): capturing without any `xp.Trace` annotations produces a trace dominated by `xla::` ops that's hard to read — always label at least per-step regions.

## Connections

- `profile-capture` — framework-agnostic capture concept.
- `pytorch-xla-trace` — `xp.start_trace`/`stop_trace`/`Trace` surface.
- `xprof-server` — in-process gRPC endpoint (`xp.start_server`).
- `torch-xla-step` — step-scoping context that typically wraps a trace-annotated training step.
- `custom-trace-annotations` — use of named context-manager blocks to label timelines.
- `mark-step-sync` — PyTorch/XLA's lazy-execution sync boundary (implicit, not covered in doc).

## See also

- [xprof](../codebases/xprof.md)
- [XProf docs: Capturing profiles](2026-xprof-capturing-profiles.md)
- [XProf docs: JAX profiling](2026-xprof-jax-profiling.md)
- [XProf docs: TensorFlow profiling](2026-xprof-tensorflow-profiling.md)
- [XProf docs: Docker deployment](2026-xprof-docker-deployment.md)
- [XProf docs: Kubernetes deployment](2026-xprof-kubernetes-deployment.md)

## Sources

- `raw/code/xprof/docs/pytorch_xla_profiling.md`
