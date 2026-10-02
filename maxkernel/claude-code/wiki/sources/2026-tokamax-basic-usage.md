---
title: "tokamax docs — basic usage"
type: source
tags: [docs, kernels, pallas, api, jax, autotune, benchmark, export, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

Canonical "how do I call tokamax" doc. Shows the end-to-end flow: import, call a kernel with an optional `implementation=` argument, JIT + grad, autotune to get a portable result, serialize with `jax.export`, and benchmark without JAX Python dispatch overhead skewing the numbers. For the TPU performance loop this is the doc a hypothesis author reads before proposing "swap op X for `tokamax.X`".

## Overview

The doc walks through a single toy function that interleaves two `tokamax.layer_norm` calls and two `tokamax.dot_product_attention` calls on an H100 GPU. It explains what happens when the `implementation=` argument is set to `None`, to `"triton"`, to `"mosaic"`, or to `"xla_chunked"`, then layers on four orthogonal concerns that every real user hits: **gradient evaluation** (the kernels are `jax.grad`-compatible), **autotuning** (one call returns an `AutotuningResult` context manager), **serialization** (kernels are JAX custom calls, so `jax.export` needs a blanket disable), and **benchmarking** (Python dispatch swamps kernel time, so tokamax ships its own timer).

This is the most practically load-bearing doc in the repo for a new user. Everything else is either a reference manual (supported_ops) or a deep dive (splash attention, autotuning, benchmarking).

## Key claims

1. **`implementation=None` is safe-by-default.** When `implementation` is unset tokamax picks the best implementation per-call, can pick different implementations for fwd and bwd passes, and will always succeed because it can fall back to `implementation='xla'`.
2. **`implementation="<name>"` is strict.** Setting an explicit name (e.g., `"mosaic"`, `"triton"`, `"xla_chunked"`) forces that backend and raises an exception if unsupported (e.g., FP64 inputs on Mosaic GPU, or pre-Hopper GPUs).
3. **Different implementations can be chosen for forward vs. backward.** The dispatcher is per-call, not per-op-instance.
4. **Gradient evaluation is transparent.** `jax.jit(jax.grad(loss))` Just Works on a function containing tokamax kernels.
5. **`tokamax.autotune(f, *args)` returns a serializable result.** The returned `AutotuningResult` can be used as a context manager (`with autotune_result: out = f_grad(x, scale)`) or dumped to/loaded from JSON (`.dumps()` / `.loads(...)`).
6. **Autotuning is non-deterministic.** Kernel-execution-time measurement is noisy, and different configs may produce different numerics, so the winning config can vary across runs. Serializing the autotune result is the recommended way to lock numerics across sessions.
7. **Users can register their own kernels for autotuning** by subclassing `tokamax.Op` and overriding `_get_autotuning_configs`. The autotuning pipeline then picks them up alongside built-in kernels.
8. **Kernels are JAX custom calls, not portable StableHLO.** `jax.export` rejects them by default; `tokamax.DISABLE_JAX_EXPORT_CHECKS` is the opt-in bypass. Exported functions lose StableHLO's device-independence.
9. **Tokamax makes two explicit serialization guarantees**:
   - A deserialized function runs on the exact device it was serialized for.
   - 6-month backward compatibility, matching JAX's custom-call compatibility policy.
10. **`jax.block_until_ready(f(x))` is an unreliable benchmark.** JAX Python overhead is often much larger than the actual kernel time, so naive timing measures dispatch, not the kernel. Tokamax ships `tokamax.benchmarking.{standardize_function, compile_benchmark}` to measure accelerator time only.
11. **On GPU, the CUPTI profiler is available** via `run(args, method='cupti')`. It instruments the kernel and adds a small overhead. Default `method=None` lets tokamax pick a method and works on both TPU and GPU.
12. **More iterations reduce noise.** `run(args, iterations=10)` trades wall-clock for precision.

## Key data points

### `implementation=` values demonstrated in this doc

| Value | Target | Behavior when unsupported |
|---|---|---|
| `None` (default) | Best-available per call | Falls back to XLA; never raises |
| `"xla"` | Portable XLA lowering | Always supported |
| `"xla_chunked"` | Memory-efficient chunked XLA ([Rabe & Staats 2021](https://arxiv.org/abs/2112.05682)) | Raises |
| `"triton"` | Pallas-Triton (GPU) | Raises (e.g., on TPU, on FP64) |
| `"mosaic"` | Pallas-Mosaic GPU (H100/B200) | Raises (e.g., FP64, pre-Hopper) |

TPU callers typically omit `implementation=` and let tokamax pick the Mosaic-TPU backend (attention/ragged-dot/CE loss) or fall back to XLA (GLU, LayerNorm).

### Canonical snippet (GPU example from the doc, abbreviated)

```python
def loss(x, scale):
  x = tokamax.layer_norm(x, scale=scale, offset=None, implementation="triton")
  x = tokamax.dot_product_attention(x, x, x, implementation="xla_chunked")
  x = tokamax.layer_norm(x, scale=scale, offset=None, implementation=None)
  x = tokamax.dot_product_attention(x, x, x, implementation="mosaic")
  return jnp.sum(x)

f_grad = jax.jit(jax.grad(loss))
```

### Input shapes the doc uses

```python
channels, seq_len, batch_size, num_heads = (64, 2048, 32, 16)
# scale : (channels,) fp32
# x     : (batch_size, seq_len, num_heads, channels) bf16
```

This is a mid-sized attention shape (batch 32 × seq 2048 × heads 16 × head-dim 64). Not a production shape — a demo shape.

### Autotune API surface

| Call | Returns | Purpose |
|---|---|---|
| `tokamax.autotune(f, *args)` | `AutotuningResult` | Sweep all tokamax ops in `f`. |
| `with autotune_result:` | context manager | Apply the tuned configs inside the block. |
| `autotune_result.dumps()` | `str` (JSON) | Serialize for reuse. |
| `AutotuningResult.loads(s)` | `AutotuningResult` | Reload. |
| Subclass `tokamax.Op`, override `_get_autotuning_configs` | — | Add user kernels to the tune pipeline. |

### Benchmarking API surface (per this doc)

```python
f_std, args = tokamax.benchmarking.standardize_function(f, kwargs={'x': x, 'scale': scale})
run = tokamax.benchmarking.compile_benchmark(f_std, args)
bench: tokamax.benchmarking.BenchmarkData = run(args)
```

Options: `method='cupti'` (GPU, instrumented); `method=None` (auto, TPU+GPU); `iterations=<int>` (more iterations → less noise).

### Export API

```python
f_grad_exported = export.export(
    f_grad,
    disabled_checks=tokamax.DISABLE_JAX_EXPORT_CHECKS,
)(
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    jax.ShapeDtypeStruct(scale.shape, scale.dtype),
)
```

## Techniques referenced

- **Implementation dispatch at the op level** (`implementation=` kwarg) — how tokamax binds a single API symbol to multiple backends and degrades gracefully to XLA.
- **Autotuning via context manager** — a pattern where the tuned configs live in a dataclass that monkey-patches lookups for the duration of a `with` block.
- **StableHLO with custom-call escape hatches** — using JAX custom calls (Pallas) inside exported StableHLO at the cost of device portability.
- **Accelerator-only timing** (CUPTI on GPU; tokamax's own timing on TPU) — isolates kernel wall-clock from Python + dispatcher overhead.

## Gaps & caveats

- **The example is GPU-only.** Every `implementation=` value shown (`"triton"`, `"xla_chunked"`, `"mosaic"`) refers to GPU backends. There is no TPU example in this doc. For TPU the relevant `implementation=` value is `"mosaic"` (which resolves to Pallas-Mosaic-TPU → splash attention), but this is not stated here.
- **`tokamax.benchmarking` namespace vs. top-level `tokamax`.** This doc imports benchmarking helpers from `tokamax.benchmarking.*`; the benchmarking doc imports the same helpers from `tokamax.*` (e.g., `tokamax.standardize_function`, `tokamax.benchmark`). Both appear to be valid — the top-level symbols are re-exports — but the two docs use different styles. Not a blocker, worth flagging.
- **Autotuning non-determinism is called out but not quantified.** The doc says "different configs chosen during autotuning can lead to different numerics" without giving a sense of magnitude. For a model-under-optimization, this means a hypothesis that autotunes must lock the result (via `.dumps()`) before running the baseline-vs-candidate comparison, or the comparison is not apples-to-apples.
- **No discussion of cache semantics.** `tokamax.autotune` consults an on-disk cache by default (`ignore_cache=False`). The doc does not mention this. The autotuning doc itself is a stub (see [tokamax autotuning](2026-tokamax-autotuning.md)).
- **`compile_benchmark` + `run(args)` pattern vs. `tokamax.benchmark(f_std, args)` pattern.** This doc uses the compile-then-run form; the benchmarking doc uses the single-call form. Both exist; the one-call form is preferred for most users.
- **Export guarantee (1): exact device reuse.** This is a strong claim — a serialized tokamax function is bound to a specific device family and device generation. That is more restrictive than standard StableHLO; worth recording as a deployment caveat.

## Connections

Concept slugs this source informs:

- `tokamax-api-dispatch` — the `implementation=` semantics.
- `autotuning` — `tokamax.autotune` is the practical surface.
- `benchmarking` — kernel-only timing on TPU and GPU.
- `pallas` — all explicit `implementation=` values point at Pallas backends.
- `jax-export` — custom-call export gotchas.
- `flash-attention` / `splash-attention` — the two attention backends reached via `implementation="xla_chunked"` and `implementation="mosaic"`.
- `stablehlo` — the export format and its portability constraints.

## See also

- [tokamax](../codebases/tokamax.md) — full codebase reference; per-backend files.
- [tokamax supported ops](2026-tokamax-supported-ops.md) — which kernels exist on which platform.
- [tokamax splash attention](2026-tokamax-splash-attention.md) — the TPU backend reached by `implementation="mosaic"` on TPU.
- [tokamax autotuning](2026-tokamax-autotuning.md) — `AutotuningResult` semantics in depth (stub at time of ingest).
- [tokamax benchmarking](2026-tokamax-benchmarking.md) — `tokamax.benchmark` method and mode options.

## Sources

- [`raw/code/tokamax/docs/basic_usage.md`](../../raw/code/tokamax/docs/basic_usage.md)
