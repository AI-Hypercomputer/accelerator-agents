---
title: "tokamax docs — benchmarking"
type: source
tags: [docs, kernels, benchmark, xprof, cupti, timing, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

How tokamax isolates *accelerator* time from JAX/Python dispatch overhead. Tokamax ships `standardize_function` (to build a clean jittable form) and `benchmark` (to actually time it), plus a `method=` kwarg selecting the underlying profiler — typically XProf on TPU and XProf-CUPTI or raw CUPTI on GPU — and a `mode=` kwarg selecting which pass to time (fwd / fwd+residuals / vjp / fwd+vjp). For the TPU performance loop this is the doc that defines what "kernel time" means.

## Overview

The doc walks through the two-line recipe (`standardize_function` → `benchmark`) and then layers on four orthogonal knobs: **iteration count** (noise vs. thermal throttling), **timing method** (`xprof_hermetic` recommended for TPU), **data distribution** (random init by default; matters because data patterns affect kernel power/thermal), and **mode** (forward-only, forward+residuals, VJP-only, or full fwd+VJP).

The key claim — and the reason this library ships its own timer at all — is that **`jax.block_until_ready(f(x))` is not an accurate benchmark on either TPU or GPU** because JAX's Python-side dispatch overhead can dwarf the actual kernel time. XProf on TPU and CUPTI on GPU both sample directly from the hardware, so the reported time reflects the kernel, not the dispatcher.

## Key claims

1. **`jax.block_until_ready(f(x))` is unreliable.** JAX Python overhead is often much larger than accelerator kernel time, so naive timing measures dispatch, not the kernel.
2. **Two-step recipe**: `standardize_function(f, kwargs=...)` builds `(f_std, args)`; `benchmark(f_std, args)` returns a `BenchmarkData`.
3. **`standardize_function` normalises calling conventions.** Non-array arguments (e.g., strings) are baked in; abstract `jax.ShapeDtypeStruct` arguments are materialised with random data; the returned function has a single `args` parameter (a list of concrete arrays) that is cleanly jittable with no static args.
4. **More iterations reduce noise** — `benchmark(f_std, args, iterations=num_iters)`.
5. **Too many iterations in a short period can trigger thermal throttling**, especially for compute-heavy kernels. The doc's suggested approach is "small iteration count per experiment, multiple spaced-out experiments".
6. **Method selection via `method=`.** For **TPU**, tokamax *strongly recommends* `method=xprof_hermetic`, which invokes the XProf profiler and measures on the hardware with almost no instrumentation overhead due to full-stack support (hardware + compiler).
7. **GPU supports the same method** — `xprof_hermetic` on GPU uses NVIDIA's CUPTI APIs under the hood; or the user may directly invoke `method=cupti`. Either GPU method imposes some variable overhead, typically up to 5%.
8. **Data distribution affects measured time.** [Prior work by Horace He](https://www.thonking.ai/p/strangely-matrix-multiplications) shows that matmul performance varies with data distribution due to hardware-level power/thermal effects. `standardize_function` therefore initialises real-valued inputs randomly to approximate training-time distributions.
9. **`mode=` selects which pass is timed.**
   - `forward` — forward only.
   - `forward_res` — forward and compute residuals.
   - `vjp` — VJP only.
   - `forward_and_vjp` — full forward + VJP.
10. **`mode=vjp` can OOM.** Benchmarking VJP-only forces the forward pass to be computed outside the standardized function, with all intermediates baked into the HLO, which remains resident in HBM.
11. **Random seeding is deterministic.** `standardize_function` seeds its random init so that successive benchmarks on the same shapes see the same data — essential for run-to-run comparability.

## Key data points

### API surface (as described in this doc)

```python
# top-level import style (this doc)
f_std, args = tokamax.standardize_function(f, kwargs={'x': x})
bench: tokamax.BenchmarkData = tokamax.benchmark(f_std, args)
```

Equivalently, the basic-usage doc uses the `tokamax.benchmarking.*` namespace — both forms exist.

### `benchmark(f_std, args, ...)` keyword arguments

| Kwarg | Type | Default | Purpose |
|---|---|---|---|
| `iterations` | int | (impl default) | Number of kernel invocations to average over. More → less noise; too many → thermal throttling. |
| `method` | str | `None` (auto) | Timer method (`xprof_hermetic`, `cupti`, ...). |
| `mode` | str | see `standardize_function` | Which pass is timed (fwd / fwd_res / vjp / fwd_and_vjp). |

### `method=` options

| Method | Platform | Overhead | Notes |
|---|---|---|---|
| `xprof_hermetic` | **TPU** (recommended) + GPU | Near-zero on TPU; ~up to 5% on GPU | Uses XProf; hardware-backed timing. On GPU, XProf reaches through to NVIDIA CUPTI. |
| `cupti` | GPU only | ~up to 5% | Direct CUPTI invocation (NVIDIA's profiling API). |
| `None` (default) | TPU + GPU | — | Tokamax picks. |

### `mode=` options (on `standardize_function`)

| Mode | What is timed | Caveat |
|---|---|---|
| `forward` | Forward pass only | — |
| `forward_res` | Forward + residuals for bwd | Residuals are part of what the bwd pass depends on |
| `vjp` | VJP only | **May OOM** — forward intermediates materialised in HBM |
| `forward_and_vjp` | Full fwd + VJP | The recommended end-to-end training microbench mode |

### Recipe for minimising noise

From the doc:

1. Use `xprof_hermetic` (TPU) or `xprof_hermetic` / `cupti` (GPU).
2. Keep `iterations` modest per run to avoid thermal throttling.
3. Run multiple spaced-out experiments rather than one long one.
4. Random init via `standardize_function` is representative of real training distributions — do not benchmark against all-zero or all-one arrays (they will mismeasure due to power/thermal effects).

## Techniques referenced

- **Hardware-backed timing** (XProf on TPU; CUPTI on GPU) — reads timestamps from on-chip profiling hardware rather than using host-side wall clocks. Sidesteps JAX/Python dispatch overhead.
- **Standardized function form** — canonicalise the callable so that jitting is trivial and initialization is reproducible. A pattern for any kernel-timing library.
- **Thermal-aware iteration design** — a methodological point about accelerator benchmarking that applies to any compute-bound microbench: long steady-state loops can shift the clock/power state and change the answer.
- **Data-distribution-aware init** — acknowledging that matmul time depends on operand distributions (power/thermal; see Horace He's post).
- **VJP-only benchmarking pitfall** — forward intermediates are resident in HBM when the timed function is just the VJP, making `mode=vjp` an OOM risk on production-scale shapes.

## Gaps & caveats

- **The `tokamax.benchmarking.*` vs. `tokamax.*` namespace inconsistency.** Basic-usage uses `tokamax.benchmarking.standardize_function` / `tokamax.benchmarking.compile_benchmark` / `run(args)`; this doc uses `tokamax.standardize_function` / `tokamax.benchmark(f_std, args)`. Both work (top-level re-exports exist); the `compile_benchmark` + `run` form is a lower-level variant. The docs do not explicitly reconcile the two styles.
- **Iteration default not stated.** The doc talks about increasing iterations for less noise but does not say what the default is.
- **"Thermal throttling" is a qualitative warning.** No numbers (e.g., "after N consecutive runs on v6e, step time rises by X%"). Users have to discover thresholds empirically.
- **`method=xprof_hermetic` requires XProf to be available.** The doc does not discuss what happens if XProf is not installed or not reachable — presumably it falls back to `method=None`'s pick, but unstated.
- **`method=None` → "tokamax chooses"** — but the doc does not say *what* it chooses on each platform. Likely `xprof_hermetic` on TPU and something simpler on GPU, but users cannot confirm without reading the code.
- **No discussion of multi-device / multi-host benchmarking.** All examples are single-process. For SPMD / sharded ops the benchmarking story is likely different.
- **No discussion of warmup.** Compilation takes the first call. The doc does not say whether `benchmark` excludes the compile call from the measurement, though standard practice is yes.
- **VJP OOM caveat is correct but under-specified.** No advice on *how* to benchmark VJP-only for a large shape (e.g., use fwd_and_vjp instead and subtract fwd-only? The doc does not say).
- **Data-distribution claim cites a blog post, not a paper.** [Horace He's thonking.ai post on strangely matrix multiplications](https://www.thonking.ai/p/strangely-matrix-multiplications) is the cited prior work. Valid, but informal.

## Connections

Concept slugs this source informs:

- `benchmarking` — this doc anchors the concept page.
- `xprof` — the recommended TPU timing method.
- `cupti` — the GPU timing method.
- `thermal-throttling` — relevant to any microbench on a production accelerator.
- `data-distribution-dependent-perf` — matmul time varies with operand distribution.
- `standardize-function` — the canonical-form pattern.
- `benchmarking-modes` — fwd / fwd_res / vjp / fwd_and_vjp.

### Practical notes for the autoresearch loop

- When running a TPU experiment that compares tokamax vs. XLA for a single op, use `method=xprof_hermetic` with `mode=forward_and_vjp` and multiple spaced-out iteration batches. This matches the doc's recommended methodology.
- Do **not** report tokamax-vs-XLA numbers from `jax.block_until_ready(...)` — they measure dispatch, not the kernel.
- Do **not** use `mode=vjp` on full-model shapes — OOM risk.
- When comparing configs across runs, lock the autotune result first (see [tokamax autotuning](2026-tokamax-autotuning.md)) — otherwise the non-determinism of autotuning + the noise of benchmarking combine and make the comparison meaningless.

## See also

- [tokamax](../codebases/tokamax.md) — `_src/benchmarking.py` is the implementation.
- [tokamax basic usage](2026-tokamax-basic-usage.md) — uses the same API via the `tokamax.benchmarking.*` namespace.
- [tokamax autotuning](2026-tokamax-autotuning.md) — benchmarking is how the autotuner picks winners; the same noise-reduction recipe applies.
- [tokamax splash attention](2026-tokamax-splash-attention.md) — the TPU attention kernel most likely to be benchmarked here.
- [tokamax supported ops](2026-tokamax-supported-ops.md) — which kernels even exist to benchmark.

## Sources

- [`raw/code/tokamax/docs/benchmarking.md`](../../raw/code/tokamax/docs/benchmarking.md)
