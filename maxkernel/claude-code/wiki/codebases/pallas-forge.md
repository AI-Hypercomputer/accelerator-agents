---
title: "pallas-forge"
type: codebase
tags: [pallas, mosaic, tpu, autotuning, roofline, rmsnorm, swiglu, matmul, kernels, teaching]
repo: linhkid/pallas-forge
commit: 090510b7b36796a31f7bacee04c31d7e1909cf76
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [pallas-forge overview](pallas-forge/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Lightweight auto-tuning framework for [Pallas](../concepts/pallas-kernel.md) kernels on TPU. Ships three reference kernels (tiled matmul, fused RMSNorm+residual, fused SwiGLU/GeGLU), a kernel-agnostic grid/random-search tuner with proper warmup + statistical timing, and a roofline + XProf-trace integration. The library's stated thesis is explicitly honest: it helps answer *"is my custom kernel actually beating XLA?"* — and in two of its three reference kernels on v5e, the answer is **no**. Already used once in this wiki's autoresearch loop (gemma4 [exp 20](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp20-pallas-forge-rmsnorm-rejected.md)) and found to lack backward passes, making it unusable in training as-is.

## Overview

pallas-forge is a **teaching-quality** library — the author names it as such in kernel docstrings — packaged as three demonstrations of when Pallas fusion is (or isn't) worth the complexity, plus a reusable auto-tuner harness for anyone writing their own kernels. Three things it does well: (1) gives a clean canonical 3D-grid Pallas matmul with the TPU alignment rules documented; (2) provides the single cleanest `tune()` API I've seen for block-size sweeps with roofline output; (3) publishes the *negative* result that custom matmul and SwiGLU are not wins against XLA's compiler, with measured numbers.

It is **not** production-ready for training — the three kernels expose forward-only `jax.jit` wrappers. There are no `jax.custom_vjp` definitions, so `jax.grad` cannot differentiate through them. In the gemma4 program exp 20 this manifested as `"Linearization failed to produce known values"` at the first `jax.value_and_grad` call; the experiment was parked. This is **the most important fact** about pallas-forge for this wiki: its kernels are not drop-in for any training hypothesis.

The companion finding from gemma4 [exp 33](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp33-pallas-rmsnorm-rejected.md) — that a custom Pallas RMSNorm with custom_vjp regressed step time by 8.1% because XLA already fused RMSNorm into neighboring matmuls — further narrows pallas-forge's usefulness: the one kernel it "wins" on (RMSNorm, 3.44× in isolation) is the one XLA already handles well in the real graph.

## Architecture

```
pallas_forge/
  __init__.py              ← top-level re-exports: tiled_matmul, fused_rmsnorm_residual, fused_swiglu, fused_geglu, tune, TuneConfig
  _compat.py               ← pallas_call_compat shim: CPU interpret-mode vs TPU/GPU
  _version.py
  kernels/
    matmul.py              ← 3D-grid Pallas matmul; alignment-rule documentation
    rmsnorm.py             ← 1D-grid Pallas RMSNorm+residual; TOKENS_PER_TILE=8 sublane rule
    swiglu.py              ← 2D-grid fused SwiGLU/GeGLU (gate+up+activation in one pass)
    _utils.py              ← pad_to_multiple / unpad helpers for non-aligned shapes
  tune/
    __init__.py            ← tune() entry point
    config.py              ← TuneConfig search space (dict / YAML)
    search.py              ← GridSearch, RandomSearch strategies
    runner.py              ← BenchmarkRunner: n_warmup + n_repeat + block_until_ready timing
    report.py              ← TuneReport: to_csv, to_json, heatmap(), best()
    trace.py               ← XProf trace capture for top-N configs
  profile/
    roofline.py            ← roofline_chart(); TPU_SPECS table
    analysis.py            ← utilization analysis
benchmarks/                ← bench_matmul.py, bench_rmsnorm.py, bench_swiglu.py, run_all.py (require TPU)
notebooks/                 ← 5 Colab-ready .ipynb walkthroughs
tests/                     ← 54 tests; run on CPU via Pallas interpret mode
```

Three layer boundaries the wiki cares about:

1. **Kernel boundary (`pallas_forge/kernels/`).** Three forward-only Pallas kernels, each wrapped in `jax.jit(static_argnames=('block_m', 'block_k', 'block_n', ...))`. No custom_vjp registered — this is the binding constraint for any training use.
2. **Tuner boundary (`pallas_forge/tune/`).** Kernel-agnostic: you pass `kernel_fn`, `input_fn(cfg)`, a `TuneConfig`, and get back a `TuneReport` with heatmaps + CSV + per-config `BenchmarkResult` (median/mean/std/min/max, optional TFLOPS + bandwidth).
3. **Roofline boundary (`pallas_forge/profile/`).** `roofline_chart(results, peak_tflops, peak_bandwidth_gb_s, ...)` + `TPU_SPECS = {'v4': {...}, 'v5e': {...}, 'v5p': {...}}` hardware-preset table. **No v6e entry** as of commit `090510b`.

## Key abstractions

| Abstraction | What it is | Where | Notes |
|---|---|---|---|
| `tiled_matmul(x, w, *, block_m, block_k, block_n, num_stages)` | 3D-grid Pallas matmul; K-axis iterated by grid; fp32 accumulator in `o_ref` | `kernels/matmul.py:51` | Alignment: `block_k`, `block_n` mult-of-128; `block_m` mult-of-8. No backward. |
| `fused_rmsnorm_residual(x, residual, weight, *, eps, block_size)` | 1D-grid fused `new_residual = x + residual; output = (new_residual / rms(new_residual)) * weight`; returns `(output, new_residual)` | `kernels/rmsnorm.py:68` | `TOKENS_PER_TILE=8` hard-coded sublane-alignment constant. No backward. |
| `fused_swiglu(x, w_gate, w_up, *, block_m, block_n, num_stages)` | 2D-grid fused `silu(x @ w_gate) * (x @ w_up)` | `kernels/swiglu.py` | `fused_geglu` variant uses `jax.nn.gelu`. No backward. Lost 0.65× vs XLA on v5e. |
| `tune(kernel_fn, input_fn, config, *, strategy, n_warmup, n_repeat, flops_fn, bytes_fn, top_n_traces, trace_output_dir, verbose)` | Auto-tune entry point | `tune/__init__.py:34` | `strategy` ∈ `{'grid', 'random'}` or `SearchStrategy` instance; `top_n_traces>0` captures XProf traces. |
| `TuneConfig` | Search-space container; `from_dict`, `from_yaml` factories | `tune/config.py` | YAML-loadable → reproducible tuning. |
| `BenchmarkResult(config, median_ms, mean_ms, std_ms, min_ms, max_ms, all_times_ms, tflops, bandwidth_gb_s)` | Per-config measurement dataclass | `tune/runner.py:20` | `tflops` / `bandwidth_gb_s` populated only if `flops_fn` / `bytes_fn` passed to `tune`. |
| `BenchmarkRunner(kernel_fn, input_fn, *, n_warmup=5, n_repeat=20, flops_fn, bytes_fn)` | Timing harness | `tune/runner.py:56` | Uses `jax.block_until_ready` + `time.perf_counter`; measures per-call, not per-batch. |
| `roofline_chart(results, peak_tflops, peak_bandwidth_gb_s, *, save_path, ...)` | Matplotlib roofline generator | `profile/roofline.py:15` | Requires `tflops` + `bandwidth_gb_s` on results. |
| `TPU_SPECS` dict | Hardware preset table | `profile/roofline.py` (also `profile/__init__.py`) | **v4, v5e, v5p only — no v6e entry.** |

## Entry points

```bash
# install
pip install -e ".[dev,viz]"              # CPU development (interpret mode)
pip install -e ".[all,tpu]"              # on a Linux TPU VM

# run tests (CPU only — 54 tests, all pass in interpret mode)
pytest tests/ -v

# benchmarks (require TPU)
python benchmarks/bench_matmul.py
python benchmarks/bench_rmsnorm.py
python benchmarks/bench_swiglu.py
python benchmarks/run_all.py             # all three, writes results/*.csv, results/*.png

# notebook walkthroughs (Colab-ready)
jupyter notebook notebooks/01_tiled_matmul.ipynb
jupyter notebook notebooks/05_reproduce_figures.ipynb   # regenerates all README figures
```

Programmatic entry:

```python
from pallas_forge import tiled_matmul, fused_rmsnorm_residual, fused_swiglu
from pallas_forge.tune import tune, TuneConfig
from pallas_forge.profile import roofline_chart, TPU_SPECS

report = tune(kernel_fn, input_fn, TuneConfig.from_dict({"block_m": [64,128,256], "block_n": [64,128,256]}))
report.to_csv("results.csv")
report.heatmap("block_m", "block_n", save_path="heatmap.png")
best = report.best(1)[0]
```

## Dependencies

- `jax` + `jax.experimental.pallas` (Pallas must be present; on CPU uses `interpret=True`).
- `libtpu` (Linux only) for TPU execution.
- `matplotlib`, `seaborn` (optional, `[viz]` extra) for heatmaps + roofline charts.
- `pyyaml` for `TuneConfig.from_yaml`.
- `tensorboard` / xprof plugin (optional) to view traces captured by `trace.py`.

Python 3.11+ per `pyproject.toml` (`conda create -n pallas-forge python=3.11` in README).

## Notable files

### `pallas_forge/kernels/matmul.py` (`tiled_matmul`)
Canonical 3D-grid TPU Pallas matmul. **Key structural notes** (from docstring lines 11–17):
- `block_k` must be mult-of-128 (last dim of `x` tile).
- `block_n` must be mult-of-128 (last dim of `w` tile + output tile).
- `block_m` must be mult-of-8 (sublane).
- Accumulator is fp32 across K iterations — `preferred_element_type=jnp.float32` in the inner `jnp.dot` — cast to output dtype happens once, outside the kernel.
- `pl.when(pl.program_id(2) == 0)` is the accumulator-zeroing guard pattern.

Used as the **reference for the canonical TPU matmul pattern** — cleanest example in this wiki of the 3D-grid + fp32-accumulator + K-axis-iteration convention.

### `pallas_forge/kernels/rmsnorm.py` (`fused_rmsnorm_residual`)
1D-grid RMSNorm + residual kernel. Signature: `(x, residual, weight) -> (output, new_residual)`. Key detail: **`TOKENS_PER_TILE = 8`** is hard-coded at `rmsnorm.py:31` to satisfy the second-to-last-dim sublane alignment. The wrapper pads the token axis to a multiple of 8 using `pallas_forge.kernels._utils.pad_to_multiple` and slices the result back.

This is the kernel with the **+3.44× vs XLA** v5e measurement — the library's only unambiguous win. **Gemma 4 exp 20** attempted to swap it into the training loop; exp 20 confirmed **no custom_vjp ⇒ backward fails**. **Gemma 4 exp 33** then implemented a Pallas RMSNorm *with* custom_vjp and found it regressed step time by 8.1% because XLA already fused RMSNorm with adjacent matmuls in the compiled graph. Both results narrow the usefulness of the `3.44×` isolated-microbench number in a real training graph.

### `pallas_forge/kernels/swiglu.py` (`fused_swiglu`, `fused_geglu`)
2D-grid fused activation. Computes `activation(x @ w_gate) * (x @ w_up)` in a single pass. The fusion saves **two HBM round-trips** on the gate and up intermediates. Measured **0.65× vs XLA on v5e** — XLA already fuses this pattern effectively. Same custom_vjp gap as the others.

### `pallas_forge/tune/runner.py` (`BenchmarkRunner`)
Timing harness. Defaults: `n_warmup=5`, `n_repeat=20`. Uses `jax.block_until_ready()` after every call, not just on the last one — confirms per-call measurement. Optional `flops_fn(cfg) -> int` and `bytes_fn(cfg) -> int` callbacks populate `BenchmarkResult.tflops` / `.bandwidth_gb_s` for roofline plots.

### `pallas_forge/tune/trace.py` + `tune/__init__.py:113`
`capture_top_n_traces(results, kernel_fn, input_fn, n, output_dir)` re-runs the top-N configs under `jax.profiler.trace(output_dir)` — integrates cleanly with the wiki's [xprof](xprof.md) viewer and [xprof-mcp](xprof-mcp.md) tooling.

### `pallas_forge/profile/roofline.py` + `TPU_SPECS`
`TPU_SPECS` hardware-preset table (from README, transcribed from the code):

| Generation | Peak bf16 TFLOPS | Peak HBM GB/s | VMEM (MB) |
|---|---|---|---|
| v4 | 275 | 1200 | 32 |
| v5e | 197 | 819 | 32 |
| v5p | 459 | 2765 | 95 |

**Missing: v6e.** Any hypothesis that uses `roofline_chart` on v6e profiles must supply peak-TFLOPS and peak-HBM directly (or extend `TPU_SPECS` locally).

### `benchmarks/run_all.py` + `notebooks/05_reproduce_figures.ipynb`
The authoritative reproduction path for the README's measured numbers. Running these on v6e (the wiki's current hardware) would produce the missing data point.

## Performance-relevant surfaces

Concrete anchors for future hypotheses, grounded in file:line:

### 1. No custom_vjp — forward-only; **not usable in training as-is**
Every kernel wrapper in `pallas_forge/kernels/*.py` is `@jax.jit(static_argnames=(...))` around a forward-only `pl.pallas_call`. None of them register a `jax.custom_vjp`. **Consequence:** any training hypothesis that tries to swap a pallas-forge kernel into a module's `forward` will crash at `jax.value_and_grad` time with `"Linearization failed to produce known values"`. Confirmed by gemma4 [exp 20](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp20-pallas-forge-rmsnorm-rejected.md). **Fix paths:** (a) wrap in a user-provided `jax.custom_vjp` with manual backward; (b) use [tokamax](tokamax.md) instead (its `layer_norm` / `rms_norm` paths expose autodiff-capable entries). (b) is cheaper.

### 2. v5e-only measured numbers
All README numbers (3.44× RMSNorm, 0.77× MatMul, 0.65× SwiGLU) are **Colab TPU v5e**. No v6e numbers are published. Gemma 4 exp 33's finding that XLA already fuses RMSNorm on v6e-4 suggests the 3.44× gap may be generation-specific or graph-context-specific. **Do not cite the 3.44× number as a v6e prediction.**

### 3. Isolated-microbench ≠ in-graph gain (generalizable lesson)
The pallas-forge measurement methodology is **isolated microbenchmark**: the kernel is timed standalone against an XLA equivalent. This is the correct methodology for the library's stated purpose ("is my kernel beating XLA?") but **systematically over-attributes benefit** relative to use inside a larger graph where XLA may already be fusing the target pattern into neighbors. Gemma 4 exp 33 is the direct demonstration — a 3.44× isolated win became an 8.1% in-graph loss. Every future "swap kernel X into model Y" hypothesis needs both measurements, not just the isolated one.

### 4. Block-size sweep methodology
`tune()` + `TuneReport.heatmap(param_a, param_b, save_path=...)` is the cleanest pattern in this wiki for a 2D block-size sweep. The README reports a **3.56× matmul spread** (1.14 ms → 0.32 ms across configs at 2048² bf16 on v5e) — evidence that block-size choice alone is a meaningful knob, independent of the compute pattern.

### 5. Alignment-rule documentation (`block_k` mult-of-128, `block_m` mult-of-8, `TOKENS_PER_TILE=8`)
These are the canonical TPU-Pallas alignment rules. Documented explicitly at `kernels/matmul.py:11-17` and `kernels/rmsnorm.py:9-17`. Any TPU Pallas kernel this wiki writes should start from these numbers rather than re-deriving them.

### 6. Canonical 3D-grid matmul + fp32-accumulator pattern
`kernels/matmul.py:32-48` is the cleanest example we have of:
- 3D grid `(grid_m, grid_n, k_tiles)`.
- K-axis iterated by the grid (not the kernel body).
- Accumulator in `o_ref` carried across K iterations.
- fp32 accumulator dtype via `preferred_element_type=jnp.float32` with a single cast to output dtype outside the kernel.
- `pl.when(pl.program_id(2) == 0)` zero-init guard.

Use this as the template for any future Pallas matmul variant.

### 7. `TPU_SPECS` table is authoritative for v4/v5e/v5p
`profile/roofline.py`'s hardware preset. Correct for peak-TFLOPS and peak-HBM-bandwidth within its generations. Extend rather than duplicate when v6e gets added.

### 8. BenchmarkRunner timing discipline
`n_warmup=5`, `n_repeat=20`, `jax.block_until_ready` after every call. This is the **timing methodology** used in the exp-20 / exp-33 comparisons. Any kernel microbenchmark in this wiki should match at least these defaults for comparability.

### 9. XProf-trace top-N capture
`tune(..., top_n_traces=3, trace_output_dir="./xprof_traces")` captures `jax.profiler.trace` output for the three fastest configs — a ready-made hook into the wiki's [xprof-mcp](xprof-mcp.md) analysis pipeline.

### 10. CPU interpret-mode testing
`pallas_call_compat` in `_compat.py` dispatches to `interpret=True` on CPU, so all 54 tests run without a TPU. This is the only codebase in the wiki with **CPU correctness coverage for Pallas kernels** — useful pattern for any Pallas work this program writes.

## Connections

Already referenced by this wiki's optimization loop:
- [gemma4 exp 20 — pallas-forge RMSNorm (rejected)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp20-pallas-forge-rmsnorm-rejected.md) — confirmed no-custom_vjp gap in training.
- [gemma4 exp 33 — Pallas RMSNorm (rejected)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp33-pallas-rmsnorm-rejected.md) — implemented backward; XLA already fuses; −8.1%.
- [gemma4 v6e-4 optimization ceiling analysis](../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — carries forward the "Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion" lesson that exp 33 produced.

Related codebases:
- [tokamax](tokamax.md) — the **training-ready** Pallas kernel library; has RMSNorm / layer_norm / splash-attention / ragged_dot with autodiff. First-line alternative to pallas-forge for any training-context kernel swap.
- [xprof](xprof.md) / [xprof-mcp](xprof-mcp.md) — the trace-capture integration target for `tune(..., top_n_traces=N)`.

## See also

- [pallas-kernel](../concepts/pallas-kernel.md) — the DSL these kernels are written in.
- [mosaic-kernel](../concepts/mosaic-kernel.md) — the TPU lowering backend.
- [autotuning](../concepts/autotuning.md) — pattern implemented here by `tune()`.
- [roofline-model](../concepts/roofline-model.md) — what `roofline_chart` produces.
- [layer-norm](../concepts/layer-norm.md) — the kernel category `fused_rmsnorm_residual` sits in.
- [gated-linear-unit](../concepts/gated-linear-unit.md) — SwiGLU / GeGLU category.
- [xla-fusion](../concepts/xla-fusion.md) — the competitor that wins on matmul and SwiGLU on v5e.
- [attention-block-sizes](../concepts/attention-block-sizes.md) — analogous tuning surface for flash/splash attention.
- [dimension-alignment](../concepts/dimension-alignment.md) — the mult-of-8 / mult-of-128 rules that `matmul.py:11-17` documents.

## Sources

- `raw/code/pallas-forge/README.md`
- `raw/code/pallas-forge/pallas_forge/__init__.py`
- `raw/code/pallas-forge/pallas_forge/kernels/matmul.py`
- `raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py`
- `raw/code/pallas-forge/pallas_forge/kernels/swiglu.py`
- `raw/code/pallas-forge/pallas_forge/tune/__init__.py`
- `raw/code/pallas-forge/pallas_forge/tune/runner.py`
- `raw/code/pallas-forge/pallas_forge/profile/roofline.py`
- `raw/code/pallas-forge/CHANGELOG.md`
- `raw/code/pallas-forge/pyproject.toml`
- Upstream: <https://github.com/linhkid/pallas-forge>
