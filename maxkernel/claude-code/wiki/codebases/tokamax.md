---
title: "tokamax — custom accelerator kernel library (TPU + GPU)"
type: codebase
tags: [kernels, pallas, library, attention, glu, layer-norm, rms-norm, ragged-dot, moe, cross-entropy]
commit: 54bdd955b439e8361fc1a1a2e20005ba2f1e8b37
created: 2026-04-22
updated: 2026-04-22
---

> **Grounded front door → [tokamax overview](tokamax/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Tokamax is a DeepMind/OpenXLA library of custom Pallas kernels for JAX, targeting both NVIDIA GPUs (Triton / Mosaic GPU) and Google TPUs (Mosaic TPU). Each public entry point (`dot_product_attention`, `gated_linear_unit`, `layer_norm`, `ragged_dot`, `linear_softmax_cross_entropy_loss`, `triangle_multiplication`) is a drop-in replacement for an XLA-lowered op; the implementation is selected at call time via an `implementation=` argument and tuned via `tokamax.autotune`. Every such kernel is a direct optimization candidate for a TPU model that currently runs the XLA reference lowering.

## Overview

- Every op has a common shape: an `api.py` wrapper that chooses an implementation, a `base.py` with the XLA reference (always available), and one or more Pallas backends (`pallas_triton.py`, `pallas_mosaic_gpu.py`, `pallas_mosaic_tpu.py`).
- Backends are registered lazily at import: missing GPU toolchains fall through to whatever is importable, so a TPU job naturally gets the `mosaic_tpu` + `xla` pair.
- Each configurable `Op` exposes a `Config` dataclass (pydantic) with block/tile/stage knobs, a `_get_heuristics_config` default, and `_get_autotuning_configs` that enumerates the search space. `tokamax.autotune(f, *args)` walks the HLO, finds all Tokamax ops, sweeps their configs, and returns a serializable `AutotuningResult` that can be re-entered as a context manager.
- Kernels use JAX custom calls, so they are not device-portable in StableHLO — `tokamax.DISABLE_JAX_EXPORT_CHECKS` is required to `jax.export` them.

## Architecture

Layering (top to bottom):

1. **Public API** (`tokamax/__init__.py`) — re-exports of kernel functions, `autotune`, `benchmark`, `Op`.
2. **Op registry** (`tokamax/_src/ops/<op>/api.py`) — builds an `IMPLEMENTATIONS` dict at import time, attempts each backend in order until one succeeds (or raises `ExceptionGroup` of `NotImplementedError`s).
3. **Op base class** (`tokamax/_src/ops/op.py`) — `Op` is a frozen dataclass carrying a `config`, a `vjp`, and overrides for `_fwd`, `_get_heuristics_config`, `_get_autotuning_configs`, `supported_on`. Handles vmap capture, HLO tagging for autotuning, and cache lookup.
4. **Backends** — one file per `(op, backend)` pair. Each subclasses `base.<Op>` and its own `Config`.
5. **Autotuner** (`tokamax/_src/autotuning/`) — `api.autotune`, `autotuner.AutotuningData`, on-disk cache (`cache.py`), and a vendored default cache (`tokamax/data/`).
6. **Benchmarking** (`tokamax/_src/benchmarking.py`) — `standardize_function` + `benchmark` that isolate kernel time from Python/JAX dispatch overhead; on GPU can route through the CUPTI profiler.

## Key abstractions

Every public kernel. Platforms: "T" = TPU (Pallas-Mosaic-TPU), "G90" = Pallas-Mosaic-GPU SM90 (H100), "G100" = Pallas-Mosaic-GPU SM100 (B200), "Tri" = Pallas-Triton (any CUDA GPU), "XLA" = portable XLA fallback. Bold = headline TPU kernel for this wiki.

| Kernel | Platforms | File | One-line description |
|---|---|---|---|
| **`dot_product_attention`** | **T**, G90, G100, Tri, cuDNN, XLA, `xla_chunked` | [`_src/ops/attention/api.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/api.py) | Scaled dot-product / flash attention with bias, mask, causal, local window, soft-cap, GQA/MQA, paged KV, quantized Q/K/V. |
| `↳ pallas_mosaic_tpu` | **T** (v5+) | [`_src/ops/attention/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py) | Wraps the experimental Splash-Attention TPU kernel (see below). |
| `↳ pallas_mosaic_gpu` SM90 | G90 | [`_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py) | Warp-specialised H100 flash attention. |
| `↳ pallas_mosaic_gpu` SM100 | G100 | [`_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py) | B200 / TMEM flash attention, supports collective 2-CTA MMA. |
| `↳ pallas_triton` | Tri | [`_src/ops/attention/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py) | Generic Triton flash attention (Algorithm 1 of the FlashAttention paper). |
| `↳ xla_chunked` | XLA | [`_src/ops/attention/xla_chunked.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py) | Memory-efficient chunked XLA attention ([Rabe & Staats 2021](https://arxiv.org/abs/2112.05682)); used as a TPU fallback alongside plain XLA. |
| **Splash attention kernel** (internal) | **T** | [`_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py`](../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py) | The actual TPU attention kernel — sparse-mask aware, supports MHA/MQA/GQA, soft-cap, separate fwd/bwd tiling. Mirrors the upstream JAX kernel at `jax.experimental.pallas.ops.tpu.splash_attention`; either entry point is viable for hypotheses. |
| **Ring attention kernel** (internal) | **T** | [`_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py`](../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py) | Sequence-parallel ring attention on TPU. Not wired into the public `dot_product_attention` API yet. |
| `gated_linear_unit` | G90, G100, Tri, XLA | [`_src/ops/gated_linear_unit/api.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api.py) | Fused `activation(x@W_gate) * (x@W_up)` — SwiGLU/GEGLU/REGLU/GLU. No TPU-specific kernel yet; **TPU runs XLA**. |
| `↳ pallas_triton` | Tri | [`_src/ops/gated_linear_unit/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py) | Fused Triton GLU. |
| `↳ pallas_mosaic_gpu` | G90, G100 | [`_src/ops/gated_linear_unit/pallas_mosaic_gpu.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py) | Mosaic-GPU GLU kernel. |
| `layer_norm` (also RMS-norm) | Tri, XLA | [`_src/ops/normalization/api.py`](../../raw/code/tokamax/tokamax/_src/ops/normalization/api.py) | LayerNorm / RMSNorm fused in a single pass. `subtract_mean=False` → RMSNorm. No TPU-specific kernel; **TPU runs XLA**. |
| `↳ pallas_triton` | Tri | [`_src/ops/normalization/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py) | Fused Triton norm + VJP variant in `pallas_triton_vjp.py`. |
| **`ragged_dot`** / `ragged_dot_general` | **T**, G90, G100, Tri, XLA | [`_src/ops/ragged_dot/api.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py) | Grouped MatMul for MoE ([Megablocks](https://arxiv.org/abs/2211.15841)). Supports fwd + dLHS + dRHS ragged-dot dimension numbers. |
| `↳ pallas_mosaic_tpu` | **T** (v5+) | [`_src/ops/ragged_dot/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py) | TPU grouped-matmul (`gmm` / `tgmm`) with optional int8/fp8 quantization via `qwix`. |
| `↳ pallas_mosaic_gpu` | G90, G100 (+ fp8, int8, post-scale) | [`_src/ops/ragged_dot/pallas_mosaic_gpu.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py) | GPU grouped-matmul, dispatches to SM90/SM100/quant variants. |
| `↳ pallas_triton` | Tri | [`_src/ops/ragged_dot/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py) | Generic Triton grouped matmul. |
| **`linear_softmax_cross_entropy_loss`** | **T only** | [`_src/ops/linear_softmax_cross_entropy_loss/api.py`](../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py) | Fused `x @ w → log-softmax → NLL` avoiding materialising the `[B, V]` logits. Memory-efficient CE loss ([Wijmans et al. 2024](https://arxiv.org/abs/2410.10989v2)). |
| `↳ pallas_mosaic_tpu` | **T** (v5+) | [`_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py) | TPU kernel, blocked over batch/hidden/vocab. |
| `↳ chunked_xla` | XLA | [`_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py`](../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py) | Pure-JAX chunked fallback (forward only in this commit). |
| `triangle_multiplication` | XLA only | [`_src/ops/triangle_multiplication/api.py`](../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/api.py) | AlphaFold triangle multiplicative update. No Pallas backend; XLA only. |
| `flex_attention` (experimental) | Tri, XLA | [`_src/ops/flex_attention/api.py`](../../raw/code/tokamax/tokamax/_src/ops/flex_attention/api.py) | PyTorch-style FlexAttention; not exported from the top-level `tokamax` package. |

Supporting abstractions:

- [`Op`](../../raw/code/tokamax/tokamax/_src/ops/op.py) — base class; frozen dataclass with generic parameters `[_P, _T, _R, _Config, _Key]`. Holds `config`, `vjp`, `config_cls`. Provides `.autotune()`, `.bind(...)` → `BoundArguments`.
- [`BoundArguments`](../../raw/code/tokamax/tokamax/_src/ops/op.py) — canonicalised call; exposes `default_config`, `autotuning_configs`, `autotuning_cache_key`.
- [`AutotuningResult`](../../raw/code/tokamax/tokamax/_src/autotuning/api.py) — JSON-serialisable override of the autotune cache, usable as a context manager and combinable via `|`.
- `base.Mask` (attention) — carries `is_causal`, `q_start/q_end`, `k_start/k_end`, explicit array mask; backends `.take(...)` the pieces they natively handle.

## Entry points

Imported directly from `tokamax`:

| Symbol | What it does |
|---|---|
| `tokamax.dot_product_attention` | Attention kernel dispatcher. |
| `tokamax.gated_linear_unit` | GLU dispatcher. |
| `tokamax.layer_norm` | LayerNorm/RMSNorm dispatcher. |
| `tokamax.ragged_dot` / `tokamax.ragged_dot_general` | MoE grouped matmul. |
| `tokamax.linear_softmax_cross_entropy_loss` | Fused linear + softmax CE loss (TPU). |
| `tokamax.triangle_multiplication` | AlphaFold triangle update. |
| `tokamax.autotune(f, *args)` → `AutotuningResult` | Autotunes all Tokamax ops in `f`. |
| `tokamax.get_bound_args(f, *args, **kwargs)` | Inspect ops without autotuning. |
| `tokamax.AutotuningResult.{load,loads,dump,dumps}` | (De)serialisation of autotune results. |
| `tokamax.benchmark(f_std, args, ...)` | Accelerator-only kernel timing. |
| `tokamax.standardize_function(f, kwargs=...)` | Freezes kwargs for `tokamax.benchmark`. |
| `tokamax.DISABLE_JAX_EXPORT_CHECKS` | Allow Tokamax custom calls through `jax.export`. |
| `tokamax.Op`, `tokamax.BoundArguments` | Subclass to add user kernels into the same autotune pipeline. |

## Dependencies

- **JAX** + **Pallas** (`jax.experimental.pallas`, `...pallas.triton`, `...pallas.mosaic_gpu`, `...pallas.tpu`) — all kernels are authored in Pallas.
- **`qwix`** — quantized array (`QArray`) abstraction used across attention and ragged_dot.
- **`pydantic`** — all `Config` dataclasses.
- **`immutabledict`**, **`jaxtyping`**, **`typing_extensions`**, **`tqdm`** (autotune progress bar).
- **cuDNN** (optional) via `jax.nn.dot_product_attention(implementation='cudnn')`.
- Python `pyproject.toml`: see [`pyproject.toml`](../../raw/code/tokamax/pyproject.toml).

## Notable files

- [`README.md`](../../raw/code/tokamax/README.md) — top-level intro, `implementation=None` dispatch semantics, autotune/benchmark snippets.
- [`docs/supported_ops.md`](../../raw/code/tokamax/docs/supported_ops.md) — per-op platform support matrix. Ingest as a source.
- [`docs/basic_usage.md`](../../raw/code/tokamax/docs/basic_usage.md) — canonical usage recipes. Ingest as a source.
- [`docs/splash_attention.md`](../../raw/code/tokamax/docs/splash_attention.md) — TPU splash-attention deep dive. Ingest as a source.
- [`docs/autotuning.md`](../../raw/code/tokamax/docs/autotuning.md) — semantics of `tokamax.autotune` + `AutotuningResult`. Ingest as a source.
- [`docs/benchmarking.md`](../../raw/code/tokamax/docs/benchmarking.md) — how `tokamax.benchmark` isolates kernel time. Ingest as a source.
- [`docs/tokamax_demo.ipynb`](../../raw/code/tokamax/docs/tokamax_demo.ipynb) — end-to-end demo notebook.
- [`tokamax/_src/ops/experimental/tpu/splash_attention/microbenchmarks.pdf`](../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/microbenchmarks.pdf) — splash-attention microbench figures.
- [`tokamax/benchmarks/`](../../raw/code/tokamax/tokamax/benchmarks/) — per-op microbenchmarks (`attention.py`, `ragged_dot.py`, `linear_softmax_cross_entropy_loss.py`, `triangle_multiplication.py`) + `benchmark_registry.pbtxt` of canonical shapes.

## Performance-relevant surfaces

Every `Config` dataclass below is what `tokamax.autotune` searches; every heuristic default is what `implementation=None` will pick without autotuning. These are the **knobs a hypothesis can change**.

### TPU — flash / splash attention

File: [`_src/ops/attention/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py). TPU v5+ only (`supported_on` gates via `pltpu.get_tpu_info().generation >= 5`).

Configurable via `Config` (lines 41–56):
- `block_q`, `block_kv`, `block_kv_compute` — must be multiples of `NUM_LANES` (128). `block_kv` must be a multiple of `block_kv_compute`.
- `q_layout`, `k_layout`, `v_layout` — `splash.QKVLayout.{HEAD_DIM_MINOR, SEQ_MINOR}`.
- `use_experimental_scheduler: bool`.
- `use_base2_exp: bool = True` — use base-2 softmax (avoid `log(2)` multiplies).

Heuristic default (lines 168–179): `block_q = block_kv = block_kv_compute = 128`, all layouts `HEAD_DIM_MINOR`, experimental scheduler on.

Autotune search space (lines 182–219): `tiles = [128, 256, 512, 1024, 2048, 4096]` for each of the three block sizes, × 2 layouts per input × 2 schedulers. Pruned by `seq_len ≥ 1024 ⇒ block ≥ 1024` and `bkv_c ≤ 1024`. TODO comment at line 186 flags that `8192` is excluded pending autotuning bug fixes.

Underlying kernel: [`SplashConfig`](../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py) (line 115) additionally exposes `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq` for the VJP — **these are currently hard-wired to 128 via `SplashConfig.get_default()`** and not exposed to the Tokamax-TPU autotuner. Potential tuning surface for backward-pass optimization.

### TPU — ragged_dot (MoE grouped matmul)

File: [`_src/ops/ragged_dot/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py). TPU v5+.

`Config` (lines 64–73):
- `tile_m`, `tile_k`, `tile_n` — all default 128.
- `input_buffer_count` — DMA prefetch depth, default 2, autotuned over `{2, 3, 4}`.
- `combine_scopes: bool` — affects `tgmm` (dRHS) code gen.

Autotune (lines 257–311): `tile_m ∈ {64·2^i}` capped at 1024, `tile_k` and `tile_n` as powers of two up to `k`/`n` plus the full-axis "single tile", `input_buffer_count ∈ {2, 3, 4}`. Constructor comment (line 278) explicitly caps at 1024 "to ensure reasonable compilation time".

Optional `qdtype` attribute on the op triggers int8/fp8 quantization per `qwix`. Three ragged-dot dimension numbers are supported: `DEFAULT` (fwd), `DLHS` (bwd wrt lhs), `DRHS` (bwd wrt rhs) — see lines 75–93.

### TPU — linear softmax cross-entropy loss

File: [`_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py). TPU v5+.

`Config` (lines 32–44):
- `b_block_size` — batch block, `≥ 1024`, multiple of 128, default 1024.
- `h_block_size` — hidden block, `≥ 128`, multiple of 128, default 512.
- `v_block_size` — vocab block, `≥ 128`, multiple of 128, default 2048 (1024 on v6e when H is not 512-aligned — see lines 47–61 — because padding causes VMEM spill / OOM on gen 6).

Heuristic default varies by TPU generation (`get_tpu_specific_default_config`, lines 47–61): gen ≥ 7 uses `(1024, 512, 2048)`; gen 6 uses `(1024, 512, 2048)` or `(1024, 512, 1024)` based on H alignment; gen 5 uses `(1024, 512, 512)`. **Direct observation**: VMEM spill on v6e is an explicit concern baked into the heuristics.

Autotune (lines 110–140): powers of two and divisors of each dim, Cartesian product. Separate VJP class has its own `_get_autotuning_configs` (lines 198–227) mirroring the forward.

### TPU — attention "ring" kernel (experimental, not wired into public API)

File: [`_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py`](../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py). Potential candidate for sequence-parallel TPU attention — currently reachable only by importing `ring_attention_kernel` directly.

**2026-04-23 update (from the [pallas-kernel directory](../analyses/2026-04-23-pallas-kernel-directory.md))**: three public reference implementations of ring-attention on TPU now exist and close the "API gap only" characterization.
- [**maxdiffusion** splash-integrated `ring_attention_kernel.py`](maxdiffusion.md) — announced 2026-04-16; composes with splash fwd + bwd primitives; reuses splash's bf16 tuning. **Recommended pattern** for wiring tokamax's ring kernel through a splash-style dispatch.
- [**haoliuhl/ringattention**](ringattention.md) — canonical paper companion; from-scratch flash inner kernel; unidirectional, straight `below_or_on_diag` causal. Useful as reference for paper-fidelity understanding.
- [**ejkernel** `ring_attention`](ejkernel.md) — splash wrapper (orchestration-only, not a novel kernel). Another reading of the same pattern.
- **Zig-Zag (load-balanced) causal variant** (Brandon et al. 2023) **remains absent** from all surveyed repos including these three — an actual open algorithmic port.

### GPU — attention (SM90 / SM100 / Triton)

Common `ConfigBase` at [`_src/ops/attention/pallas_mosaic_gpu_common.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py) (lines 37–55):
`block_q` (multiple of 64, default 64), `block_kv` (multiple of 64, default 64), `num_stages` (default 2), `fold_q_sequence_heads` (bool), `split_k` (default 1). SM100 adds `collective` (2-CTA MMA) and `persistent` flags.

Heuristics: SM100 picks `block_q = 256` when collective else `128`, `block_kv = 128`, `num_stages = max(256/head_dim, 1) * cluster_size` (see [`pallas_mosaic_gpu_kernel_sm100.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py) lines 108–143). Autotuner enumerates `block_q ∈ {64, 128, ...}` with the constraint `2·block_q | seq_len_q` and `block_kv | seq_len_kv`, `num_stages ∈ {2, 3, 4}`.

Triton: [`_src/ops/attention/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py), own `Config` with `block_m`, `block_k`, `block_d`, `block_d_out`, `num_warps`, `num_stages`, `use_base2`, `use_stable_softmax`, `pack_mask`.

### GPU — gated linear unit

Triton ([`_src/ops/gated_linear_unit/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py), `Config` at lines 81–86): `block_m`, `block_n`, `block_k` (default 32), `num_warps`, `num_stages`. Heuristic caps `block_m ≤ 128`, `block_n ≤ 256`, `block_n ≤ 128·128/block_m` (register-pressure cap).

Mosaic-GPU ([`_src/ops/gated_linear_unit/pallas_mosaic_gpu.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py)): SM90 / SM100 specialisations in `pallas_mosaic_gpu_kernel_sm{90,100}.py`.

### GPU — normalization

[`_src/ops/normalization/pallas_triton_config.py`](../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py), `Config`: `block_m` (power of 2), `block_n` (power of 2 or `None`), `num_warps` (power of 2). Heuristic (lines 82–125) sizes blocks to stay within a register budget (`NUM_REGISTERS_PER_SM / 4`) and to saturate at least `4 × core_count` blocks. VJP has its own config in `pallas_triton_vjp_config.py`.

### GPU — ragged_dot

Common `Config` at [`_src/ops/ragged_dot/pallas_mosaic_gpu_common.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py) (lines 45–62): `block_m` (multiple of 8), `block_n`, `block_k`, `num_stages`, `split_k`, `split_m`, `persistent`, `post_scale`, `collective` (B200 2-CTA), `grid_minor_dim` (M/N), `grid_tile_width`. Dedicated quant variants live in `pallas_mosaic_gpu_kernel_sm{90,100}[_fp8|_i8|_post_scale]_quant.py`.

### Autotune / cache / benchmarking

- [`_src/autotuning/api.py`](../../raw/code/tokamax/tokamax/_src/autotuning/api.py) — `autotune(f, *args, ignore_cache=False, all_implementations=False)` walks lowered HLO, dedupes by `autotuning_cache_key`, sweeps each op's `_get_autotuning_configs`, returns `AutotuningResult`.
- [`_src/autotuning/autotuner.py`](../../raw/code/tokamax/tokamax/_src/autotuning/autotuner.py) — `AutotuningData` (per-op per-key timings).
- [`_src/autotuning/cache.py`](../../raw/code/tokamax/tokamax/_src/autotuning/cache.py) — on-disk cache; prebuilt entries under [`tokamax/data/`](../../raw/code/tokamax/tokamax/data/).
- [`_src/benchmarking.py`](../../raw/code/tokamax/tokamax/_src/benchmarking.py) — `benchmark(..., method=None | 'cupti', iterations=...)`, `standardize_function`.
- [`tokamax/config.py`](../../raw/code/tokamax/tokamax/config.py) — global config flags (e.g. `tokamax_autotuning_cache_miss_fallback`).

## Connections

- See the `attention` concept family (flash attention, splash attention, ring attention) — tokamax is the main library implementing these for TPU in JAX.
- See the `gated_linear_unit` / `layer_norm` concepts — tokamax currently has **no TPU-specific kernel** for these; any TPU hypothesis that wants to swap them will need to reach for XLA and rely on compiler fusion, or upstream a TPU kernel.
- See the `moe` / `ragged_dot` concepts — `tokamax.ragged_dot` is the direct TPU implementation candidate.
- See the `memory-efficient cross-entropy` concept — `linear_softmax_cross_entropy_loss` is the TPU implementation.
- When a [model](../models/) uses any of these ops on TPU, the ranked-hypothesis list for that model should include: swap to tokamax kernel, autotune tokamax kernel, adjust heuristic block sizes.

## See also

- [`scaling-book.md`](./scaling-book.md)
- [`stablehlo.md`](./stablehlo.md)

## Sources

- [`raw/code/tokamax/README.md`](../../raw/code/tokamax/README.md) @ `54bdd95`
- [`raw/code/tokamax/docs/supported_ops.md`](../../raw/code/tokamax/docs/supported_ops.md)
- [`raw/code/tokamax/docs/basic_usage.md`](../../raw/code/tokamax/docs/basic_usage.md)
- [`raw/code/tokamax/docs/splash_attention.md`](../../raw/code/tokamax/docs/splash_attention.md)
- [`raw/code/tokamax/docs/autotuning.md`](../../raw/code/tokamax/docs/autotuning.md)
- [`raw/code/tokamax/docs/benchmarking.md`](../../raw/code/tokamax/docs/benchmarking.md)
- [`raw/code/tokamax/tokamax/__init__.py`](../../raw/code/tokamax/tokamax/__init__.py)
- [`raw/code/tokamax/tokamax/_src/ops/attention/api.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/api.py)
- [`raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py)
- [`raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py`](../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py)
- [`raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py)
- [`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py`](../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py)
- [`raw/code/tokamax/tokamax/_src/ops/normalization/api.py`](../../raw/code/tokamax/tokamax/_src/ops/normalization/api.py)
- [`raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py`](../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py)
- [`raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api.py)
- [`raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py`](../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py)
- [`raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py`](../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py)
- [`raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py`](../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py)
- [`raw/code/tokamax/tokamax/_src/autotuning/api.py`](../../raw/code/tokamax/tokamax/_src/autotuning/api.py)
- [`raw/code/tokamax/tokamax/_src/ops/op.py`](../../raw/code/tokamax/tokamax/_src/ops/op.py)
- [Ultra-Scale Playbook (2025)](../sources/2025-ultrascale-playbook.md) — GPU-side reference for FlashAttention, Ring Attention, MoE all-to-all, mixed precision; informs tokamax hypothesis surfaces: `ring_attention_kernel` API wiring, splash-attention backward block autotuning, TPU Pallas kernels for `gated_linear_unit` and `layer_norm` (currently XLA fallback), DeepSeek-V3 1×128/128×128 FP8 tile scheme for `ragged_dot`.
