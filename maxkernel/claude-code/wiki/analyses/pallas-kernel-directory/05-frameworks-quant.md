---
title: "Pallas kernel directory — §5 Frameworks & quantization libraries"
type: analysis
tags: [directory, pallas, kernels, qwix, aqt, jaxite, pytorch-xla, marin, levanter, tunix, paxml]
created: 2026-04-23
updated: 2026-04-23
---

Ten repositories that ship Pallas kernels embedded in framework code, wrap Pallas for a different front-end, or (in one case) emit Pallas as a compiler target. These repos are **consumers and thin re-packagers** of Pallas far more than they are kernel authors — almost every genuinely novel kernel here lives in [marin/levanter](https://github.com/marin-community/marin), [google/qwix](https://github.com/google/qwix), or [google/jaxite](https://github.com/google/jaxite); the rest are glue to upstream `jax.experimental.pallas.ops.tpu.*`. Part of [2026-04-23 Pallas kernel directory](../2026-04-23-pallas-kernel-directory.md).

## 5.1 google/tunix

Post-training library (SFT/DPO/GRPO/RL). **Does not define custom Pallas kernels** — model files import upstream JAX Pallas directly. Three of six model families use Pallas; the rest are pure XLA.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `splash_attention_kernel.make_splash_mha` | [qwen2/model.py L25](https://github.com/google/tunix/blob/main/tunix/models/qwen2/model.py#L25), [qwen3/model.py L27](https://github.com/google/tunix/blob/main/tunix/models/qwen3/model.py#L27), [gemma4/model.py L26](https://github.com/google/tunix/blob/main/tunix/models/gemma4/model.py#L26) | `mosaic_tpu` (upstream) | stable (upstream) | none in-repo | Training flash attention gated by `use_flash_attention`, `seq_len > 1` only | Qwen2, Qwen3, Gemma4 attention blocks | Hard-codes `flash_attention_block_size=1024` default per model, exposed via `ModelConfig`. All `splash.BlockSizes` fields passed through. Wrapped in `shard_map`. Decode falls back to einsum GQA |
| `megablox.gmm` / `megablox.tgmm` | [qwen3/model.py L26, L866-L880](https://github.com/google/tunix/blob/main/tunix/models/qwen3/model.py#L26) | `mosaic_tpu` (upstream) | stable (upstream) | none in-repo | Grouped MoE FFN (gate/up/down) for Qwen3 MoE | `Qwen3MoeBlock.__call__`, behind `use_megablox=True` | Wraps three `megablox.gmm` calls inside `sharded_megablox_moe` via `shard_map`. Falls back to dense when disabled or on CPU. **No custom kernel** |

Apache-2.0. Autotune surface: block sizes only via `flash_attention_block_size` + megablox defaults. No hardware-specific tuning table.

## 5.2 google/qwix

Modern Google quantization library, **explicit AQT successor**. Two relevant files: lifted `pallas_call` for `QArray` plumbing + a real Pallas kernel.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `pallas_call` wrapper + `update_block_specs_for_qarray` / `transform_block_specs_for_tpu` | [qwix/_src/core/pallas.py](https://github.com/google/qwix/blob/main/qwix/_src/core/pallas.py) | `mosaic_tpu` | stable | none | Pytree-level support for passing `QArray` (value + scale + zero-point) through `pl.pallas_call`; scale-tile BlockSpec derivation; memory-saving transpose for trailing-dim-1 tensors | Any Qwix PTQ/QAT path calling a Pallas kernel with quantized weights | Not a kernel — a Pallas lifting layer. Mirrors AQT's `pallas_call.py` role, reduced |
| `quantized_matmul_kernel` | [qwix/contrib/kernels/quantized_matmul.py](https://github.com/google/qwix/blob/main/qwix/contrib/kernels/quantized_matmul.py) | `mosaic_tpu` | **experimental** (`INTERPRET: bool = True` hard-coded with TODO to flip for prod) | none | `QArray × QArray → f32/bf16` GEMM tile-by-tile; supports per-tile scales where `k_tile_size == bk` and scale grid must match `(mdim//bm, ndim//bn)` or be 1 | Opt-in from `dot_general` when `can_use_qmm_in_dot_general` matches (no zero-point, contracting on `(1,)/(0,)`) | `QuantizedMatmulConfig(bm=128, bk=128, bn=128, dtype=f32)`. Output dtype = `preferred_element_type=o_ref.dtype`; per-tile `* sx_ref * sy_ref` scaling; zero-init when `pl.program_id(2) == 0`. Contrib — not stable API. Apache-2.0 |

Qwix README: "Qwix is a Jax quantization library supporting QAT and PTQ for both XLA targets (CPU/GPU/TPU) and ODML targets (LiteRT)." Migration: "*AQT is reaching end-of-life. github.com/google/qwix is replacing AQT.*"

## 5.3 google/aqt

Predecessor to qwix. **End-of-life** per README banner. Files remain in `main` but are effectively deprecated.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `pallas_call` (QTensor-lifted) | [aqt/jax/v2/pallas/pallas_call.py](https://github.com/google/aqt/blob/main/aqt/jax/v2/pallas/pallas_call.py) | `mosaic_tpu` | **deprecated** | none | `pl.pallas_call` wrapper accepting `QTensor` as pytree leaves, per-scale `BlockSpec` via `make_qtensor_blockspec`, memory-saves by swapping minor/second-minor axes when `shape[-1] == 1 and shape[-2] > 1` (prevents 128-lane padding waste) | Any AQT Pallas kernel | Precursor to qwix's `pallas_call` |
| `dot_general` (Pallas-callable) | [aqt/jax/v2/pallas/dot_general.py](https://github.com/google/aqt/blob/main/aqt/jax/v2/pallas/dot_general.py) | `mosaic_tpu` | **deprecated** | none | Drop-in `lax.dot_general` replacement *inside* Pallas kernels; always dequantizes output; `int4/int8/bf16/f32` + fp8 via `fp8_numerics._convert_to_fp8dtype`; configurable `lhs_dequant_mode` / `rhs_dequant_mode` ∈ `{OUTPUT, OTHER_INPUT, THIS_INPUT}` | AQT-Flax/Pax user kernels | Driven by `aqt_dot_general.dot_general_raw_make` |
| `pallas_tensor.TransposedTensor` + `make_qtensor_blockspec` | [aqt/jax/v2/pallas/pallas_tensor.py](https://github.com/google/aqt/blob/main/aqt/jax/v2/pallas/pallas_tensor.py) | `mosaic_tpu` | **deprecated** | none | Flax-struct pytree wrapper carrying permute axes; `untransposed` property re-applies inverse permutation via `np.argsort(permute_axes)` inside kernel | `pallas_call` above | Scale-block-spec calibration: for each calibration axis (`scale.shape[i] == 1`), forces `scale_blk_shape[i] = 1` + rewrites index map to 0 |
| `quantizer.quant` | [aqt/jax/v2/pallas/quantizer.py](https://github.com/google/aqt/blob/main/aqt/jax/v2/pallas/quantizer.py) | `mosaic_tpu` | **deprecated** | none | Per-tensor / per-channel quant inside Pallas with `scale_stop_grad=False`, `scale_dtype=f32` (forced — VPU ops float32-only) | AQT Pallas kernels | File comment: "TODO(wppark): Remove this file. Temporary before the official release of AQT quant / dequant API." |

All Apache-2.0. No tunable block-size autotuner; shapes are caller-chosen.

## 5.4 google/jaxite

**Non-ML Pallas use case**: CGGI-style FHE (Fully Homomorphic Encryption) on TPU/GPU. Only FHE backend targeting TPU Pallas directly; reference for integer-heavy, non-neural Pallas.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| Negacyclic vector-matrix polymul (`_i32_matmul_unreduced`, `_i32_matmul_unreduced_CGGI`, `_decomposed_vector_matrix_polymul`, `bat_matmul`) | [jaxite/jaxite_lib/polymul_kernel.py](https://github.com/google/jaxite/blob/main/jaxite/jaxite_lib/polymul_kernel.py) | `mosaic_tpu` | research | "reducing the number of matmuls by 4x" (CGGI optimization, source comment) | `u32 × u32` polynomial multiplication as four `i8 → bf16 → f32 accumulate` matmuls then bit-shifted sum to reconstruct `u32` | `jaxite_bool` boolean gate API + CGGI bootstrap primitives; backend for Google FHE compiler | Module docstring: `"""Kernel for negacyclic vector-matrix polymul."""`. Splits 32-bit ints into 4 bytes, multiplies in bf16 with `preferred_element_type=f32`, reassembles with shifts. `bat_matmul` is batched variant using `jnp.einsum("cmnpq,cnkq->cmkp")`. Non-Pallas fallbacks exist but don't lower well on TPU per comments. Apache-2.0. Paper refs: CGGI — [eprint 2018/421](https://eprint.iacr.org/2018/421) |

Clearest public example of "use Pallas to express something XLA lowers badly, in a non-ML domain."

## 5.5 google/paxml + google/praxis

Legacy Pax JAX framework. Listed file `praxis/layers/gpu_fast_attention.py` is **GPU-only** Triton Pallas wrapper tested only on A100 — does not contain TPU Pallas. No other praxis/paxml file imports `jax.experimental.pallas`.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| GPU MHA + layer norm + decode attention (upstream imports) | [praxis/layers/gpu_fast_attention.py](https://github.com/google/praxis/blob/main/praxis/layers/gpu_fast_attention.py) | `triton` (via `jax.experimental.pallas.ops.gpu`) | experimental | Header: "*Experimental only. Only tested on NVIDIA A100s.*" | Fused MHA fwd/bwd (`attention.mha`), layer norm, flash decoding via `decode_attention.mha` with `k_splits` (default 16) — linked to [flashdecoding blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | `GpuCudnnFusedDotProductAttention`, `GpuTritonFused{GQA,MQA}`, `GpuFlashDecoding*` layer subclasses | Imports guarded by `try/except ImportError` + `logging.warning('jax_triton not found')`. Bwd selected by `pax_flash_attention_backward_pass_impl` (default `'xla'`). Knobs: `use_flash_decoding: bool = False`, `flash_decoding_k_splits: int = 16`. **No TPU Pallas kernels anywhere in praxis/paxml** |

Low priority for TPU perf work. Apache-2.0.

## 5.6 pytorch/xla

Five files under `torch_xla/experimental/pallas_kernels/`. None invoked directly by user code — imported from [`torch_xla/experimental/custom_kernel.py`](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/custom_kernel.py), which uses a `trace_pallas` → `make_kernel_from_pallas` lowering path (StableHLO serialized payload, tensor args through XLA). **This is the integration pattern PyTorch-on-TPU uses** to call Pallas kernels from eager/lazy Torch.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `paged_attention` (multi-query) | [multi_queries_paged_attention_kernel.py](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/pallas_kernels/multi_queries_paged_attention_kernel.py) | `mosaic_tpu` | experimental | docstring: "PagedAttention TPU kernel with query_len>1 support" | Decode/prefill paged attention with mixed query lengths; int8-quantized K/V via `quantization_utils.from_int8` | `torch_xla.experimental.custom_kernel.multi_queries_paged_attention` (line ~1200) | **Derivative** of `jax.experimental.pallas.ops.tpu.paged_attention` — imports `quantization_utils` from upstream. `MultiPageAsyncCopyDescriptor` uses `pltpu.make_async_copy` with per-page semaphores |
| `quantized_matmul_int8` | [quantized_matmul_kernel.py](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/pallas_kernels/quantized_matmul_kernel.py) | `mosaic_tpu` | experimental | none | INT8 W8A8 / W8A16 matmul with optional on-the-fly `x` quantization (`_quantize_array`); per-tile x-abs-max reuse via `x_q_scratch` + `x_scale_scratch` scratch accumulators | `torch_xla.experimental.custom_kernel.quantized_matmul_int8` (line ~1078), falls back to `quantized_matmul_xla` | **Novel** (not direct upstream port). Flags: `quantize_activation`, `save_acc`, `save_x_q`. Block sizes `batch_block_size`, `out_block_size`, `in_block_size` come from caller. Uses `unfold_args` pattern to specialize bool predicates into `lax.cond` branches at trace time |
| `paged_attention` (ragged, v1) | [ragged_paged_attention_kernel.py](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/pallas_kernels/ragged_paged_attention_kernel.py) | `mosaic_tpu` | experimental (deprecated in favor of v2) | none | Prior-gen ragged paged attention | Not default since v2 landed | Similar MultiPageAsyncCopyDescriptor |
| `ragged_paged_attention` (v2) | [ragged_paged_attention_v2.py](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/pallas_kernels/ragged_paged_attention_v2.py) | `mosaic_tpu` | experimental | File docstring: "highly optimized implementation of ragged paged attention, specifically designed for TPU" | vLLM-style mixed prefill+decode batches | `torch_xla.experimental.custom_kernel.ragged_paged_attention` (line ~969, falls back to `_ragged_paged_attention_nonkernel`) | Header: "# Copyright 2025 The JAX Authors." — **vendored from upstream JAX**. BSD-3-Clause over Apache-2.0 |

`torch_xla.experimental.custom_kernel` also imports (from upstream JAX, not vendored): `jax.experimental.pallas.ops.tpu.flash_attention` (`_flash_attention_impl`, `_flash_attention_bwd_dq/dkv`, `SegmentIds`), `paged_attention_kernel.paged_attention`, `megablox.gmm` / `megablox.tgmm`. A `trace_pallas_arg_to_payload: Dict[Tuple[Any], str]` dict caches trace-to-StableHLO payload process-locally (`_xla_increment_counter('trace_pallas_cache_hit')`).

## 5.7 google-pytorch/torchtitan (fork)

Returned **404** on API enumeration for the documented `torchtitan/experiments/tpu/kernels/linear_softmax_cross_entropy_loss.py` — fork appears private/empty/renamed. The kernel it was supposed to host is upstream in marin/levanter (§5.8) as **tokamax-derived**. Treat as unavailable vendored redirect.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `linear_softmax_cross_entropy_loss` | [torchtitan/experiments/tpu/kernels/linear_softmax_cross_entropy_loss.py](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/linear_softmax_cross_entropy_loss.py) (**404 at catalog time**) | `mosaic_tpu` | experimental | none observable | Fused linear + softmax + CE loss for TPU | TorchTitan TPU training experiments | Documented as **tokamax-derived**; kernel body lives upstream in marin/levanter `fused_cross_entropy_loss/pallas_tpu.py` with header: "This implementation is heavily based on Tokamax's linear softmax cross-entropy Pallas Mosaic TPU kernel (Apache-2.0). We adapt it for Levanter's API and add optional logsumexp penalty, logit soft-cap, and external loss weighting support." |

## 5.8 marin-community/marin (vendors levanter)

**Most interesting repo in this group.** Pallas kernels at `lib/levanter/src/levanter/kernels/pallas/`. Four sub-kernels + three harness modules + one template.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `linear_softmax_cross_entropy_loss_pallas` (fwd + streaming bwd) | [fused_cross_entropy_loss/pallas_tpu.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py) | `mosaic_tpu` | stable | none numeric; features over tokamax upstream: optional logsumexp penalty, logit soft-cap (`tanh(logits/cap)*cap`), external loss weighting, optional argmax return | Fused `x @ w → logits → softmax → CE` with streaming V-tile bwd | [api.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py) dispatcher | **Tokamax-derived**, explicitly. Provides `pl.estimate_cost`-based `_fwd_cost_estimate` + `_backward_cost_reference`. Uses `with_io_bytes_accessed` from `cost_estimate_utils.py`. Env var `LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH` toggles bwd streaming. Apache-2.0 |
| `linear_softmax_cross_entropy_loss_pallas_gpu` | [fused_cross_entropy_loss/pallas_gpu.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py) | `triton` (mosaic_gpu fallback) | stable on H100/A100, experimental on GB10 | Empirical constants: "`_NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376` … H100 has 232,448 bytes per-SM shared memory; kernel overhead (input tiles, accumulators, Triton metadata) consumes ~131 KB, leaving 101,376 bytes for the weight tile. Same limit applies to all NVIDIA GPUs including GB10." `_GB10_MAX_H_TILES = 512`; per-device V-block caps for GB10 and H100 | Same kernel, NVIDIA backend | Same dispatcher | Separate GB10 native-forward opt-in via `LEVANTER_PALLAS_GPU_GB10_NATIVE_FORWARD`. `_GB10_FULL_MATMUL_MAX_OUTPUT_ELEMENTS = 67_108_864` guards compile behavior |
| SSD (state-space duality) | [ssd/api.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/ssd/api.py), [ssd/xla.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/ssd/xla.py) | `xla` (**TPU Pallas explicitly absent**) | stable (XLA only) | none | Mamba-style SSD chunked fwd / intra-chunk / chunk-state | Levanter Mamba-family models | `ssd_intra_chunk_pallas` raises `PallasUnsupportedError("SSD TPU Pallas kernel is intentionally absent; use the XLA path.")`. Scaffolding present (dispatcher + `reference`/`xla`) but Pallas backend deliberately not filled. XLA variants: three prefix-emit (`einsum3`, `scan_fused`, `auto`) + four local-output strategies |
| Mamba3 MIMO | [mamba3/api.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py) | `xla` | stable (XLA only) | none | Mamba3 chunked fwd, ranked/unranked MIMO, rank-expand/collapse | Levanter Mamba3 models | Only `"xla"` and `"reference"` in `Implementation` — explicitly **no `pallas_tpu`** |
| `template_kernel.py` | [template_kernel.py](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/kernels/pallas/template_kernel.py) | — | scaffold | Docstring: "This file is intentionally not 'the best kernel'. It is a scaffold showing the expected structure: 1) a vanilla JAX reference implementation (the oracle), 2) one or more accelerated implementations (e.g. Pallas on TPU), 3) a stable API entrypoint that selects among implementations, following the same pattern as Tokamax: an explicit `implementation=` option plus a best-available default fallback order." | Template for future kernels | The `.agents/skills/add-pallas-kernel/` skill referenced is their Cursor-style auto-scaffolding helper | Apache-2.0 |

### Autotune harness

**The most valuable artifact in this group.** Three files implement a kernel-agnostic Pallas block-size autotuner.

**`autotune_utils.py`** (161 lines):
- `sharding_of`, `hlo_sharding_of`, `named_sharding_of`, `value_uses_manual_sharding` — extract sharding metadata (including from tracers); detect `shard_map`-local manual shardings.
- `shape_dtype_struct_for_benchmark` — builds `jax.ShapeDtypeStruct` preserving the array's sharding unless it's a manual tracer (strips sharding, which would otherwise prevent lowering).
- `contains_tracer` + `benchmark_lowering_args` — when any input is already a `jax_core.Tracer`, pass through; otherwise synthesize abstract `ShapeDtypeStruct`s to avoid allocating real arrays per candidate.
- `should_offload_compile` — decides if the candidate must compile off-thread (under JIT / manual-sharded / mesh-bound with `jax._src.mesh.thread_resources.env.physical_mesh` non-empty).
- `compile_benchmark_fn` + `compile_benchmark_fn_current_thread` — measure `jax.jit(fn).lower(*args).compile()` wall-clock; submits to single-thread `ThreadPoolExecutor(max_workers=1, thread_name_prefix="pallas_autotune")` when off-thread.
- `maybe_wrap_in_shard_map` — if all inputs globally `NamedSharding`-ed on a single mesh, wraps in `jax.shard_map`; if any input is already manual, returns unchanged.

**`cost_estimate_utils.py`** (30 lines): `with_io_bytes_accessed(body_cost, kernel_inputs_specs, kernel_outputs_specs)` copies a `pl.CostEstimate`'s FLOPs/transcendental/remote-bytes fields + overwrites `bytes_accessed` with `sum(prod(shape) * itemsize)` over real kernel IO — because `pl.estimate_cost` on a pure-JAX reference over-counts compute intermediates but under-counts IO for a real Pallas tile.

**`autotune_cache_utils.py`** (~120 lines): filesystem-backed cache keyed under user's `jax_compilation_cache_dir` at `<cache>/levanter_kernel_autotune/<kernel_name>/<filename>` using `rigging.filesystem.url_to_fs` (same gs://-aware path layer as JAX's compile cache). Atomic-ish read-through/write-back.

**Autotune flow in `fused_cross_entropy_loss/api.py`:**
- `_ensure_autotune_cache_loaded` + `_persist_autotune_cache` materialize `dict[str, BlockSizes]` keyed by `"<impl>|<backend>|<device_kind>|<B>|<H>|<V>|<dtype>|<soft_cap>|<return_argmax>|<jaxpr_hash>"`.
- `_autotune_jaxpr_hash` sha256s `str(jax.make_jaxpr(_loss_only)(x, labels, w).jaxpr)[:16]` — pins cache key to jaxpr shape.
- `_autotune_enabled()` controlled by `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS` (default on).
- `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` — candidate sizes whose compile time exceeds 0.2s vs baseline are **filtered** (the autotuner discards block sizes that dominate training step time with PJRT compile cost).
- `_VMEM_COMPILE_FALLBACK_WARNINGS_EMITTED` catches TPU VMEM OOMs (`"resource_exhausted" in message and "vmem" in message`) during candidate compile + falls through to next implementation.
- `tuned_block_sizes.py` carries hand-curated fallback `TUNED_BLOCK_SIZES: dict[str, dict[tuple[dtype, shape_bucket], BlockSizes]]` with per-device-kind entries (`"default"`, `"NVIDIA"`, `"NVIDIA GB10"`, `"NVIDIA H100"`, `"NVIDIA A100"` ...).

Only the fused CE kernel is currently autotune-wired; SSD + Mamba3 don't call this harness. But because it's kernel-agnostic, any future Pallas kernel can opt in just by having `BlockSizes` + an API entrypoint reading from `_AUTOTUNE_BLOCK_SIZE_CACHE`.

Apache-2.0.

## 5.9 marin-community/levanter

Upstream of §5.8. At inspection, `src/levanter/kernels/pallas` on main returned empty; marin's `lib/levanter/` vendors a newer snapshot. **Kernel content identical to §5.8**; for current Pallas work read the marin vendored copy.

## 5.10 pytorch/pytorch — `torch/_inductor/codegen/pallas.py`

**Not a Pallas kernel library** — it's PyTorch Inductor's codegen target that emits Pallas source code.

| Kernel | Source path | Backend | Stability | Perf claims | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `PallasPrinter` + `SIMDKernel` Pallas backend | [torch/_inductor/codegen/pallas.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/pallas.py) (200,861 bytes) | `mosaic_tpu` (via `torch_dtype_to_jax` bridge) | experimental | none | Compile `torch.compile`'d graphs to Pallas source (SIMD-style) — a `PythonPrinter` subclass translating sympy → `jnp.where`/`jnp.minimum`/`jnp.maximum`, driven by `SIMDKernel`/`SIMDScheduling` from inductor | `torch.compile` when Pallas backend selected | Extends `OpOverrides`, `CSEVariable`, `IndentedBuffer`, `PythonPrinter`, `BackendFeature` from inductor common; uses `BlockPatternMatcher` from `block_analysis`. BSD-3-Clause. **Compilation target, not a library of reusable kernels** |

## Cross-repo observations

**Derivative / vendored:**
- Everything in `tunix/models/**` (splash, megablox) is direct upstream import — no repo-local kernels.
- `pytorch/xla`'s `ragged_paged_attention_v2` is **vendored** from JAX (header still says "Copyright 2025 The JAX Authors"). Its `multi_queries_paged_attention_kernel.py` and `ragged_paged_attention_kernel.py` (v1) reuse `quantization_utils` from upstream + follow same `MultiPageAsyncCopyDescriptor` pattern.
- `marin/levanter`'s `fused_cross_entropy_loss/pallas_tpu.py` is **explicitly tokamax-derived** ("heavily based on Tokamax's linear softmax cross-entropy Pallas Mosaic TPU kernel") — Levanter's additions: logsumexp penalty, logit soft-cap, loss weighting, argmax return.
- `google-pytorch/torchtitan`'s advertised CE kernel is also tokamax-derived; currently unreachable via public API (404).
- `aqt/jax/v2/pallas/pallas_call.py` and `qwix/_src/core/pallas.py` are the same pattern (QTensor-aware `pallas_call`) re-done — qwix supersedes AQT.

**Genuinely novel kernels in this group:**
- `jaxite/jaxite_lib/polymul_kernel.py` — only kernel here with non-ML target domain (FHE polymul via byte-split bf16 matmul + shift reassembly).
- `pytorch/xla`'s `quantized_matmul_kernel.py` — INT8 W8A8/W8A16 matmul with optional on-the-fly activation quant + scratch-buffer reuse pattern not in upstream JAX.
- `qwix/contrib/kernels/quantized_matmul.py` — per-tile scale W8A8 matmul for `QArray`; `INTERPRET = True` hard-coded.
- **`marin/levanter`'s autotune harness** (`autotune_utils.py` + `cost_estimate_utils.py` + `autotune_cache_utils.py`) — not a kernel, but a reusable tuner.

### What the marin autotune harness enables that tokamax's built-in autotune doesn't

1. **Compile-time-aware filtering.** Tokamax's autotuner measures run time; marin's `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` discards candidate block sizes whose XLA compile time alone dominates the training step. At autoresearch scale (many experiments, many shapes, PJRT compile cache cold) this matters more than kernel wall-time.
2. **VMEM-OOM-aware fallthrough.** `_is_tpu_vmem_compile_error` + `_warn_vmem_compile_fallback_once` demote candidates that hit `resource_exhausted … vmem` at lowering and move to next impl instead of raising. Tokamax's tuner assumes pre-filtered candidate space.
3. **Sharding-preserving benchmark lowering.** `shape_dtype_struct_for_benchmark` keeps real `NamedSharding` when input is globally sharded; only strips under `shard_map` manual tracers. Tokamax tunes on single-device abstracts; marin tunes on exact mesh-shard combination production sees — avoids the "block size that works under FSDP differs from single-device bench" trap.
4. **Off-thread compile for mesh-bound contexts.** Single-worker `ThreadPoolExecutor` in `_AUTOTUNE_THREAD_POOL` lets tuner run while main thread holds JIT / mesh state. `jax_core.unsafe_am_i_under_a_jit_DO_NOT_USE()` is the switch. Tokamax doesn't handle nested-JIT autotune.
5. **jaxpr-hashed cache keys.** `_autotune_jaxpr_hash` pins cache entry to stringified jaxpr (truncated SHA-256) — silent shape/dtype/soft-cap changes invalidate. Tokamax's signature-arg keying can miss jaxpr-level differences (e.g., enabling `return_argmax`).
6. **GCS-aware persistent cache reuse.** `autotune_cache_utils` writes under `<jax_compilation_cache_dir>/levanter_kernel_autotune/` — the **same bucket PJRT caches compiles into**, via `rigging.filesystem.url_to_fs`. Every training job shares tuning transparently.

**Tokamax's `tokamax.autotune` is a kernel-author tool (write-time, one-shot). Marin's harness is a deployment-time tuner: kernel-agnostic, shard-aware, compile-cost-aware, cache-persistent.** For the autoresearch loop in this wiki, the marin pattern is the one to emulate.

### Integration patterns worth noting

- **`pytorch/xla` → Pallas**: trace + StableHLO payload cache (`trace_pallas_arg_to_payload`, `_xla_increment_counter('trace_pallas_cache_hit')`) — if we benchmark `torch_xla` paths, trace-payload cache hits are a first-order variable.
- **`tunix`, `aqt`, `qwix` → Pallas**: pytree lifting (`pallas_call` wrappers aware of `QArray` + scale tiles) — the right layer to introduce new quantization optimization hypotheses.
- **`marin/levanter` → Pallas**: implementation-dispatcher + autotuner (`IMPLEMENTATIONS: dict[str, ArrayImpl]`, `_DEFAULT_IMPLEMENTATION` tuple, `try/except ImportError` for optional backends) — template pattern for any new kernel needing both semantic fallback and speed path.

## Sources

- Web-research agent, 2026-04-23.

## See also

- [Directory main page](../2026-04-23-pallas-kernel-directory.md)
- §1 [Upstream JAX + tokamax](01-upstream-jax-tokamax.md)
- §3 [Inference engines](03-inference-engines.md)
