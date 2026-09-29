---
title: "marin (marin-community/marin) — narrow: levanter Pallas kernels + autotune harness"
type: codebase
tags: [marin, levanter, pallas, autotune-harness, fused-cross-entropy-loss, mamba, ssd, deployment-time-tuner, crown-jewel]
repo: marin-community/marin
commit: 7a56e016d
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [marin overview](marin/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Ingestion narrow-scoped to `lib/levanter/src/levanter/kernels/pallas/` — the **deployment-time autotune harness** and fused-cross-entropy-loss kernels. **The single most-valuable non-kernel artifact surveyed in this wiki**: a kernel-agnostic, shard-aware, compile-cost-aware, GCS-persistent autotuner — superior to tokamax's write-time autotune for an autoresearch loop. The patterns this wiki should emulate. Per-kernel detail in [§5.8 of the Pallas kernel directory](../analyses/pallas-kernel-directory/05-frameworks-quant.md#58-marin-communitymarin-vendors-levanter).

## Architecture

```
lib/levanter/src/levanter/kernels/pallas/
  fused_cross_entropy_loss/
    api.py                     ← implementation-dispatcher (TPU / GPU / XLA); autotune flow lives here
    pallas_tpu.py              ← tokamax-derived linear_softmax_cross_entropy_loss (fwd + streaming bwd)
    pallas_gpu.py              ← triton counterpart (H100 stable, GB10 experimental)
    tuned_block_sizes.py       ← hand-curated fallback BlockSizes dict (per-device-kind)
    ...
  ssd/api.py + xla.py          ← SSD; TPU Pallas deliberately absent (explicit PallasUnsupportedError)
  mamba3/api.py                ← Mamba3 MIMO; XLA only
  template_kernel.py           ← scaffold for new kernels (pattern template)
  autotune_utils.py            ← 161 lines — the core of the harness
  cost_estimate_utils.py       ← 30 lines — IO-bytes-accessed override
  autotune_cache_utils.py      ← ~120 lines — GCS-aware persistent cache
```

## Key abstractions

| Component | File | What it is |
|---|---|---|
| `linear_softmax_cross_entropy_loss_pallas` (fwd + streaming bwd) | `fused_cross_entropy_loss/pallas_tpu.py` | **Tokamax-derived**; additions: logsumexp penalty, logit soft-cap (`tanh(logits/cap)*cap`), external loss weighting, optional argmax return. `pl.estimate_cost`-based `_fwd_cost_estimate` + `_backward_cost_reference` |
| Same on GPU | `fused_cross_entropy_loss/pallas_gpu.py` | Triton (mosaic_gpu fallback); stable on H100/A100, experimental on GB10. Hard constants: `_NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376` (per-SM shared minus ~131 KB overhead), `_GB10_MAX_H_TILES = 512`, `_GB10_FULL_MATMUL_MAX_OUTPUT_ELEMENTS = 67_108_864` |
| `autotune_utils.py` | — | The harness; `sharding_of`, `should_offload_compile`, `compile_benchmark_fn`, `maybe_wrap_in_shard_map`, single-thread `ThreadPoolExecutor(max_workers=1)` named `"pallas_autotune"` |
| `cost_estimate_utils.with_io_bytes_accessed` | — | Overrides `pl.CostEstimate.bytes_accessed` with `sum(prod(shape) * itemsize)` over real IO — because `pl.estimate_cost` over-counts compute intermediates and under-counts IO |
| `autotune_cache_utils` | — | Filesystem-backed cache at `<jax_compilation_cache_dir>/levanter_kernel_autotune/<kernel>/<file>`, `rigging.filesystem.url_to_fs` (GCS-aware, same bucket as PJRT compile cache) |
| `_autotune_jaxpr_hash` | `fused_cross_entropy_loss/api.py` | `sha256(str(jax.make_jaxpr(...).jaxpr))[:16]` — pins cache key to jaxpr shape |
| `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` | `fused_cross_entropy_loss/api.py` | Discards candidates whose compile time alone > 0.2 s vs baseline |
| `_is_tpu_vmem_compile_error` / `_warn_vmem_compile_fallback_once` | `fused_cross_entropy_loss/api.py` | VMEM-OOM-aware fallthrough at lowering |
| `template_kernel.py` | — | Scaffold showing the expected structure (reference + Pallas impl + dispatcher with `implementation=` arg). Docstring: *"This file is intentionally not 'the best kernel'. It is a scaffold showing the expected structure..."* |

## Entry points

```bash
cd raw/code/marin/lib/levanter

python -c "from levanter.kernels.pallas.fused_cross_entropy_loss import api"
```

## Dependencies

- `jax` / `jaxlib` / `libtpu` (TPU) or `triton` (GPU) for Pallas backends.
- `tokamax` (upstream for the tpu kernel).
- `rigging` (filesystem URL → fs helper — used for GCS paths).

## Performance-relevant surfaces

### 1. Deployment-time kernel-agnostic autotune harness
**The pattern to emulate in this wiki's autoresearch loop.** Six distinguishing properties vs tokamax's write-time autotuner:

1. **Compile-time-aware filtering.** `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` discards candidate block sizes whose XLA compile time alone dominates the training step. At autoresearch scale, this matters more than kernel wall-time.
2. **VMEM-OOM-aware fallthrough.** Demotes candidates that hit `resource_exhausted … vmem` at lowering and moves to the next impl instead of raising.
3. **Sharding-preserving benchmark lowering.** Keeps real `NamedSharding` when input is globally sharded; strips only under `shard_map` manual tracers. Tunes on the exact mesh-shard combination production sees.
4. **Off-thread compile for mesh-bound contexts.** Single-worker `ThreadPoolExecutor` lets the tuner run while the main thread holds JIT / mesh state; switch on `jax_core.unsafe_am_i_under_a_jit_DO_NOT_USE()`.
5. **jaxpr-hashed cache keys.** Silent shape/dtype/soft-cap changes invalidate cache.
6. **GCS-aware persistent cache reuse.** Writes under `<jax_compilation_cache_dir>/levanter_kernel_autotune/` — the **same bucket PJRT caches compiles into**. Every training job shares tuning transparently.

### 2. `linear_softmax_cross_entropy_loss` — the canonical fused CE reference
Tokamax upstream + these levanter enhancements (logsumexp penalty, logit soft-cap, loss weighting, argmax) are the public reference. Useful for any CE-loss hypothesis.

### 3. GPU hard constants worth recording
`_NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376` is the weight-tile limit, same on GB10 and H100. Other per-device caps: `_GB10_MAX_H_TILES = 512`; V-block caps per device. These are concept-facts.

### 4. `template_kernel.py` — scaffold for any future kernel
`reference JAX impl → Pallas-accelerated impl(s) → dispatcher with implementation=`. Same structure Tokamax uses. Usable as the template when this wiki writes its own Pallas kernel.

### 5. IO-bytes-accessed override (`with_io_bytes_accessed`)
`pl.estimate_cost` over-counts compute intermediates and under-counts IO for a real Pallas tile. `cost_estimate_utils.with_io_bytes_accessed` copies FLOPs / transcendental / remote-bytes fields + overwrites `bytes_accessed` with `sum(prod(shape) * itemsize)` over real IO. Worth citing when reading any Pallas roofline in the wiki.

### 6. SSD/Mamba3 deliberately XLA-only (the absent-Pallas signal)
`ssd/api.py` raises `PallasUnsupportedError("SSD TPU Pallas kernel is intentionally absent; use the XLA path.")` Same signal as exp 33's "Pallas loses when XLA already fuses" — levanter maintainers arrived at the same conclusion. Counter-evidence for any future "write Pallas SSD" hypothesis.

## Connections

- **[Pallas kernel directory §5.8](../analyses/pallas-kernel-directory/05-frameworks-quant.md#58-marin-communitymarin-vendors-levanter)** — per-kernel detail + harness description.
- [tokamax](tokamax.md) — upstream for `linear_softmax_cross_entropy_loss` and autotune API; levanter's harness is the deployment-time counterpart to tokamax's write-time tuner.
- [pallas-forge](pallas-forge.md) — also an autotune framework but ships forward-only kernels (no custom_vjp); the marin harness here is kernel-agnostic, training-ready.
- [axlearn](axlearn.md) — has Pallas SSM/Mamba; marin deliberately doesn't. Counterpoint.
- Gemma4 ceiling analysis — persistent compile cache (named as a lever) is one of the properties this harness already exercises.

## See also

- [autotuning](../concepts/autotuning.md)
- [memory-efficient-cross-entropy](../concepts/memory-efficient-cross-entropy.md)
- [pallas-kernel](../concepts/pallas-kernel.md)
- [vmem](../concepts/vmem.md)

## Sources

- `raw/code/marin/lib/levanter/src/levanter/kernels/pallas/`
- Upstream: <https://github.com/marin-community/marin>
