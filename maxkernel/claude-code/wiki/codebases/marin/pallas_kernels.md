---
title: "marin — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, marin, levanter]
created: 2026-07-04
updated: 2026-07-04
---

`marin` (via its vendored [levanter](overview.md) snapshot) authors exactly **one genuine TPU Pallas
kernel** — the fused linear-softmax cross-entropy loss (`mosaic_tpu`), heavily derived from Tokamax
and extended for Levanter's API. Everything else in the kernel tree is either an **XLA/reference
implementation with the Pallas backend deliberately left absent** (SSD, Mamba3), a **`triton` GPU
kernel** (grug MoE token gather/sum), or a **CUTLASS CuTe DSL GPU kernel** (grug FlashAttention-4,
`sm90`) — none of which are Pallas. The real reusable asset alongside the one TPU kernel is the
**deployment-time autotune harness** (subpage 05 calls it "the most portable deployment-time tuner").
Source diverged since the 2026-04-23 catalog (see Notes).

## Kernels

### Cross-entropy loss (Pallas — TPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `linear_softmax_cross_entropy_loss` (fwd + streaming-V bwd) | mosaic_tpu | Fused `x @ w → logits → softmax → CE` in one kernel; forward emits per-lane LSE, backward streams over V-tiles to avoid materializing the full logits matrix. The one `pl.pallas_call` in the repo. | [fused_cross_entropy_loss/pallas_tpu.py#L281](../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py#L281) · [catalog](catalog/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.md) | stable | vendored-from-tokamax (+ extended: logsumexp penalty, logit soft-cap, external loss weighting) |

### MoE token combine (Triton — GPU, in grug)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_sonic_token_gather_sum_kernel` | triton | MoE "sonic" combine: gathers expert-routed token rows and sums the top-k contributions per token in one pass (fused scatter-combine), tiled over the hidden dim. | [grug/_moe/sonic.py#L45](../../../raw/code/marin/lib/levanter/src/levanter/grug/_moe/sonic.py#L45) | experimental (GPU-only; requires `jax-triton`) | novel |
| `_sonic_token_gather_sum_bwd_kernel` | triton | Backward of the sonic gather/sum — scatters output cotangents back to the per-expert token rows. | [grug/_moe/sonic.py#L100](../../../raw/code/marin/lib/levanter/src/levanter/grug/_moe/sonic.py#L100) | experimental (GPU-only) | novel |

## Notes

**The one TPU kernel is Tokamax-derived, explicitly.** `pallas_tpu.py` header: *"This implementation is
heavily based on Tokamax's linear softmax cross-entropy Pallas Mosaic TPU kernel (Apache-2.0). We adapt
it for Levanter's API and add optional logsumexp penalty, logit soft-cap (`tanh(logits/cap)*cap`),
external loss weighting support."* It carries `pl.estimate_cost`-based `_fwd_cost_estimate` /
`_backward_cost_reference` and uses `with_io_bytes_accessed` from `cost_estimate_utils.py`. Env var
`LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH` toggles whether the backward uses the XLA streaming
path. The `api.py` dispatcher follows the Tokamax `implementation=` pattern with a best-available
fallback order.

**Scaffolded-but-absent Pallas (XLA-only, NOT kernels).** Two subsystems ship a Pallas *slot* that is
deliberately empty:
- **SSD** (`ssd/api.py`) — `ssd_intra_chunk_pallas` raises
  `PallasUnsupportedError("SSD TPU Pallas kernel is intentionally absent; use the XLA path.")`. The
  dispatcher lists `pallas_tpu` but it falls back to XLA. Only `reference` and `xla` are real.
- **Mamba3 MIMO** (`mamba3/api.py`) — `Implementation` is `Literal["xla", "reference"]`; there is no
  `pallas_tpu` entry at all.

These are not padded into the table because there is no local Pallas implementation to reuse.

**`splash_attention.py` is a wrapper, not a kernel.** It imports `splash_attention_kernel` /
`splash_attention_mask` from `jax.experimental.pallas.ops.tpu.splash_attention` and only provides
Levanter-side mask/segment-id lowering helpers — the kernel body is upstream JAX.

**grug FlashAttention-4 is CUTLASS CuTe, not Pallas.** `grug/attention/_fa4_cute*.py` and `_fa4_thd.py`
author flash-attention forward/backward launchers (`segmented_flash_attention_forward_launcher`,
`segmented_flash_attention_backward_sm90_launcher`, …) in the **CUTLASS CuTe Python DSL** and dispatch
them through JAX `cutlass.jax.cutlass_call` (`cute.jit`), targeting NVIDIA `sm90` (Hopper). They are
genuine hand-written GPU kernels but use neither `pl.pallas_call` nor Mosaic/Triton, so they are out of
scope for this Pallas inventory — listed here only so the reader knows the repo has a second,
non-Pallas custom-kernel path.

**Autotune harness (reusable, not a kernel).** `kernels/pallas/autotune_utils.py` +
`cost_estimate_utils.py` + `autotune_cache_utils.py` form a **deployment-time** tuner that subpage 05
flags as the most portable in the survey. What it does that Tokamax's write-time autotuner does not:
- **Compile-cost filtering** — `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` discards candidate block sizes
  whose XLA compile time alone would dominate a training step.
- **Sharding-preserving benchmark lowering** — `shape_dtype_struct_for_benchmark` keeps the real
  `NamedSharding` (tunes on the exact mesh-shard the production run uses, not a single-device abstract).
- **GCS-aware persistent cache** — `autotune_cache_utils.py` writes under
  `<jax_compilation_cache_dir>/levanter_kernel_autotune/<kernel>/` via `rigging.filesystem.url_to_fs`,
  the same bucket PJRT caches compiles into, so every job shares tuning transparently.
- Gated by `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS` (default on).

**`template_kernel.py`** is a scaffold, not a kernel: reference oracle → accelerated impl → stable
`implementation=`-selecting API, matching the Tokamax pattern for future kernel authoring.

**Source divergence vs the 2026-04-23 catalog.** The catalog (subpage 05 §5.8) listed a second Pallas
kernel `linear_softmax_cross_entropy_loss_pallas_gpu` in `fused_cross_entropy_loss/pallas_gpu.py`
(`triton` / mosaic_gpu fallback, with GB10/H100 shared-memory constants). **That file is absent** in
the current checkout (submodule HEAD `240bf20a8`; the `marin.md` codebase page pins the older
`7a56e016d`). The current GPU kernels instead live under the newer `grug/` tree (the sonic Triton
combine above + the CuTe attention). Treat the catalog's `pallas_gpu.py` row as stale.

## See also
- [marin (levanter) — overview](overview.md) — the repo's grounded wiki front door.
- [pallas_tpu — catalog](catalog/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.md) — per-symbol grounded page for the one TPU kernel.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 05 — frameworks & quant §5.8/§5.9](../../analyses/pallas-kernel-directory/05-frameworks-quant.md).
