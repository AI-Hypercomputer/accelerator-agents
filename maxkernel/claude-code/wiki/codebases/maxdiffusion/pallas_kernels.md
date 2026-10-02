---
title: "maxdiffusion — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, maxdiffusion]
created: 2026-07-04
updated: 2026-07-04
---

MaxDiffusion ships **six locally-defined Pallas kernels, all `mosaic_tpu`**, split across two files under
`src/maxdiffusion/kernels/`: a **diffusion-tuned fork of jax-ml/jax splash attention**
(`splash_attention/splash_attention_kernel.py` — forward tile kernel, a ring-aware forward variant, and
two backward kernels) and a **novel custom forward flash-attention kernel** authored in-repo
(`custom_splash_attention.py` — a plain fwd kernel plus a multi-heads-per-tile variant, tuned for 720p
Wan2.1 on v6e/v7x and wired into a TorchAX SDPA path). A seventh surface,
`splash_attention/ring_attention_kernel.py`, is a **ring-attention orchestrator, not a new tile kernel** —
it composes the splash forward primitives with `lax.ppermute` (see Notes). The `custom_splash_attention.py`
kernels are **new since the 2026-04-23 directory catalog** (file added 2026-06-26) and are not yet in that
catalog's rows.

## Kernels

### Attention — splash fork (vendored from jax-ml/jax, diffusion-tuned)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `flash_attention_kernel` | mosaic_tpu | Tile-level forward flash kernel invoked from `_splash_attention_forward`; online-softmax accumulation over `block_q × block_kv` tiles, MQA/segment-ids/soft-cap, no `[L,L]` materialization. `dimension_semantics=("parallel","arbitrary")`. | [splash_attention_kernel.py#L282](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L282) | stable | vendored-from-jax (splash), diffusion-tuned |
| `_splash_attention_forward_ring_raw` | mosaic_tpu | Ring-aware forward: emits the **un-normalized softmax numerator plus running stats** (`m` max-logit, `l` denominator) instead of the normalized output, so an outer ring loop can merge per-shard contributions and normalize once at the end. `pl.pallas_call` at L1097. | [splash_attention_kernel.py#L862](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L862) | stable | novel addition atop vendored splash |
| `_flash_attention_dq_kernel` | mosaic_tpu | Backward pass computing `dq` (unfused bwd path, used when `use_fused_bwd_kernel=False`). | [splash_attention_kernel.py#L1277](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1277) | stable | vendored-from-jax (splash) |
| `_flash_attention_dkv_kernel` | mosaic_tpu | Backward pass computing `dk`/`dv`; also the fused-bwd body (single `pl.pallas_call` at L1969 emits `dq_unreduced, dk, dv`). | [splash_attention_kernel.py#L1394](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1394) | stable | vendored-from-jax (splash) |

### Attention — custom diffusion flash (novel-to-this-repo, forward-only)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_flash_attention_kernel` | mosaic_tpu | In-house **forward-only** flash kernel for bidirectional (no causal mask) diffusion attention. 3-D grid, `PrefetchScalarGridSpec`; inner `bkv_compute`/`bkv_compute_in` two-level KV tiling (default 1024 / 256 — "cranked up for MXU throughput" / "kept at 256 to protect VPU registers"); base-2 exp softmax; supports a ring path (writes `l_ring_ref`/`m_ring_ref` stats when un-normalized output is requested). Defaults `bq=3328, bkv=2816`, "tuned for 720p Wan2.1 on v6e/v7x". | [custom_splash_attention.py#L54](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L54) | stable (in production SDPA path) | novel |
| `_flash_attention_kernel_mhpt` | mosaic_tpu | Multi-heads-per-tile variant of the above: loops `heads_per_tile` heads inside one grid cell (extra leading tile dim on q/k/v/scratch refs) to raise MXU occupancy when per-head work is small. Selected when `heads_per_tile > 1`. | [custom_splash_attention.py#L212](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L212) | stable | novel |

**Imported / re-exported (not defined here):** `make_splash_mha` / `make_splash_mqa` /
`make_dynamic_splash_mha` (splash_attention_kernel.py) are factory wrappers around the vendored splash
kernels, not separate Pallas kernels. When `attention_kernel == "tokamax_flash"`, `attention_flax.py`
calls **tokamax's** splash `make_splash_mha` (upstream, not copied here). The
[Tokamax ring-attention integration](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py)
defines no `pl.pallas_call` of its own — see Notes.

## Notes

**Vendoring lineage.** `splash_attention/{splash_attention_kernel,splash_attention_mask,splash_attention_mask_info,base}.py`
are a diffusion-tuned fork of
[`jax.experimental.pallas.ops.tpu.splash_attention`](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention)
(copyright "2025 DeepMind Technologies Limited"). Diffusion-specific additions over upstream: `SplashConfig`,
`QKVLayout`, `get_kernel_name`, the `_splash_attention_forward_ring_raw` variant, and tuning for long
video/image sequences (frames × resolution / VAE / patchify ratios) with MQA + segment-ids + logit soft-cap;
sinks are unsupported. Backward is fused by default in the tokamax flavor (`block_q_dq`/`block_kv_dq` unused).

**The `custom_splash_attention.py` kernels are the real novelty** — despite the "splash" filename they are a
from-scratch forward flash implementation (own `make_splash_mha`, distinct from the fork's factory of the same
name), targeting inference/denoising. `tpu_custom_attention` wraps them in a `shard_map` with q/kv partition
specs that adapt to 1-D (TP), 2-D (DP×TP or Ulysses), and 3-D meshes; `make_custom_splash_sdpa` exposes them
as a **TorchAX `scaled_dot_product_attention`** replacement. No backward kernel exists here — forward path only.

**Autotune surface.** Custom kernel: `block_q`, `block_kv`, `block_kv_compute`, `block_kv_compute_in`,
`heads_per_tile`, `use_base2_exp`, `use_experimental_scheduler` (sets Pallas flag
`XLA_TPU_FORCE_LP_LLO_SCHEDULER`), `vmem_limit_bytes` — all overridable via a `flash_block_sizes` dict. Splash
fork: `SplashConfig` / `BlockSizes` (`block_q`, `block_kv`, `block_kv_compute`, `block_q_dkv`, `block_kv_dkv`,
`block_kv_dkv_compute`, `use_fused_bwd_kernel`, `QKVLayout`). Source comment: block sizes are "key to
saturating HBM bandwidth… It is highly recommended to tune them."

**Ring attention (integration, not a new kernel).**
`ring_attention_kernel.py` (`make_ring_attention`, `_ring_attention_fwd`/`_ring_attention_bwd`,
`RingSplashAttentionKernel`) rotates K/V across a `ring_axis` mesh axis via `lax.ppermute`, calling the splash
forward primitives (`_splash_attention_forward`, `_splash_attention_forward_ring_raw`, `_splash_attention_bwd`)
per ring step and doing online-softmax combine across shards — a `custom_vjp`-wrapped orchestrator with
segment-id rotation and torus-aware nearest-neighbor permutations, but **no `pl.pallas_call` of its own** (the
tile compute is the shared splash kernel). This is MaxDiffusion's Tokamax ring-attention integration point on
TPU (announced 2026-04-16), selected by `attention_kernel == "tokamax_ring"` in
[`models/attention_flax.py`](../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py).

**Hardware matrix.** All `mosaic_tpu`; no Mosaic-GPU or Triton kernels in this repo. Custom kernel defaults are
explicitly tuned for v6e/v7x at 720p Wan2.1 resolution.

**Divergence from the directory catalog.** The 2026-04-23 catalog (subpage 02) lists the splash fork and ring
kernel but predates `custom_splash_attention.py` (added 2026-06-26, HEAD `09844570`); the two custom kernels
above are new rows relative to that catalog.

## See also
- [maxdiffusion — overview](overview.md) — the repo's grounded wiki front door.
- [maxdiffusion.kernels.splash_attention.splash_attention_kernel](concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md) — grounded concept page for the splash fork.
- [custom_splash_attention (catalog)](catalog/src/maxdiffusion/kernels/custom_splash_attention.md) — per-symbol catalog page for the novel kernel.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 02 — AI-Hypercomputer](../../analyses/pallas-kernel-directory/02-ai-hypercomputer.md).

## Sources
- [raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py) — commit `09844570`
- [raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py)
- [raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py](../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py)
