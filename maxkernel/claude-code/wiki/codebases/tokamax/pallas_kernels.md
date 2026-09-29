---
title: "tokamax — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, tokamax]
created: 2026-07-04
updated: 2026-07-04
---

tokamax is a dedicated Pallas kernel library — every transformer/MoE primitive is authored here as a set
of hardware-specific kernels behind one uniform [`Op`](concepts/tokamax-_src-ops-op.md) abstraction,
so this repo has the **richest backend matrix** of any repo in the catalog: nearly every op ships
`mosaic_tpu`, `triton`, `mosaic_gpu_sm90` (Hopper), and `mosaic_gpu_sm100` (Blackwell) implementations
side by side. ~30 locally-defined Pallas kernels span attention (splash/ring TPU, Triton, Mosaic-GPU
sm90/sm100, all with matching VJP kernels), ragged_dot / grouped-matmul (a new **v2 gmm/tgmm** Mosaic-TPU
pair plus a full sm100 quantization family), flex_attention, gated_linear_unit, normalization,
linear-softmax-cross-entropy, ragged_gather, and an experimental DeepSeek **MLA** ragged-paged-attention
TPU kernel. The kernels are largely **novel-to-this-repo**: the TPU splash/ring copy is an extended fork
of upstream JAX, and the Mosaic-GPU kernels compose upstream `plgpu`/`mgpu` building blocks but implement
their own pipelines. Source is commit `946ae78` — materially ahead of the 2026-04-23 directory catalog
(commit `54bdd95`); divergences noted below.

## Kernels

### Attention — TPU (splash / ring)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `flash_attention_kernel` (splash) | mosaic_tpu | Block-sparse "Splash" flash attention; skips fully-masked blocks via `MaskInfo`; base-2 softmax (`LOG2E`) for the faster `exp2` instruction; MHA/MQA/GQA | [splash_attention_kernel.py#L299](../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L299) | research/experimental | novel (extended fork of jax splash) |
| `PallasMosaicTpuFlashAttention` (wrapper) | mosaic_tpu | Public TPU SDPA op — thin `Op` wrapper dispatching to the tokamax-internal splash kernel (not upstream) | [pallas_mosaic_tpu.py#L60](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L60) | experimental | novel (dispatcher) |
| splash VJP | mosaic_tpu | Splash backward (dq / dkv) | [pallas_mosaic_tpu_vjp.py](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py) | experimental | novel |
| `RingSplashAttentionKernel` | mosaic_tpu | Ring Attention — KV-sharded across a device ring; re-slices `MaskInfo` per ring step for long-context sequence parallelism; custom VJP | [ring_attention_kernel.py#L531](../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L531) | research | novel (builds on splash) |

### Attention — GPU (Triton + Mosaic-GPU sm90/sm100)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_fwd_kernel` (Triton attn) | triton | FlashAttention forward for pre-Hopper NVIDIA; `use_base2` scales logits for `exp2` softmax | [pallas_triton.py#L47](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L47) | experimental | novel |
| Triton attn VJP | triton | Triton backward (dq/dk/dv/ds) | [pallas_triton_vjp.py#L468](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L468) | experimental | novel |
| sm90 flash attn fwd | mosaic_gpu_sm90 | H100 FlashAttention; WGMMA + TMA warp-specialized pipeline (`compute_wgs` fan-out) | [pallas_mosaic_gpu_kernel_sm90.py#L567](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L567) | experimental | novel |
| sm90 flash attn VJP | mosaic_gpu_sm90 | Hopper backward, split dq/ds and dk/dv kernels | [pallas_mosaic_gpu_vjp_kernel_sm90.py#L705](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L705) | experimental | novel |
| sm100 flash attn fwd | mosaic_gpu_sm100 | B200 FlashAttention; collective 2-CTA MMA (`collective` / `num_tma_splits`), TCGEN05, packed-bit mask | [pallas_mosaic_gpu_kernel_sm100.py#L949](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L949) | experimental | novel |
| sm100 flash attn VJP | mosaic_gpu_sm100 | Blackwell backward | [pallas_mosaic_gpu_vjp_kernel_sm100.py#L435](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L435) | experimental | novel |

### Attention — MLA (experimental, TPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_mla_ragged_paged_attention_kernel` | mosaic_tpu | DeepSeek-style **Multi-head Latent Attention** ragged-paged-attention: compressed/latent KV, mixed prefill+decode over paged KV, sub-word VMEM packing of newly-computed KVs | [pallas_mosaic_tpu_kernel.py#L257](../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L257) | research/experimental | novel |

### Flex attention

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| flex attention fwd | triton | PyTorch-style FlexAttention — inlines user `score_mod`/`mask_mod` via `jax.experimental.pallas.fuser`; returns `out, l, m` | [pallas_triton.py#L447](../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L447) | experimental | novel |

### MoE / grouped matmul (ragged_dot)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| gmm (v1) | mosaic_tpu | MoE forward grouped matmul on TPU; `custom_buffered_pallas_call`; `qwix` quantization | [pallas_mosaic_tpu_kernel.py#L594](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L594) | experimental | novel |
| tgmm (v1) | mosaic_tpu | Transposed (backward) grouped matmul on TPU | [pallas_mosaic_tpu_kernel.py#L923](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L923) | experimental | novel |
| **gmm v2** | mosaic_tpu | Rewritten forward GMM; gates dequant timing on the MXU column size | [pallas_mosaic_tpu_v2_gmm_kernel.py#L1307](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L1307) | experimental | novel (new since catalog) |
| **tgmm v2** | mosaic_tpu | Rewritten transposed GMM; accounts for XLU-transpose-caching VMEM cost | [pallas_mosaic_tpu_v2_tgmm_kernel.py#L761](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L761) | experimental | novel (new since catalog) |
| ragged_dot (Triton) | triton | MoE grouped matmul on pre-Hopper / generic GPU; `qwix` quant | [pallas_triton.py#L225](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L225) | experimental | novel |
| sm90 ragged_dot | mosaic_gpu_sm90 | MoE grouped matmul on H100; builds on upstream `ragged_dot_mgpu` | [pallas_mosaic_gpu_kernel_sm90.py#L165](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L165) | experimental | novel (composes jax mgpu) |
| sm90 ragged_dot quant | mosaic_gpu_sm90 | Quantized-weight MoE on H100 | [pallas_mosaic_gpu_kernel_sm90_quant.py#L357](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L357) | experimental | novel |
| sm100 ragged_dot | mosaic_gpu_sm100 | MoE on B200 (`plgpu.kernel`, TMEM) | [pallas_mosaic_gpu_kernel_sm100.py#L236](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L236) | experimental | novel |
| sm100 ragged_dot quant | mosaic_gpu_sm100 | Generic quantized MoE on B200 | [pallas_mosaic_gpu_kernel_sm100_quant.py#L526](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L526) | experimental | novel (new since catalog) |
| sm100 ragged_dot fp8 quant | mosaic_gpu_sm100 | FP8-weight MoE on B200 (int4→f8e4m3 dequant via TMEM) | [pallas_mosaic_gpu_kernel_sm100_fp8_quant.py#L841](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant.py#L841) | experimental | novel (new since catalog) |
| sm100 ragged_dot i8 quant | mosaic_gpu_sm100 | INT8-weight MoE on B200 | [pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L649](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L649) | experimental | novel (new since catalog) |
| sm100 ragged_dot post-scale quant | mosaic_gpu_sm100 | Post-scale quantized MoE on B200 | [pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L493](../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L493) | experimental | novel (new since catalog) |

### Gated linear unit (GLU / SwiGLU / GEGLU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| GLU (Triton) | triton | Fused GLU/SwiGLU/GEGLU on pre-Hopper / generic GPU | [pallas_triton.py#L170](../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L170) | experimental | novel |
| GLU sm90 | mosaic_gpu_sm90 | Fused GLU on H100; WGMMA pipeline | [pallas_mosaic_gpu_kernel_sm90.py#L151](../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L151) | experimental | novel |
| GLU sm100 | mosaic_gpu_sm100 | Fused GLU on B200 | [pallas_mosaic_gpu_kernel_sm100.py#L260](../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L260) | experimental | novel |

*(TPU has no Pallas GLU backend — falls back to XLA.)*

### Norm / loss / gather

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_normalization_kernel` (LN/RMS) | triton | Fused LayerNorm / RMSNorm forward; `subtract_mean=False` → RMS | [pallas_triton.py#L166](../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L166) | experimental | novel |
| normalization VJP | triton | LN/RMS backward (dx/dscale/doffset); separate tuning config | [pallas_triton_vjp.py#L149](../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L149) | experimental | novel |
| linear-softmax-CE fwd | mosaic_tpu | Fused final-projection + softmax + cross-entropy on TPU, tiled batch×hidden×vocab; the sole TPU-only kernel in README's supported matrix (arXiv:2410.10989) | [pallas_mosaic_tpu_kernel.py#L301](../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L301) | experimental | novel |
| linear-softmax-CE bwd | mosaic_tpu | Backward: x_grad / w_grad | [pallas_mosaic_tpu_kernel.py#L644](../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L644) | experimental | novel |
| `main_kernel` (ragged gather) | mosaic_tpu | Optimized ragged gather — gathers rows within a dynamic `[start, end)` index range | [pallas_mosaic_tpu_kernel.py#L26](../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py#L26) | experimental | novel (new since catalog) |

### Imported / not-a-Pallas-kernel (noted, not counted)

- `attention/xla_chunked.py`, `linear_softmax_cross_entropy_loss/chunked_xla.py` — pure-XLA fallbacks.
- `attention/jax_nn.py` — wraps `jax.nn.dot_product_attention` (cuDNN), not Pallas.
- `triangle_multiplication/` — op directory exists (AlphaFold triangle multiplicative update) but is
  `Implementation = Literal["xla"]` only; **no Pallas backend yet**.
- Per-op `base.py` (XLA reference) and `*_common.py` (`Config` dataclasses) — not kernels.

## Notes

**Backend matrix (the point of this repo).** Attention and ragged_dot each ship the full sweep —
`mosaic_tpu` + `triton` + `mosaic_gpu_sm90` + `mosaic_gpu_sm100`, forward and VJP. GLU is GPU-only
(Triton + sm90 + sm100, no TPU Pallas). Normalization is Triton-only. linear-softmax-CE, ragged_gather,
and MLA are TPU-only. `triangle_multiplication` is XLA-only.

**Vendoring lineage.** The TPU splash/ring/mask/mask_info files are an **extended fork** of upstream
`jax.experimental.pallas.ops.tpu.splash_attention` — tokamax carries its own copy with additional mask
support and ring integration, and dispatches to it (not upstream) from
[`PallasMosaicTpuFlashAttention`](../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L31).
The Mosaic-GPU kernels **compose** upstream `plgpu`/`mgpu` primitives (`plgpu.kernel`,
`mgpu_lib.tiled_swizzled_smem`, `plgpu.inline_mgpu`, TCGEN05 helpers) but the pipelines are authored here;
sm90 ragged_dot builds on upstream `ragged_dot_mgpu`, sm100 on `blackwell_ragged_dot_mgpu`.

**Autotune surface.** Every op resolves a `Config` via the [`Op`](concepts/tokamax-_src-ops-op.md)
five-tier chain (explicit → cached autotune → live autotune → heuristics → error). `pydantic`-validated
`Config` dataclasses enforce hardware invariants at construction (block sizes multiple of 64, at-most-one
cluster axis > 1, epilogue tiles evenly dividing main tiles). ragged_dot's `GroupSizes` carries a
*representative* (not runtime-true) group distribution so autotuned configs survive step-to-step group
churn.

**Micro-optimizations.** Base-2 softmax recurs: Triton attention (`use_base2` → `jnp.exp2`), flex
attention, and TPU splash (`LOG2E`/`LOG2E_INV`) all scale logits for the faster `exp2` instruction, then
convert LSE residuals back to natural-log units. The v2 GMM/TGMM TPU kernels gate dequant timing on the
MXU column width and account for XLU-transpose-caching VMEM cost.

**Divergence from the 2026-04-23 catalog (`54bdd95` → `946ae78`).** New here and absent from
[subpage 01](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md): the ragged_dot **v2
gmm/tgmm** Mosaic-TPU kernels, the sm100 quantization family (`quant`, `fp8_quant`, `i8_quant`,
`quant_post_scale`), the **ragged_gather** Mosaic-TPU kernel, and a fully-realized **MLA** ragged-paged
Mosaic-TPU kernel (the catalog listed MLA as base-only / TBD).

## See also

- [tokamax — overview](overview.md) — the repo's grounded wiki front door.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the
  cross-repo Pallas catalog; this repo appears in
  [subpage 01 — Upstream JAX + tokamax](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md).
- [Op abstraction](concepts/tokamax-_src-ops-op.md) — the bind → configure → dispatch contract every kernel sits behind.
- Attention: [base](concepts/tokamax-_src-ops-attention-base.md),
  [Triton](concepts/tokamax-_src-ops-attention-pallas_triton.md),
  [Mosaic-GPU common](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md),
  [sm100 fwd](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md),
  [sm100 VJP](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md),
  [sm90 VJP](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90.md).
- Splash/ring TPU: [splash kernel](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md),
  [ring kernel](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel.md),
  [mask](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md),
  [mask_info](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask_info.md).
- MLA: [experimental MLA TPU kernel](concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md).
- Flex attention: [Triton flex attention](concepts/tokamax-_src-ops-flex_attention-pallas_triton.md).
- ragged_dot: [base](concepts/tokamax-_src-ops-ragged_dot-base.md),
  [Mosaic-TPU](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md),
  [v2 gmm](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md),
  [v2 tgmm](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel.md),
  [Triton](concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md),
  [Mosaic-GPU common](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common.md).
- GLU: [base](concepts/tokamax-_src-ops-gated_linear_unit-base.md),
  [Mosaic-GPU common](concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md).
- [Pallas BlockRef](concepts/tokamax-_src-pallas-block.md) — partial-block boundary handling used by these kernels.

## Sources

- [raw/code/tokamax](../../../raw/code/tokamax) — commit `946ae78`, `tokamax/_src/ops/**` (read directly).
- [wiki/analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) — prior cross-repo rows (commit `54bdd95`).
