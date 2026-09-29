---
title: "easydel — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, easydel]
created: 2026-07-04
updated: 2026-07-04
---

EasyDeL's attention/inference-op registry under `easydel/operations/kernels/` is *mostly* thin
adapters re-exporting `ejkernel` TPU Pallas kernels (flash, blocksparse/Splash, ragged/paged,
MLA, SSM, ring) — those are **not defined here** and are listed in the "Imported (not defined
here)" note. But two files diverge from the pure-adapter story and author **6 novel-to-EasyDeL
`mosaic_tpu` Pallas kernels** for the linear-attention *decode* path — a grouped Gated-Delta-Rule
step, a fused conv-state/depthwise-conv/SiLU step, a fused Gated-DeltaNet decode step, and two
triangular-matrix-inverse kernels (Newton–Schulz and block forward-substitution) used by the
chunked GDR prefill. All are single-token/ragged decode kernels, all `bf16`/`fp32`, all fwd-only
(the training path for these ops lives in `ejkernel`). This corrects the 2026-04-23 directory's
§6.2 claim that EasyDeL kernels are *only* thin re-export adapters.

## Kernels

All backend `mosaic_tpu`. Grouped by functional category.

### Linear attention — Gated Delta Rule / Gated DeltaNet decode

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_gdr_grouped_decode_kernel` | mosaic_tpu | Grouped GDR (Gated Delta Rule) single decode step; grid `(batch,)`, one program per batch element entirely in VMEM, iterating all `num_k_heads` × `expand_ratio` value groups: decay `s*=exp(decay)`, kv-mem contraction, gated delta `delta=(v−kv_mem)*beta`, output `q·s`. Avoids materializing the 5D `[b,kh,expand,dk,dv]` intermediate the JAX path needs. `dimension_semantics=("parallel",)`. | [gated_delta_rule.py#L75](../../../raw/code/EasyDeL/easydel/operations/kernels/gated_delta_rule.py#L75) (call [#L579](../../../raw/code/EasyDeL/easydel/operations/kernels/gated_delta_rule.py#L579)) / [catalog](catalog/easydel/operations/kernels/gated_delta_rule.md#_gdr_grouped_decode_kernel) | experimental (decode-only, "float32 benchmarking inputs") | novel-to-easydel |
| `_fused_conv_decode_kernel` | mosaic_tpu | Fuses three depthwise-conv decode ops into one kernel: state shift (drop oldest token, append new), depthwise conv `sum(new_state*kernel)`, and SiLU. Tiles over `conv_dim` (not `num_slots`) in `CONV_TILE` chunks to dodge the TPU 8-lane block-alignment constraint when `num_slots % 8 != 0`. All math fp32. `dimension_semantics=("parallel",)`. | [gated_delta_rule.py#L147](../../../raw/code/EasyDeL/easydel/operations/kernels/gated_delta_rule.py#L147) (call [#L744](../../../raw/code/EasyDeL/easydel/operations/kernels/gated_delta_rule.py#L744)) / [catalog](catalog/easydel/operations/kernels/gated_delta_rule.md#_fused_conv_decode_kernel) | experimental | novel-to-easydel |
| `_pallas_gdn_decode_kernel` | mosaic_tpu | Fused Gated-DeltaNet (GDN, Qwen3-Next) ragged decode step; grid `(T//B_TOK,)` with `B_TOK=16` tokens unrolled per program, one recurrent state slot per token (identity `state_indices = arange`). Per token: fp32 recurrence `k·state`/`q·state`, `v_new=beta*(v−exp_g*k_state)`, `out=exp_g*q_state+q_k*v_new`, `new_state=state*exp_g+k⊗v_new`; `valid` mask leaves invalid slots' state unchanged. Per-head matmul loop because TPU Pallas matmul supports only 1 batch dim. `beta`/`exp_g`/`valid` padded to 128 lanes. | [inference_gdn.py#L638](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L638) (call [#L710](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L710)) / [catalog](catalog/easydel/operations/kernels/inference_gdn.md#_pallas_gdn_decode_kernel) | experimental | novel-to-easydel |

### Triangular-matrix inverse (GDR chunked-prefill helpers)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `newton_schulz_inverse_pallas_kernel` | mosaic_tpu | Newton–Schulz iterative inverse of unit lower-triangular matrices; grid `(cdiv(N,block_size),)`, block_size 64, each program inverts a block of stacked `(N,N)` matrices via the doubling `S ← S(2I − AS)` recurrence at `Precision.HIGHEST`. Used to invert the `(I − tril(β·kk))` factor in chunked GDR. | [inference_gdn.py#L77](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L77) (call [#L88](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L88)) / [catalog](catalog/easydel/operations/kernels/inference_gdn.md#newton_schulz_inverse_pallas_kernel) | experimental | novel-to-easydel |
| `decompose_triangular_matrix_inverse_pallas_kernel` | mosaic_tpu | Alternative block-wise triangular inverse: solves `A X = I` by block Gaussian elimination — inner blocks of `block_size=16` solved by explicit `local_forward_substitution`, cross-block coupling subtracted via `matmul` at `Precision.HIGHEST`. Grid `(cdiv(N,n_block_size),)`, `vmem_limit_bytes=64 MiB`. Numerically-exact counterpart to the Newton–Schulz kernel (`TriangleSolverImpl` selects between them). | [inference_gdn.py#L128](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L128) (call [#L176](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L176)) / [catalog](catalog/easydel/operations/kernels/inference_gdn.md#decompose_triangular_matrix_inverse_pallas_kernel) | experimental | novel-to-easydel |

## Imported (not defined here)

The remaining 16 files in `easydel/operations/kernels/` are registry adapters that `from ejkernel.modules import ...` and re-export — **no local `pl.pallas_call`**. They are catalogued cross-repo, not here. Notable ones:

- `flash_attention.py`, `blocksparse_attention.py` (Splash), `scaled_dot_product_attention.py`, `vanilla_attention.py`, `ring_attention.py` (ejkernel's Splash-wrapped ring — not a from-scratch ring kernel), `unified_attention.py` (vLLM-style prefill+decode), `paged_flash_attention.py`, `decode_attention.py` (ragged decode) — all import from `ejkernel.modules`.
- `ragged_page_attention.py`, `multi_latent_ragged_page_attention.py` — ejkernel RPA v2/v3 + MLA paged, incl. turboquant variants.
- `ssm1.py`, `ssm2.py`, `kda.py` (Kimi Linear delta attn) — ejkernel `state_space_v1/v2` and KDA ops.
- `gated_delta_rule.py` and `inference_gdn.py` **also import** ejkernel's `gated_delta_rule`/`ragged_gated_delta_rule` (training + the `_decode_path`/`_ragged_gdr_chunked_prefill` fallbacks) alongside their local decode kernels above — i.e. these two files are hybrids, not pure adapters.
- `glm_moe_dsa_indexer.py`, `inference_conv1d.py` — no `pl.pallas_call` found on read (op-registry / JAX-level logic); the fused conv Pallas kernel for the conv1d decode path lives in `gated_delta_rule.py`'s `_fused_conv_decode_kernel`.

## Notes

- **Divergence from the 2026-04-23 directory.** §6.2 of the [community directory](../../analyses/pallas-kernel-directory/06-community-research.md) lists EasyDeL's kernel files as "thin adapters re-exporting ejkernel implementations." That is true for 16 of 18 files but **misses the 6 locally-authored decode kernels** in `gated_delta_rule.py` and `inference_gdn.py`. These are the reuse-worthy, novel-to-EasyDeL surface: linear-attention (GDR/GDN) ragged single-token decode, where EasyDeL keeps the recurrent state update in VMEM instead of materializing large intermediates. Source has diverged (grown) since the catalog snapshot.
- **Authorship.** `gated_delta_rule.py` header: `Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi)`. `inference_gdn.py` carries mixed `Copyright 2026 Google LLC` / `Copyright EasyDeL` headers — the triangular-inverse math is derivative of standard Newton–Schulz / block-inversion, wrapped in EasyDeL's Pallas glue.
- **Backend / hardware.** All six use `jax.experimental.pallas` + `jax.experimental.pallas.tpu as pltpu` with `pltpu.CompilerParams` (no Triton/`plgpu`) ⇒ TPU Mosaic only. No explicit v4/v5e/v6e gating in-source; VMEM limits set to 64 MiB where large states are tiled.
- **Reuse gotchas.** (a) `grouped_gdr_decode_pallas` requires `decay` non-None and is documented "intended for float32 benchmarking inputs" — not the production dtype path. (b) `_pallas_gdn_decode_kernel` assumes identity `state_indices` (token i → slot i) and `T % 16 == 0`; the general ragged path with arbitrary slot mapping is not covered by the kernel. (c) `_fused_conv_decode_kernel` tiles `conv_dim`, requires `conv_dim % CONV_TILE == 0`. (d) All decode kernels are fwd-only; there is no Pallas backward here — training GDR/GDN gradients route through `ejkernel`.
- **No local attention (flash/splash/paged) or matmul Pallas kernels.** Everything attention- and matmul-shaped is imported from ejkernel; EasyDeL's local Pallas footprint is entirely linear-attention decode + its triangular-solve helpers.

## See also
- [easydel — overview](overview.md) — the repo's grounded wiki front door.
- [gated_delta_rule.py — catalog](catalog/easydel/operations/kernels/gated_delta_rule.md) — per-symbol grounded page (`_gdr_grouped_decode_kernel`, `_fused_conv_decode_kernel`, `GatedDeltaRuleOp`).
- [inference_gdn.py — catalog](catalog/easydel/operations/kernels/inference_gdn.md) — per-symbol grounded page (`_pallas_gdn_decode_kernel`, the two triangular-inverse kernels, `RaggedGatedDeltaRule`).
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 06 §6.2](../../analyses/pallas-kernel-directory/06-community-research.md).

## Sources
- [raw/code/EasyDeL/easydel/operations/kernels/gated_delta_rule.py](../../../raw/code/EasyDeL/easydel/operations/kernels/gated_delta_rule.py)
- [raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py](../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py)
