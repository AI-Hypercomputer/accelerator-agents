---
title: "MaxText vs qwen3-jax trainer — architecture & optimization inventory: what is missing (and what isn't)"
type: analysis
tags: [qwen3-cc5, jax, maxtext, feature-gap, architecture, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# MaxText feature-gap inventory — what's missing in our trainer

User-directed companion to [retrospective #3](2026-06-12-qwen3_cc5-jax-retrospective-3.md).
Code-level inventory of MaxText (`raw/code/maxtext`, sibling-wiki checkout,
read-only) vs our trainer at the v043 frontier (6,265 tok/s/chip / ≈36.0%
MFU; MaxText reference 6,953 = −9.9%). Grounded against the measured
profiles: our conv bucket is **HBM-BW-bound (49% BW-util)** where theirs is
**compute-bound (26.9%)** on identical math — the gap is layout/fusion, not
kernel quality.

## Missing in ours — ranked at our op-point

| # | Feature | MaxText source | Est. gain | Conf | Effort |
|---|---|---|---|---|---|
| 1 | **Splash residual checkpointing** (`residual_checkpoint_name="context"`): the kernel saves out+lse so the backward never re-runs splash fwd. Targets our 9.5% remat-recompute + part of the 28.1% splash bucket. Both our splash versions support the kwarg (verified). HBM: +~4.6 GiB stacked at bs2 vs 2.63 free → pair with `save_and_offload(saved=[q,k,v], offloaded=[context])` (DMA hides easily; 4.6 GB ≪ the 18 GB that hid at v026) | `attention_op.py:1193,1307`; policy `minimal_with_context` | **3–7%** | med-high | S–M |
| 2 | **Fused gate+up MLP** (one `(D, 2·12288)` matmul; halves FFN-input activation reads, merges two MXU calls). Numerics-identical. **HLO pre-filter required** (SCHEMA: XLA may already co-fuse the shared-input reads) | `fused_mlp` (default False), `linears.py:410-424` | 2–5% | medium | S |
| 3 | **Weight layout (in, out)** — DenseGeneral-style kernels contracted `x@W` with no transpose; we store HF `(out, in)` ≡ `x@Wᵀ`. Prime suspect for the 26.9%-vs-49% conv BW-util split | `linears.py:156-163` | 0–8% (high variance) | low-med | M |
| 4 | **Explicit activation sharding constraints** at every sub-block + `out_sharding=` on dot_generals (constrains XLA layout choices; complements #3) | `models/qwen3.py:1152-1236`, `linears.py:236-244` | 0–3% | low-med | S |
| 5 | **`param_scan_axis=1`** (stack scanned weights on axis 1, not 0 — deliberate layout choice) | `base.yml:343` | 0–3% | low | S |
| 6 | v6e mesh device reorder (`optimize_mesh_for_tpu_v6e`) | `max_utils.py:554-576` | ≤0.5% (our AG wait is 0.9 ms) | low | S |
| 7 | **int8 AQT training** (fwd + both grad matmuls int8; injected into all DenseGenerals except logits) | `quantizations.py:380-398` | potentially large BUT **changes numerics ⇒ `invalid` per SCHEMA rule 8** — concept-only unless the human waives | — | L |

Notable non-gap: **`fused_qkv` is MHA-only** (weight `(D,3,heads,hd)`) —
incompatible with Qwen3's 32q/8kv GQA, and their qwen3-8b run doesn't use
it. Not their advantage.

## What WE have that MaxText lacks

- **Tokamax fused linear+CE** (chunked_xla, f32-x) — zero `[BL,V]` logits in
  HBM; their default materializes fp32 logits + T5X CE; their
  `num_vocab_tiling` alternative all-gathers ALL params over FSDP first.
- **Autotuned splash** (bq2048/bkv1024, fused-bwd ON) — measured **89 ms/step
  faster** than their flash-2048 at the identical architecture (their
  defaults: 512 blocks, fused-bwd off).
- `JAX_SCAN_UNROLL` + `JAX_REMAT_SKIP_EVERY` probe knobs; lean NNX direct
  trainer (no Linen bridge / metrics machinery in the step).
- Leaner step: no global-norm grad clip (they pay one every step).

## DO-NOT-PORT (closed on our side with measurements)

Optimizer/param host offload (we beat offload at bs2; HBM fits); gradient
accumulation & PP/CP (off-scale at 8 chips); batch > bs2 (3 substrates);
TP (−23.6%); SC-offload flags (compile-fatal); ACF flags (worth 0);
AG-pipeliner flags (fire-and-hurt); their flash kernel & block defaults
(slower than our splash).

## Verdict on the −9.9%

Parity or better on: flags, attention kernel, CE, remat save-list, memory
mechanism. The credible remaining MaxText advantages are exactly two:
**(1) attention-residual checkpointing ("context")** — directly removes our
measured 250 ms recompute; **(2) the matmul-layout complex** ((in,out)
kernels + fused gate-up + explicit activation shardings + axis-1 scan
stacking) — the best explanation for compute-bound-vs-BW-bound conv at
identical math. Both are now filed as hypotheses; #1 dispatches first
(S–M, kwarg verified available).

## Sources

- MaxText checkout (read-only): `/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki/raw/code/maxtext` (file refs in tables)
- Our trainer: `wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/` (trunk `23aa5e7`)
- Measured grounding: [v038/v043 profiles](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v043-bs2-svqkv-val.md), [bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md), [mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md)

## See also

- [Retrospective #3](2026-06-12-qwen3_cc5-jax-retrospective-3.md) · [Phase summary](2026-06-12-maxtext-gap-closing-phase-summary.md) · [Model page](../models/qwen3-cc5-jax.md)
