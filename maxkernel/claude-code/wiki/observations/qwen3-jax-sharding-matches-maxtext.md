---
title: "Qwen3-jax FSDP sharding already matches MaxText's logical_axis_rules — the MXU gap is not a sharding lever"
type: observation
tags: [qwen3-cc, jax, sharding, mxu, logical-axis-rules, maxtext, fsdp]
created: 2026-06-02
updated: 2026-06-02
---

The v035 profile attributed ~29% (+156 ms) of the residual seq8192 gap to MaxText to an "MXU occupancy"
difference (53.6% vs 61.2%), speculatively pinned on `logical_axis_rules` tile alignment. **A source-level
investigation refutes that as a distinct lever: our sharding already matches MaxText's.**

## What was observed

MaxText's `MaxText/configs/base.yml` `logical_axis_rules` shard the **`embed` (hidden D) axis on `fsdp`**
and put `mlp` / `heads` / `q_heads` / `kv_heads` / `vocab` on the **`tensor`** axis. The qwen3-8b reference
run (`benchmarks/maxtext_trillium_model_configs.py`: `qwen3_8b_8192_ref` / `_2048_ref`) sets
`ici_fsdp_parallelism: -1` (= all 8 chips on fsdp), `ici_tensor_parallelism` unset (= 1),
`ici_fsdp_transpose_parallelism: 1`, `per_device_batch_size` 3 (seq8192) / 4 (seq2048). So MaxText ran
**pure 1D FSDP=8** — `tensor`/`fsdp_transpose` collapse to 1 → only the `embed`/D axis is sharded.

Our [`model/sharding.py`](../experiments/qwen3_cc_autoresearch_optimization/jax/model/sharding.py)
`SHARDING_PLAN` under `tp_parallelism=1` resolves to exactly the same layout (the `tp` axis collapses to
`None`, leaving only `fsdp` on the D dimension of every weight):

| weight | logical | our spec (tp=1) | sharded dim | MaxText |
|--------|---------|-----------------|-------------|---------|
| q/k/v_proj | (heads·hd, D) | (None, fsdp) | D (embed) | embed→fsdp ✓ |
| gate/up_proj | (ffn, D) | (None, fsdp) | D (embed) | embed→fsdp ✓ |
| o_proj / down_proj | (D, …) | (fsdp, None) | D (embed) | embed→fsdp ✓ |
| embed / lm_head | (V, D) | (fsdp via tp-collapse) | … | vocab→tensor(=1), embed→fsdp ✓ |

Activations: ours `input_sharding = P(fsdp, None)` (batch-sharded) = MaxText `activation_batch→fsdp`.

**There is no config-level sharding/tiling change available — the layouts are identical.**

## Implications

- The "MXU/logical-axis tile alignment" lever from the [v035 profile](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
  and [kernel-gap hypothesis](../hypotheses/qwen3-jax-seq8192-kernel-gap.md) **does not exist as a distinct
  config lever** — closed without a TPU dispatch (the investigation-before-editing discipline avoided a
  guaranteed-no-op sharding rewrite).
- The MXU occupancy gap (53.6% vs 61.2%) is therefore **downstream of the remat recompute**: our
  `nothing_saveable` recomputes the RMSNorms (loop-fusion 17.4% vs MaxText 7.2%), adding non-MXU work that
  dilutes the MXU fraction of the step. It is the *same* root cause as the +85% remat bucket, not a separate
  one — so the residual to MaxText collapses to a single lever: **reduce the norm recompute**.
- Reducing the norm recompute needs either a **norm-inclusive selective-save remat** (constrained: HBM is at
  90.7% at bs3, so saving norm activations on-device likely OOMs → only feasible at bs2, giving back bs3
  throughput) or a **pipelined host offload** (kernel-authoring, out of scope; stock pinned_host offload
  refuted v030/v036). Both are constrained; the seq8192 frontier v035 (6,030 = 86.9% of MaxText) is at the
  practical config/refactor ceiling.

## Seen in

- [v035](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md) (profile that raised the question), [v036](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v036-maxtext-ce-offload-s8k-bs3.md) (offload refuted), [retrospective #4](../analyses/2026-06-02-qwen3_cc-jax-retrospective-4.md).
- Source: `maxtext-tpu-recipes-v0.1.4/MaxText/configs/base.yml` + `benchmarks/maxtext_trillium_model_configs.py`.
