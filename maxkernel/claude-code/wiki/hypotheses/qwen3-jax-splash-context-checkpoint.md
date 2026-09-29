---
title: "Qwen3 jax — splash residual checkpointing ('context'): backward never re-runs the splash forward"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "3–7% step (targets the 250 ms remat-recompute bucket + part of the 28.1% splash bucket)"
confidence: medium-high
effort: S-M
origin: 2026-06-12-maxtext-feature-gap-inventory
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — splash context checkpointing

*Hypothesis*: passing `residual_checkpoint_name="context"` to splash (the
kernel tags its out+logsumexp as a named checkpoint) and keeping those
residuals via `save_and_offload_only_these_names(saved=[q,k,v],
offloaded=[context])` removes the backward's splash-forward re-run —
measurable as bs2/seq8192 step < 2,563 ms (**> 6,390 tok/s/chip** =
frontier +2%).

*Rationale*: #1 of the [MaxText feature-gap inventory](../analyses/2026-06-12-maxtext-feature-gap-inventory.md)
— MaxText's `minimal_with_context` policy + `attention_op.py:1307`. Our
v043 profile shows 250 ms (9.5%) remat recompute that is largely the splash
fwd re-run. Kwarg verified present in our jax splash
(`splash_attention_kernel.py:2528`) and tokamax splash. Context mass
≈4.6 GiB stacked at bs2 → offloaded to pinned host (DMA hides easily;
v026 hid 18 GB). Numerics: policy-only change; CPU smoke grads bit-identical.

*Falsification criterion*: ≤ 6,390 tok/s/chip (or OOM/numeric drift) →
refuted; the recompute bucket is then attributed to the FFN/norm
rematerialization, not splash.

**Supported** by [v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)
(2026-06-12): **+7.3% step (2,615 → 2,437 ms), +6.7% tok/s/chip (6,265 →
6,723/chip, ≈38.6% MFU)**; loss bit-identical (±0.0001); no OOM, no XLA
S(5) error. HLO confirms: splash custom-call dropped 735→529 ms (−206 ms,
−6.4 pp); S(5) context tensors `bf16[36,2,32,8192,128]` + `f32[36,2,32,8192]`
= 4.57 GiB in host-pinned memory; zero `splash_mha_fwd_no_residuals`
occurrences; async-done grew +25 ms (DMA overhead, not fully hidden).
**MaxText gap: −9.9% → −3.3%** (85 ms remaining). **Certified 50-step
stable** (v045-ctxval: 6,715 tok/s/chip median, 0.31% spread). Merging
to trunk.

## See also

- [Inventory (origin)](../analyses/2026-06-12-maxtext-feature-gap-inventory.md)
- [v044 (the test)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)
- Next: [qwen3-jax-fused-gate-up](qwen3-jax-fused-gate-up.md), [qwen3-jax-ctx-dma-pipeline](qwen3-jax-ctx-dma-pipeline.md), [qwen3-jax-ctx-hbm-only](qwen3-jax-ctx-hbm-only.md)
