---
title: "Qwen3 jax — context residuals in HBM (no host offload)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "0.5–1% step (eliminates the 25 ms DMA overhead from host offload; may OOM if HBM headroom insufficient)"
confidence: low
effort: S
origin: 2026-06-12-v044-context-checkpoint
hlo_prefilter: n/a
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — context residuals in HBM (no host offload)

*Hypothesis*: keeping the splash attention context residuals (out+lse) in
HBM instead of offloading them to pinned host memory (`save_qkv_ctx_hbm`
policy variant) eliminates the 25 ms DMA round-trip overhead observed in
v044, at the cost of 4.57 GiB of HBM headroom. The question is whether
the headroom exists.

*Rationale*: v044 shows HBM peak at 22.03 GiB (70.5% of 31.25 GiB cap),
leaving ~9.2 GiB of measured headroom. The context tensors are 4.57 GiB;
if placed in HBM rather than host-pinned S(5), peak HBM would be ~26.6 GiB
(85.1% cap) — which is tighter than v043's save_attn config (~27 GiB) but
should not OOM. If HBM fits, the async-done bucket (49 ms, 2.0%) shrinks
back toward the 24 ms baseline, recovering ~25 ms (≈1% step gain).

*Risk*: compile-time HBM is typically higher than run-time peak; if the
shard-map liveness analysis forces the full stacked `[36, ...]` residual
array to be live simultaneously during compilation, a compile-OOM is
possible. The scan over layers bounds liveness per iteration in theory, but
the `dynamic-update-slice` accumulator may still force the full buffer at
compile time.

*Falsification criterion*: compile OOM → refuted (HBM-only variant not
viable at this op-point); OR step time ≥ 2,435 ms (HBM variant slower due
to HBM pressure → refuted).

## See also

- [v044 (origin)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)
- [qwen3-jax-ctx-dma-pipeline](qwen3-jax-ctx-dma-pipeline.md) — alternative: hide the DMA overhead rather than eliminating it.
