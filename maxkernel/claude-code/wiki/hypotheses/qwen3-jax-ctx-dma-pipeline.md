---
title: "Qwen3 jax — context DMA scheduling improvement"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: retired
expected_gain: "0.5–1.5% step time (targets the 49 ms async-done bucket: ~25 ms residual DMA latency)"
confidence: low-medium
effort: S
origin: 2026-06-12-v044-context-checkpoint
hlo_prefilter: n/a
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — context DMA scheduling improvement

*Hypothesis*: the v044 context-checkpoint achieves its 178 ms step gain but
leaves 25 ms of DMA latency visible in the `async-done` bucket (49 ms total
vs 24 ms baseline). Improving the XLA scheduler's placement of the context
DMA `async-start` earlier in the scan body — so it fires before the preceding
layer's MXU work starts rather than after — can hide this residual latency
behind compute.

*Rationale*: v044's HLO uses `dynamic-update-slice-start/done` async pairs
to write context residuals to pinned host S(5) memory. The `async-done`
bucket grew from 24 ms (v043 baseline, from unrelated DMA) to 49 ms (+25 ms).
The 4.57 GiB context offload at v6e's ~1.5 TB/s HBM bandwidth would take
~3 ms in the ideal case; the measured 25 ms excess indicates the DMA
`async-start` is being scheduled inside the compute window rather than being
pre-issued. Possible approaches: (a) `xla_tpu_host_transfer_overlap_limit`
tuning (currently 24; reducing it may help); (b) manual barrier annotation
in the trainer code to pre-issue the async DUS before the layer's matmuls;
(c) inspecting the HLO post-optimizations to confirm the `async-start` fires
before or after the surrounding `dot` ops.

*Falsification criterion*: step time ≥ 2,435 ms (no measurable improvement)
after the scheduling change → refuted. The async-done bucket must visibly
shrink in xprof.

## See also

- [v044 (origin)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)
