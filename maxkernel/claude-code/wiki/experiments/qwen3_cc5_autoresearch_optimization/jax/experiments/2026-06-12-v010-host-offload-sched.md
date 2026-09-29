---
title: "Qwen3-8B jax v010 — HOST_OFFLOAD scheduler flag bundle"
type: experiment
hypothesis: qwen3-jax-host-offload-sched
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: 3ef2b1e (trunk; flag-only)
status: refuted
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, flags, latency-hiding, scheduler]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v010 — HOST_OFFLOAD scheduler flag bundle

Tests [qwen3-jax-host-offload-sched](../../../../hypotheses/qwen3-jax-host-offload-sched.md)
at the [v007 frontier](2026-06-12-v007-bs2-vmem.md) shape: add MaxText's
`HOST_OFFLOAD_FLAGS` scheduler bundle, nothing else.

## Hypothesis under test

**Hypothesis**: The 11-flag latency-hiding scheduler bundle lifts tok/s/chip
≥ 3% over 6,095 by overlapping FSDP collectives (30% of step at v007) with
compute.

**Mechanism**: `LIBTPU_INIT_ARGS` adds (verbatim from
`maxtext/benchmarks/xla_flags_library.py::HOST_OFFLOAD_FLAGS`):
`--xla_tpu_enable_all_experimental_scheduler_features=true
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_host_transfer_overlap_limit=24
--xla_tpu_aggressive_opt_barrier_removal=ENABLED
--xla_lhs_prioritize_async_depth_over_stall=ENABLED
--xla_tpu_enable_ag_backward_pipelining=true
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED
--xla_should_add_loop_invariant_op_in_chain=ENABLED
--xla_max_concurrent_host_send_recv=100
--xla_tpu_scheduler_percent_shared_memory_limit=100
--xla_latency_hiding_scheduler_rerun=2` on top of the scoped-vmem flag.
Known-good bundle (blueprint Phase 2); llama3-jax +10% standalone precedent.

**Predicted signal**: `collective-permute-done` / `all-gather` buckets shrink
(better overlap), step time < 667 ms; HLO program semantics unchanged
(scheduler-only flags). Watch HBM: memory-pressure-tracking may shift peak
slightly (28.13 GiB + ~3 GiB headroom at v007).

**Falsification criterion**: tok/s/chip ≤ 6,095 + 1% noise → refuted at this
op-point.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:latest` (trunk `3ef2b1e`).
- Workload: `alekseyv-qwen3-cc5-jax-v010-hosch`.
- Command: identical to v007 + the bundle in `LIBTPU_INIT_ARGS`.

## Results

20 steps clean (exit 0); flags compiled with no warnings; no OOM; loss
trajectory normal.

| Metric | v007 frontier | This run (+bundle) | Δ |
|--------|---------------|--------------------|---|
| Median steady step | 667 ms | 671.5 ms | +0.7% (noise) |
| Median tok/s | 48,758 | ~48,800 | +0.1% (noise) |
| Compile time | ~90 s | 126 s | +40% (LHS rerun ×2) |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v010-hosch`
  (run `…/2026_06_12_02_47_05`); profiled steps 12–14.
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v010-hosch/plugins/profile/2026_06_12_02_47_05/`
- Flat step time at identical shape is the complete signal — the scheduler
  found no additional overlap to exploit at this scale.

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v010-hosch/hlo/` — 1058 files.

## Verdict

**refuted** at this op-point — falsification criterion fired (≤ +1% noise
band). The llama3-jax +10% from this bundle was measured at seq8192/bs4-5
where per-step collectives are much larger and overlap headroom exists; at
seq2048/bs2 the latency-hiding scheduler finds nothing more to hide. Second
llama3-precedent flag lever (after SC offload, v009) that does NOT transfer
to this shape — flag bundles appear shape-gated; re-test both at seq-8192
once it lands. Only cost: +40% compile time.

## Next hypotheses

- [qwen3-jax-selective-remat](../../../../hypotheses/qwen3-jax-selective-remat.md) — the memory lever for bs3+ (dispatched as v012; phase 1 measures remat overhead at bs2, phase 2 probes bs3).
- [qwen3-jax-host-offload-sched](../../../../hypotheses/qwen3-jax-host-offload-sched.md) — retired at seq2048; re-test at seq-8192 together with SC offload (the llama3 op-point) when that shape lands.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v010-hosch/`
- Flag source: `raw/code/maxtext/benchmarks/xla_flags_library.py` (HOST_OFFLOAD_FLAGS)
- Prior: [v007 (frontier)](2026-06-12-v007-bs2-vmem.md), [v009 (SC offload refuted)](2026-06-12-v009-sc-offload.md)
