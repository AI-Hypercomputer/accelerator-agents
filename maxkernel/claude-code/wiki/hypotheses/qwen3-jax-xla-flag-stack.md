---
title: "Qwen3 jax — MaxText HOST_OFFLOAD XLA flag stack"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "5-10% MFU"
confidence: high
effort: S
origin: 2026-06-02-qwen3_cc-jax-retrospective
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — MaxText HOST_OFFLOAD XLA flag stack

The single biggest unexplored lever on this lane (retrospective rec #1). The MaxText
`HOST_OFFLOAD` XLA flag bundle — latency-hiding scheduler, host-transfer overlap,
all-gather backward pipelining, scheduler memory-pressure tracking — drove the
llama3-jax sibling lane from ~34% → 43% MFU on identical v6e-8 hardware. qwen3-jax has
done remat+splash+CE-correctness but has touched **only one XLA flag** (`scoped_vmem`);
this bundle is entirely untried.

## Statement

Stacking the MaxText HOST_OFFLOAD XLA flag bundle on the v008 frontier config
(remat+splash+bs4+vmem, seq2048) raises MFU above 32.4% (target +5–10 pp, ≥ +1 pp to count).

## Rationale

- **Sibling-model evidence**: llama3-8b-jax (same architecture class, same hardware) records
  the HOST_OFFLOAD stack as its largest single XLA win (+~10%), via better compute/transfer
  overlap and scheduler memory-pressure awareness. See the
  [retrospective](../analyses/2026-06-02-qwen3_cc-jax-retrospective.md).
- **Flag set** (from the llama3-jax lane, all in `LIBTPU_INIT_ARGS` / `XLA_FLAGS`):
  `xla_tpu_enable_all_experimental_scheduler_features=true`,
  `xla_tpu_enable_scheduler_memory_pressure_tracking=true`,
  `xla_tpu_enable_ag_backward_pipelining=true`,
  `xla_tpu_host_transfer_overlap_limit=24`,
  `xla_tpu_scheduler_percent_shared_memory_limit=100`,
  `xla_latency_hiding_scheduler_rerun=2`,
  `xla_max_concurrent_host_send_recv=100`,
  `xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`.
- **Cheapest possible**: flag-only, reuses the `qwen3-8b-jax:v006-splash` image, no rebuild.

## Proposed experiment

v018: v008 frontier launch + the 8 flags above appended to `LIBTPU_INIT_ARGS` (keeping
`--xla_tpu_scoped_vmem_limit_kib=98304`). Measure MFU/tok-s-chip vs 32.4%; check loss parity
and exit 0. If it wins, follow with a flag-ablation to attribute the gain.

## Risks

- Some flags may be rejected by this libtpu build (would log "unknown flag" — benign, drop them).
- Scheduler changes could regress at this small shape; falsified if MFU ≤ 32.4%.
- No semantic risk (scheduling/transfer only).

## Dependencies

None — flag-only on the existing frontier image.
