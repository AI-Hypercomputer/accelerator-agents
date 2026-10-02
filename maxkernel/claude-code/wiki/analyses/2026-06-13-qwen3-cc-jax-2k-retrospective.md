---
title: "Qwen3 JAX 2k v6e-8 Retrospective"
type: analysis
model: qwen3-cc-jax
lane: jax
scope: "since 2026-06-12, context length 2048 campaign"
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX 2k v6e-8 Retrospective

## Scope

Scoped retrospective for the 2k-context campaign only: experiments v081-v128
under `wiki/experiments/qwen3_cx_autoresearch_optimization/jax/experiments/`.
This intentionally excludes the full 8k history except as cross-lane/context
signal from the model page.

## Search-strategy assessment

- **Current 2k best**: v119, 71,053 tok/s global, 8,882 tok/s/chip, approx
  45.7% script MFU; XProf 3718.9 ms, 67.9% MXU. v122 measured 71,054 tok/s but
  was byte-identical and not a new frontier.
- **Frontier stuck**: no clean graph frontier since v114/v119. v120-v127 are
  OOMs, no-ops, or throughput regressions.
- **Single-axis heavy**: the 2k campaign is dominated by batch, VMEM, shmem,
  Splash block, SparseCore, ACF, and latency-multiplier probes. Structural or
  kernel-code work is effectively absent in this scoped campaign.
- **Closed topics**: batch scaling beyond per-chip 16 is closed for the current
  memory shape; RS latency 5 is closed; all-gather offload is closed; all-gather
  ACF and base ACF must stay enabled; ACF multiple-steps should stay enabled.
- **Risk**: continuing scalar compiler-flag sweeps has a high chance of
  producing no-op HLOs or memory-only wins that reduce throughput.

## Mechanism Tree

- **Batch-size + sequence amortization**
  - Supported: v081 -> v085 raised per-chip batch 4 -> 16 and moved throughput
    from 65,505 to about 70,679 tok/s.
  - Refuted/invalid: v089 batch18 OOM, v090/v091 batch17 completed but did not
    beat batch16, v094 batch17 under shmem100 failed. Carry per-chip batch16.
- **VMEM / scratch memory**
  - Supported: v086/v092/v093 found VMEM98304 + shmem100 as the stable high
    throughput base.
  - Refuted: v088/v099 VMEM97280 slower, v098 VMEM100352 slower, v123 VMEM98816
    changed to copy-heavy HLO and regressed. Keep VMEM98304.
- **Splash attention**
  - Supported: v095/v096 BKV/BKV_COMPUTE 2048 improved the batch16 frontier.
  - Refuted: v100 DQ steps failed, v124 BQ1024+BKV2048 regressed and became
    copy-heavy. Keep BQ/BKV/BKV_COMPUTE 2048.
- **FSDP / collective optimization**
  - Supported: v114/v119 RS latency multiplier 4 plus rerun4 is the current
    measurement best.
  - Refuted: all-reduce latency 1/3 did not beat, RS latency 5 OOMed even with
    AR latency 3, all-gather offload OOMed, disabling all-reduce offload slowed.
- **libtpu / XLA flags**
  - No-op/tie: async AG/RS/AR caps, all-experimental scheduler true, shmem99,
    SparseCore aggregator false.
  - Refuted: layer scheduler true caused extreme compile OOM; disabling
    all-gather ACF, base ACF, or ACF multiple-steps reduced throughput.
- **Pallas kernels / structural code work**
  - Unexplored in this 2k campaign. The existing MaxText CE path is still used;
    no new fused lm-head+CE, vocab-parallel CE, QKV layout pinning, or
    attention-kernel code change has been attempted here.

## Main Inference

The current 2k frontier is not limited by a missing scalar flag in the
neighborhood already explored. It is a tight v6e memory/scheduler balance where
some memory-saving schedules free HBM but lose enough collective/fusion quality
to reduce throughput. The profile remains compute-heavy enough that pure
overlap toggles are not moving the needle, while the HLO evidence shows many
flag changes collapse to byte-identical programs.

## Ranked Next Directions

1. **Structural CE/logits path**: implement or port a faster streaming or
   vocab-parallel CE/lm-head path. The 8k lane already showed existing Tokamax
   CE fits larger batch but is too slow; the opportunity is not to toggle it,
   but to make this path competitive.
2. **Layout/data-formatting drilldown**: use XProf and HLO neighborhoods around
   repeated copy/reshape/data-formatting ops in the v119 frontier to find a
   model-code layout pin or sharding constraint that removes copies without
   perturbing the good collective schedule.
3. **Attention-kernel block sweep only if code exposes missing backward knobs**:
   scalar BQ/BKV forward changes are mostly closed; useful work likely requires
   exposing backward Splash block controls rather than more one-off env flags.
4. **Retire more compiler-flag treadmill after v128** unless a profile/HLO
   signature identifies a specific bucket above 5% that the flag is expected to
   move.

## Experiment Ledger

| Range | Topic | Outcome |
|---|---|---|
| v081-v085 | Batch scaling | Supported through batch16 |
| v086-v104 | VMEM/shmem/Splash setup | v095/v096 BKV2048 + shmem100 carried |
| v105-v112 | Scheduler and SparseCore toggles | Mostly no-op or refuted |
| v113-v119 | RS/AR latency bracket | v114/v119 carried; RS5 OOM closed |
| v120-v124 | AG offload, aggregator, shmem99, VMEM midpoint, BQ1024 | Refuted or no-op |
| v125-v127 | ACF ablation | Refuted; base/all-gather/multiple-step ACF required |
| v128 | Scheduler memory-pressure tracking | Running at retrospective time |
