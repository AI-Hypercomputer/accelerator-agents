---
title: "Qwen3-8B jax v024 — XLA scheduler flag ablation (umbrella flag alone)"
type: experiment
hypothesis: qwen3-jax-xla-flag-ablation
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-splash-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, xla-flags, ablation, remat, splash, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v024 — XLA scheduler flag ablation (umbrella flag alone)

## Hypothesis under test

**Hypothesis**: Of the 7 LIBTPU scheduler flags that gave v018 its +3.4 pp, the umbrella
`xla_tpu_enable_all_experimental_scheduler_features=true` may carry most of the benefit. Running it
ALONE (+ the required scoped_vmem) should recover MFU ≈ 35.8% (within ~1 pp). Attribution experiment —
de-risks the torchax-lane transfer and yields a reusable minimal-flag prior; not expected to beat v018.

**Mechanism**: v018 frontier shape (`--use_remat --use_splash --batch_size=4 --seqlen=2048`) with
`LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_all_experimental_scheduler_features=true'`
only (the other 6 scheduler flags dropped). Flag-only, splash image.

**Predicted signal**: MFU ≈ 35.8% (umbrella carries it) OR ≈ 32.4% (v008 — umbrella insufficient, other
flags matter). Loss parity; exit 0.

**Falsification criterion**: this is attribution, not a frontier attempt — "supported" is not the frame.
Outcome is the data point (umbrella sufficient vs not). Verdict will be `inconclusive` (informative,
non-frontier) unless it unexpectedly beats 35.8% (then supported) or crashes (invalid).

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v006-splash`. Flag-only.
- **Launch**: v018 frontier shape (`--use_remat --use_splash --batch_size=4 --seqlen=2048`) with `LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_all_experimental_scheduler_features=true'` ONLY — the other 6 scheduler flags dropped.

## Results

| Config | MFU | tok/s/chip | step (ms) |
|--------|-----|-----------|-----------|
| v008 (no scheduler flags) | 32.4% | 6,299 | ~1,301 |
| **v024 (umbrella flag only)** | **31.9%** | **6,203** | **~1,301** |
| v018 (full 7-flag bundle) | 35.8% | 6,964 | ~1,154 |

Loss 12.0954→12.0693 stable, no NaN, exit 0.

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-jax-v024-xla-flag-ablation
- **Run name**: `2026-06-02-qwen3-jax-v024-xla-flag-ablation`
- **On-disk pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v024-xla-flag-ablation/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v024-xla-flag-ablation/)
- **GCS**: `.../2026-06-02-qwen3-jax-v024-xla-flag-ablation/plugins/profile/2026_06_02_08_35_05/` (2 hosts; steps 12–14).
- Contents: xprof trace + HLO dump.

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v024-xla-flag-ablation/hlo/` — 1,060 files (text + proto).

## Verdict

**Refuted** (the ablation hypothesis "umbrella flag alone recovers most of +3.4 pp"). The umbrella
`xla_tpu_enable_all_experimental_scheduler_features=true` alone lands at **31.9% MFU / 6,203 tok/s/chip /
~1,301 ms — essentially the v008 no-scheduler-flags baseline (32.4%)**, NOT v018's 35.8%. **Attribution
result**: v018's +3.4 pp comes from the **6 explicit individual scheduler flags** (`scheduler_memory_pressure_tracking`,
`ag_backward_pipelining`, `host_transfer_overlap_limit=24`, `scheduler_percent_shared_memory_limit=100`,
`latency_hiding_scheduler_rerun=2`, `max_concurrent_host_send_recv=100`), not the umbrella switch — the
umbrella enables features with different/looser default tuning that nets to baseline here. **Implication
for the torchax transfer: apply the full explicit 7-flag bundle, not the umbrella shortcut.** Non-frontier
(attribution); frontier stays v018 (35.8%).

## Next hypotheses

- **None on the jax lane** — the seq2048 frontier (35.8%) is confirmed at its practical ceiling: every
  cheap/medium lever is exhausted, refuted, neutral, or intrinsic (6 post-v018 non-wins v019–v024). The
  load-bearing flag set is now known (the explicit 6, not the umbrella).
- **Pivot to the torchax sibling lane** (19.2% MFU, untouched by the optimization loop): transfer the
  proven jax stack — remat + splash + the **explicit 7-flag** XLA scheduler bundle (framework-agnostic
  libtpu/compiler levers). Plausible large win (19.2% → 30%+). Requires `/start-experiment qwen3_cc torchax`.
