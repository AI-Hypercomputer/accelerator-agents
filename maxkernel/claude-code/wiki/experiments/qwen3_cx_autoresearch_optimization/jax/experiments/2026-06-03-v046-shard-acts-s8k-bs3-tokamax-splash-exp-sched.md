---
title: "Qwen3 JAX v046 - shard acts s8k bs3 tokamax splash exp sched"
type: experiment
hypothesis: "Enabling tokamax Splash's experimental scheduler on top of v045 may reduce DKV/backward custom-call time."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, scheduler, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v046 - shard acts s8k bs3 tokamax splash exp sched

## Hypothesis under test

**Hypothesis**: Enabling tokamax Splash's experimental scheduler on top of v045
may reduce DKV/backward custom-call time.

**Mechanism**: Keep v045 fixed (`USE_TOKAMAX_SPLASH=1`, scan, remat, MaxText
CE, activation sharding, seq8192, global batch 24) and set only
`TOKAMAX_USE_EXP_SCHED=1`.

**Predicted signal**: Clean run above v045's 49,069 tok/s / 35.2% MFU, ideally
with lower DKV or total custom-call self-time.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v045's 49,069 tok/s / 35.2% MFU.

## Setup

GKE workload `alekseyv-qwen3-v046`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v046-shard-acts-s8k-bs3-tokamax-splash-exp-sched`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_USE_EXP_SCHED=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but it regressed below v045 and landed near the pre-tokamax-Splash
frontier:

- Worker 0: 48,202 tok/s total, 6,025 tok/s/chip, 34.6% MFU.
- Worker 1: 48,177 tok/s total, 6,022 tok/s/chip, 34.5% MFU.
- Step time stabilized around 4.08 s after warmup.
- Loss trajectory matched v045/v041, ending at 12.0511.

The `tokamax-splash` config confirmed `exp_sched=True`, so the knob was active.
Compared with v045's 49,069 tok/s / 35.2% MFU, the experimental scheduler costs
about 867 tok/s and 0.6-0.7 MFU points.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v046-shard-acts-s8k-bs3-tokamax-splash-exp-sched`
- Detailed XProf analysis skipped because wall-clock logs already refute the
  hypothesis against v045. Use the trace if a later comparison needs the exact
  custom-call split.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v046-shard-acts-s8k-bs3-tokamax-splash-exp-sched/hlo/`
- **Size**: 18.16 MiB
- **Object count**: 32

## Verdict

**Refuted.** Keep tokamax Splash's experimental scheduler disabled. The default
tokamax Splash path from v045 remains the frontier.
