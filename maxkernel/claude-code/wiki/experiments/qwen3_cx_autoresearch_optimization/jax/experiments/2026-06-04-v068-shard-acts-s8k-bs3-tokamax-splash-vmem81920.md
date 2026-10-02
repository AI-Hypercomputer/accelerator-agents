---
title: "Qwen3 JAX v068 - shard acts s8k bs3 tokamax splash vmem81920"
type: experiment
hypothesis: "Lowering the v056 VMEM limit from 98304 KiB to 81920 KiB may improve the frontier schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, vmem, remat, maxtext-ce, activation-sharding, long-seq, supported, frontier, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v068 - shard acts s8k bs3 tokamax splash vmem81920

## Hypothesis under test

**Hypothesis**: The valid v056 frontier runs at the v6e VMEM cap
(`--xla_tpu_scoped_vmem_limit_kib=98304`). Lowering the limit to the blueprint's
portable JAX-stack starting point, 81920 KiB, may give XLA a better schedule for
the same Tokamax Splash program.

**Mechanism**: Keep v056 fixed (`USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_MAX_LOGIT_CONST=30.0`, scan/remat/MaxText CE/activation sharding,
scheduler rerun3, and AG backward pipelining enabled) and change only
`--xla_tpu_scoped_vmem_limit_kib=98304` to `81920`.

**Predicted signal**: Loss stays near the v056 trajectory, ending around 12.05,
and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or the profile shows a
lower loop-fusion/custom-call share without increasing HBM pressure.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU
with unchanged profile shape.

## Setup

Planned GKE workload `alekseyv-qwen3-v068-vmem81920`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v068-shard-acts-s8k-bs3-tokamax-splash-vmem81920`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_tpu_scoped_vmem_limit_kib=81920` instead of
  v056's `98304`, keeping the rest of the v056 scheduler bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run and new valid throughput frontier:

- Worker 0: 50,790 tok/s total, 6,349 tok/s/chip, 36.4% MFU.
- Worker 1: 50,763 tok/s total, 6,345 tok/s/chip, 36.4% MFU.
- Step time stabilized around 3.87 s after warmup.
- Loss trajectory stayed normal, ending at 12.0508 on both workers.

Compared with v056's 50,383 tok/s / 36.1% MFU, lowering the VMEM limit to
81920 improves raw throughput by about 0.75-0.81% and raises displayed MFU to
36.4%.

## Profile

- **XProf run**:
  `2026-06-04-qwen3-jax-v068-shard-acts-s8k-bs3-tokamax-splash-vmem81920/2026_06_04_02_14_42`
- **MXU utilization**: 56.5%.
- **Peak HBM**: 28.32 GiB, 2.928 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 49.1%.
- Custom-call: 23.7%.
- Loop fusion: 18.0%.
- Data formatting: 4.1%.
- All-gather: 0.9%.

The custom-call split remains similar to v056 but not identical: DKV is 12.5%,
and the two forward residual custom-calls are 5.6% each. The win is therefore a
small schedule-efficiency improvement rather than a new kernel shape or a
semantic/model-code change.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v068-shard-acts-s8k-bs3-tokamax-splash-vmem81920/hlo/`
- **Size**: 30.47 MiB
- **Object count**: 1619
- **Dominant module**:
  `module_0267.jit_train_step.cl_854318611`

The HLO flagfile confirms the hypothesis fired:
`--xla_tpu_scoped_vmem_limit_kib=81920`, with scheduler rerun3 and AG backward
pipelining still enabled. The final train-step after-optimization HLO is
1,375,090 bytes, versus 1,364,488 bytes for v056/v067, so this was not a silent
no-op. The memory report shows 28.27 GiB used and a 22.50 GiB preallocated temp
arena, with the same major tensor families as the v056 stack.

## Verdict

**Supported.** `--xla_tpu_scoped_vmem_limit_kib=81920` is a clean valid
frontier move at the v056 op-point. Keep 81920 as the current VMEM value and
follow with the lower end of the v6e sweep, 65536 KiB, before returning to
structural attention work.
