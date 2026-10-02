---
title: "Qwen3 JAX v101 - shard acts s8k bs3 tokamax splash layeragg no async collective fusion"
type: experiment
hypothesis: "Disabling the base async collective fusion flag on the current v090 scheduler frontier may show whether that inherited bundle still contributes after the layer-scheduler plus sparse-core-aggregator pair."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, async-collective-fusion, latency-hiding, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v101 - shard acts s8k bs3 tokamax splash layeragg no async collective fusion

## Hypothesis under test

**Hypothesis**: The v090 frontier still carries the broad async collective
fusion enable from the inherited scheduler bundle. The later v077-v084 probes
tested async collective fusion sub-modes, but not whether the base async
collective fusion enable is still useful once selective resources, scoped VMEM,
the latency-hiding layer scheduler, and sparse-core collective aggregation are
all active.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_enable_async_collective_fusion=true` to
`--xla_tpu_enable_async_collective_fusion=false`. Leave the subordinate
all-gather/reduce-scatter/multiple-step flags in the command so the single
effective variable is the base enable.

**Predicted signal**: Loss stays finite and throughput either exceeds the
v090 best-observed 51,033 tok/s / 36.6% MFU or shows a useful profile change in
collective/data-formatting time. If throughput is at or below the v090 band,
keep async collective fusion enabled.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 51,033 tok/s best-observed / 51,026+ repeated
frontier without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v101-noacf`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v101-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-async-collective-fusion`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_enable_async_collective_fusion=false`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v101-noacf` completed cleanly on 2026-06-04 with
valid loss, but regressed below the v090 frontier.

- Worker 0: 50,361 tok/s, 6,295 tok/s/chip, 36.1% MFU
- Worker 1: 50,322 tok/s, 6,290 tok/s/chip, 36.1% MFU
- Final loss: 12.0511 on both workers
- Exit code: 0

Compared with v090's 51,033 tok/s best observed and 51,026+ repeated band,
disabling base async collective fusion loses about 1.3%.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v101-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-async-collective-fusion/2026_06_04_07_58_11`

Profile:

- Total time: 94,108.0 ms; idle 373.5 ms / 0.4%.
- `jit_train_step`: 93,733.8 ms / 99.6%.
- Convolution fusion: 45,394.4 ms / 48.2%.
- Custom-call: 22,080.3 ms / 23.5%.
  - `splash_mha_dkv_no_residuals.1`: 11,655.0 ms / 12.4%.
  - `splash_mha_fwd_residuals.2/3`: 10,425.3 ms / 11.1%.
- Loop fusion: 16,640.0 ms / 17.7%.
- Data formatting: 3,700.3 ms / 3.9%.
- All-gather: 2,123.8 ms / 2.3%.
- Peak HBM: 28.03 GiB of 31.25 GiB, with 3.22 GiB free.

The attention custom-call split is essentially unchanged from v090. The main
visible profile change is lower stack reservation / HBM and more explicit
all-gather time, which does not help wall clock.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v101-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-async-collective-fusion/hlo/`

- Object count: 1,839.
- Train-step optimized HLO:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,187,819 bytes, versus v089/v090's
  1,375,271 bytes.
- Pod command confirms `--xla_tpu_enable_async_collective_fusion=false` with
  the rest of the v090 scheduler frontier unchanged.

## Verdict

**Refuted.** The base async collective fusion enable is still useful on the
v090 layer-scheduler/sparse-core-aggregator frontier. Disabling it is valid and
reduces peak HBM to 28.03 GiB, but throughput regresses to 50,361 tok/s /
36.1% MFU and all-gather rises to 2.3% of device time. Keep
`--xla_tpu_enable_async_collective_fusion=true` in the frontier stack.
