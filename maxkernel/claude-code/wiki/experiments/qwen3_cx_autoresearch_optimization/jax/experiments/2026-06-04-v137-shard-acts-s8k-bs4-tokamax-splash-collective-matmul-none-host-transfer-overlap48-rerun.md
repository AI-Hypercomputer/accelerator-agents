---
title: "Qwen3 JAX v137 - shard acts s8k bs4 tokamax splash collective matmul none host-transfer overlap48 rerun"
type: experiment
hypothesis: "v136 showed a noise-level average edge for host-transfer overlap48; rerun to test whether the effect is repeatable."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, host-transfer-overlap, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, rerun, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v137 - shard acts s8k bs4 tokamax splash collective matmul none host-transfer overlap48 rerun

## Hypothesis under test

**Hypothesis**: v136 kept the exact v123/v127 train-step HLO and produced a
best-worker average of 56,456 tok/s, only 3 tok/s above v127 and below v123's
best-observed steady step. Rerun the same configuration to distinguish a real
host-transfer scalar preference from run-to-run noise.

**Mechanism**: Exact rerun of v136:

```text
--xla_tpu_host_transfer_overlap_limit=48
```

with the v127 simplified stack and batch-4/global-32 shape unchanged.

**Predicted signal**: If overlap48 is a real scalar improvement, the rerun should
again match or exceed v127's 56,453 tok/s best-worker average and approach or
exceed v123's 56,599 best-observed step. If not, it should fall back into the
56.43k-56.45k normal frontier band.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO,
or clean completion that does not repeat v136's small average edge.

## Setup

GKE workload `alekseyv-qwen3-v137-bs4-hto48r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v137-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v136 stack rerun.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0461.

- Worker0: 56,456 tok/s avg, 7,057 tok/s/chip, 40.5% MFU; best observed steady step 56,565 tok/s.
- Worker1: 56,383 tok/s avg, 7,048 tok/s/chip, 40.4% MFU; best observed steady step 56,566 tok/s.

The rerun repeats v136's worker0 average but does not reproduce a robust
two-worker improvement: worker1 drops below the v127/v135/v136 frontier band,
and both best observed steps remain below v123's 56,599 tok/s.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v137-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48-rerun/2026_06_04_14_40_22`

- Step average: 2009.6 ms, stddev 2501.6 ms.
- Total time: 112,110.6 ms; compute time 111,504.6 ms; idle 606.0 ms / 0.5%.
- MXU utilization: 63.7%.
- Peak HBM: 26.91 GiB (21.118 GiB stack, 5.789 GiB heap, 4.339 GiB free).
- Train-step program: `jit_train_step(16296087512277873197)`.
- Top buckets: convolution fusion 60,201.0 ms / 53.7%; custom-call 29,348.8 ms / 26.2%; loop fusion 11,669.9 ms / 10.4%; all-reduce-scatter fusion 3,762.3 ms / 3.4%.
- Splash attention custom calls: DKV 15,477.7 ms / 13.8%; forward residuals 13,871.1 ms / 12.4% combined.

Compared with v136, the profile is the same program class but slightly slower
overall, with lower MXU and higher idle time. That supports a noise/tie verdict
rather than a new scalar setting.

## HLO Dump

No fresh HLO objects were emitted under the v137 run directory, consistent with
a persistent compilation-cache hit. The XProf train-step program hash remains
`16296087512277873197`, matching v123/v126/v127/v135/v136.

Use v136's compiled evidence for the same workload shape and flags: optimized
HLO size 787,978 bytes, content hash
`3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, peak
compiler memory 28,862,302,652 bytes / 26.88 GiB.

## Verdict

Refuted/tie. Host-transfer overlap48 is not a robust improvement. The rerun
kept the same train-step program and repeated only a noise-level worker0 average
edge while worker1 regressed and best observed steps stayed below v123.

Do not carry overlap48; restore `--xla_tpu_host_transfer_overlap_limit=24`.
The host-transfer overlap scalar sweep is closed for the current batch-4
collective-matmul-none frontier. Next probe the adjacent host scalar
`--xla_max_concurrent_host_send_recv=128` from the v127 simplified baseline.
