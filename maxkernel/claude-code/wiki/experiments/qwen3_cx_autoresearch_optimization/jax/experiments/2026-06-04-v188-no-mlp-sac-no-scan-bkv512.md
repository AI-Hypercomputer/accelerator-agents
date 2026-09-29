---
title: "Qwen3 JAX v188 - no MLP intermediate sac, no scan, Splash BKV512"
type: experiment
hypothesis: "The no-scan unrolled graph has much larger fusion/copy pressure than the scan graph, so smaller forward Splash KV tiles may reduce pressure enough to improve steady throughput despite regressing in the scan family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-bkv512
status: invalid
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, splash-tiling, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v188 - no MLP intermediate sac, no scan, Splash BKV512

## Hypothesis under test

**Hypothesis**: v156 showed `SPLASH_BKV=512` regressed in the scan family, but
the confirmed no-scan frontier is a much larger unrolled program with different
fusion/copy pressure. Smaller forward KV tiles may reduce attention custom-call
pressure enough to improve steady step time under no-scan.

**Mechanism**: Runtime-only Splash tiling change on top of the confirmed v183
frontier:
`SPLASH_BKV=512`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band. A fresh HLO/hash with lower wall time would
justify carrying the smaller forward KV tile only in the no-scan schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v188-noscan-bkv512`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v188-no-mlp-sac-no-scan-bkv512`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 SPLASH_BKV=512 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime attention-kernel tiling change
only and preserves model math.

## Results

Invalid launch. The run picked up `SPLASH_BKV=512`, but left
`bkv_compute=1024`. Tokamax Splash rejects that combination before train-step
compilation:

```text
ValueError: bkv=512 must be a multiple of bkv_compute=1024.
EXIT_CODE=1
```

No performance result was produced. The intended smaller forward-KV-tile test
requires setting both `SPLASH_BKV=512` and `SPLASH_BKV_COMPUTE=512`, as in the
older v156 scan-family probe.

## Profile

No profile was produced.

## HLO Dump

Only startup/setup artifacts were emitted; no train-step HLO was produced.

## Verdict

Invalid launch. Do not interpret as a performance result. Rerun as v189 with
the corrected environment pair `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`.
