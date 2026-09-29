---
title: "Qwen3 JAX v183 - no MLP intermediate sac without scan rerun"
type: experiment
hypothesis: "If v182's no-scan gain is real rather than noise, an exact rerun should repeat above the v169/v170 frontier and close no-scan as the new confirmed schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, maxtext-ce, long-seq, rerun, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v183 - no MLP intermediate sac without scan rerun

## Hypothesis under test

**Hypothesis**: v182 showed that disabling scan on the current v169 schedule
family fits and improves steady-state throughput to 58,307-58,330 tok/s /
41.8% MFU. An exact rerun should repeat above the v169/v170 frontier if the gain
is durable.

**Mechanism**: Exact rerun of v182: reuse `v169-no-mlp-sac` with
`--use_scan=False`, keeping all other runtime flags, environment, shape, and XLA
flags fixed.

**Predicted signal**: Support requires clean compile/loss and repeated
throughput above the confirmed v169/v170 frontier band (57,935-57,947 tok/s
average, 58,051 best observed). If it repeats near v182, carry no-scan as the
new confirmed frontier.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion back inside/below the v169/v170 frontier band.

## Setup

GKE workload `alekseyv-qwen3-v183-noscan-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v183-no-mlp-sac-no-scan-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is an exact rerun of a runtime
lowering/scheduling expression change only and preserves model math.

Per user request, CPU equivalence was rerun in the v169 dependency image with
the checked-out JAX lane mounted:

```text
==== EQUIVALENCE PASS ====
logits max delta: 2.682e-07
loss delta: 0
gradients: all 25 OK, max gradient delta 7.823e-08
```

## Results

The JobSet and pods were cleaned up before delayed log capture, so the worker
stdout throughput lines were not recoverable. The XProf Chrome traces were
present and contain device-side `jit_train_step(403931025898375493)` spans on
both hosts. These spans confirm that the steady-state no-scan schedule reruns
above the prior v169/v170 frontier band.

Profile-derived steady timing:

| Host trace | Device spans used | Avg steady step | Avg TPS | Best span TPS |
|------------|-------------------|-----------------|---------|---------------|
| `g30s` | 12/12 | 4.497381 s | 58,288 tok/s | 58,302 tok/s |
| `rkr2` | 8/12 steady spans | 4.497194 s | 58,291 tok/s | 58,301 tok/s |

The first `rkr2` captured device span was 4.6219 s and is excluded from the
steady average because it is a profiler-window edge artifact; the following two
steps on all devices match `g30s` at 4.496-4.498 s. With global batch 32 and
sequence length 8192, each step processes 262,144 tokens.

This is lower than v182's best stdout step of 58,443 tok/s, but it remains above
v169/v170's confirmed repeated-average band (57,935-57,947 tok/s) and best
observed step (58,051 tok/s). Treat the no-scan gain as confirmed.

## Profile

Profile files:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v183-no-mlp-sac-no-scan-rerun/plugins/profile/2026_06_04_21_37_36/gke-tpu-46dd3e54-g30s.trace.json.gz`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v183-no-mlp-sac-no-scan-rerun/plugins/profile/2026_06_04_21_37_36/gke-tpu-46dd3e54-rkr2.trace.json.gz`
- matching `.xplane.pb` files for both hosts.

## HLO Dump

No fresh HLO dump was emitted under the v183 run prefix, consistent with
persistent compile-cache replay of v182's unrolled no-scan program:

- HLO bytes: 19,043,435
- SHA256: `1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`
- Memory report: 27.66 GiB

## Verdict

Supported. v183 confirms that v182's no-scan improvement is not a one-off. Carry
`--use_scan=False` on top of the v169 no-MLP-intermediate-sac schedule as the
current confirmed frontier, while accounting for much larger HLO size and
heavier compile/warmup.
