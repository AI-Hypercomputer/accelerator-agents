---
title: "Qwen3 JAX v011 — splash TP2 remat dots bs8"
type: experiment
hypothesis: "Splash attention plus TP=2 and per-chip batch 8 will raise throughput by improving attention memory behavior and MXU occupancy."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: unknown-image-v6e8-qwen3-8b-jax-20260602-v011-splash-tp2-remat-dots-bs8
verdict: invalid
status: invalid
backfilled: true
tags: [qwen3-cc, jax, flax-nnx, splash, tp2, batch-scaling, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v011 — splash TP2 remat dots bs8

## Hypothesis under test

**Hypothesis**: Splash attention plus TP=2 and per-chip batch 8 will raise throughput by improving attention memory behavior and MXU occupancy.

**Mechanism**: Run the native Flax NNX trainer with `--use_splash=True`, `--tp_parallelism=2`, and `--batch_size=8` at sequence length 2048. This changes the mesh to `fsdp=4,tp=2` and routes attention through the Splash kernel.

**Predicted signal**: If the hypothesis fires, the run should compile, reach the profiled window, and show lower attention-related memory traffic with higher tokens/sec than the baseline. The HLO should contain Splash/Pallas attention structure instead of the baseline XLA SDPA path.

**Falsification criterion**: Compile-time HBM OOM, no completed steps, or no throughput/MFU improvement after a valid profile.

## Setup

Backfilled from live GKE workload `alekseyv-qwen3-v011` after the `/start-experiment` session adopted it as the active track.

Image:

```text
<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v011-splash-tp2-remat-dots-bs8
```

Command:

```bash
cd /app/trainer &&
LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304'
JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v011-splash-tp2-remat-dots-bs8/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re=.*'
python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
  --batch_size=8 --seqlen=2048 --tp_parallelism=2 --train_steps=20 \
  --weights_dtype=bf16 --use_splash=True \
  --profile_dir=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v011-splash-tp2-remat-dots-bs8 \
  --profile_start_step=12 --profile_steps=3
```

Observed startup:

```text
[mesh] fsdp=4 tp=2 mesh=Mesh('fsdp': 4, 'tp': 2, axis_types=(Auto, Auto))
[attn] splash kernel selected (JAX_ATTENTION_IMPL=splash); mesh registered
[data] fake (random ints) global_batch=32 seqlen=2048
[train] starting train_steps=20 per_chip_batch=8 global_batch=32
[splash] bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 fused_bwd=True bq_dq=None bkv_dq=None
```

## Results

No training step completed. Worker 0 exited with `EXIT_CODE=1`.

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 35.21G of 31.25G hbm.
Exceeded hbm capacity by 3.96G.

Total hbm usage >= 35.46G:
    reserved        260.00M
    program          29.49G
    arguments         5.72G

Program hbm requirement 29.49G:
    HLO temp         29.46G
```

Largest reported allocation:

```text
Size: 2.32G
Operator: op_name="jit(train_step)/forward_backward/jvp()/dot_general"
Shape: bf16[8,2048,75968]
Allocation type: HLO temp
```

## Profile

No XProf trace was captured because the program failed during compile before `profile_start_step=12`.

## HLO Dump

HLO dump path:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v011-splash-tp2-remat-dots-bs8/hlo/
```

Artifact count: 479 GCS objects.

Dominant optimized train-step module:

```text
module_0089.jit_train_step.cl_854318611.after_optimizations.txt
size: 18.25 MiB
```

Headline HLO signal: the OOM is dominated by materialized CE/log-softmax tensors:

```text
f32[16384,75968] log_softmax/sub/exp
bf16[8,2048,75968] and f32[8,2048,75968] tensors in jvp(log_softmax)
```

This points at the lm-head / softmax CE memory wall rather than an attention-only issue.

## Verdict

`invalid`.

The configuration does not fit in v6e-8 HBM. The failure is informative: per-chip batch 8 with TP=2 and Splash still materializes the `[B,S,V]` CE/log-softmax path and exceeds HBM by about 4 GiB. The next viable experiment should reduce batch or replace plain CE with streamed/chunked CE before retrying this occupancy regime.

## Next hypotheses

- Retry a smaller occupancy point, e.g. TP=2 + Splash at batch 4 if not already recorded.
- Prioritize tokamax streamed CE before batch 8, because the `[B,S,V]` tensors are the largest OOM signal.
