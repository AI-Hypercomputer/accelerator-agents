---
title: "qwen3-jax-v057-n-tokamax-layer-dots-bs8"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "This is a re-run of `v057-m` because `<your-cluster>` became occupied by a Ray job. The exact same configuration (layer-wise `checkpoint_dots` policy) will be run on `charles-v6e`."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-m-tokamax-layer-dots-bs8
---

# qwen3-jax-v057-n-tokamax-layer-dots-bs8

## Hypothesis under test

**Hypothesis**: By combining layer-wise checkpointing with the `checkpoint_dots_with_no_batch_dims` policy, we can reduce the 15 GB intermediate boundary cost of layer-wise checkpointing down to a much smaller size (just the dot product inputs), allowing `batch_size=8` and `seqlen=8192` to fit within the 31.25 GB HBM limit.

**Mechanism**:
1. Layer-wise checkpointing with `policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims`.
2. Outer `loss_fn` checkpointing with the same policy to avoid `[65536, 151936]` logits materialization.
3. Set `batch_size=8` in `train.py`.
4. Run on `charles-v6e`.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).
Cluster: `charles-v6e`

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
