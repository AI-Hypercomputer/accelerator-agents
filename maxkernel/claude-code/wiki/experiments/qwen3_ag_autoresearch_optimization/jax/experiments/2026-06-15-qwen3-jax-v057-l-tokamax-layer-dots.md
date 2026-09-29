---
title: "qwen3-jax-v057-l-tokamax-layer-dots"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "By combining layer-wise checkpointing (from `v057-g`) with the `checkpoint_dots_with_no_batch_dims` policy on both the transformer layer boundaries AND the outer `loss_fn` boundary, we can significantly reduce the boundary memory footprint without causing Pallas compiler hangs (which were observed in block remat `v057-i`). The `checkpoint_dots` policy prevents the materialization of the full logits tensor (`[65536, 151936]`) in the outer loss function, and reduces the 15 GB intermediate boundary cost of layer-wise checkpointing, allowing `batch_size=8` and `seqlen=8192` to fit within the 31.25 GB HBM limit."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-j-tokamax-layer-ckpt-bs4
---

# qwen3-jax-v057-l-tokamax-layer-dots

## Hypothesis under test

**Hypothesis**: In `v057-g`, standard layer-wise checkpointing at `bs=8` used `35.83 GB` of HBM due to storing 15 GB of boundary activations. Attempting to group layers into blocks (`v057-i`) reduced boundary memory but triggered a 60-minute Pallas compiler hang.
Applying the `jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims` policy to standard per-layer checkpointing will reduce the boundary storage by only saving the inputs to dot-products (attention weights, MLP weights) rather than full intermediate activations. This policy will also be applied to the outer `loss_fn` to ensure the full logits matrix is not materialized. This will allow the configuration to compile quickly and execute at `bs=8` without OOMing or hanging.

**Mechanism**:
1. Revert to `v057-g`'s per-layer checkpointing, but inject the policy: `self._ckpt_call = jax.checkpoint(self._call_impl, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)`
2. Apply the same policy to `loss_fn`: `jax.checkpoint(loss_fn, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)`
3. Set `batch_size=8` in `train.py`.
4. Keep Tokamax CE and Splash Attention enabled.
5. Keep `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` and Splash block variables `SPLASH_BQ=2048 SPLASH_BKV=2048 SPLASH_BQ_DKV=2048 SPLASH_BKV_DKV=2048`.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
