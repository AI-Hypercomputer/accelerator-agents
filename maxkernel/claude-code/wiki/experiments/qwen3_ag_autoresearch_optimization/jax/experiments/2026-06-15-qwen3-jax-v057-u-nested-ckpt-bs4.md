---
title: "qwen3-jax-v057-u-nested-ckpt-bs4"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "In `v057-t`, perfectly configuring block rematerialization (custom outer policy saving 3.75GB boundaries + nothing_saveable inner barriers) proved that `batch_size=8` at `seqlen=8192` (65,536 tokens/chip) is physically impossible due to HLO temp buffer requirements exceeding 31.25GB during the backward pass recomputation. By reducing `batch_size` to 4 (32,768 tokens/chip) while keeping the same mathematically sound nested checkpointing strategy, the peak temp memory requirements will halve, allowing the model to successfully compile and run with Tokamax CE and Splash Attention."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-t-nested-ckpt-custom-outer-policy-bs8
---

# qwen3-jax-v057-u-nested-ckpt-bs4

## Hypothesis under test

**Hypothesis**: Lowering `batch_size` to 4 while using the `v057-t` nested checkpointing strategy successfully compiles and runs.

**Mechanism**:
1. Keep `Qwen3Block` with `K=4` layers.
2. Apply custom `False` policy to `Qwen3Block` checkpoint.
3. Ensure `jax.checkpoint(layer._call_impl, policy=jax.checkpoint_policies.nothing_saveable)` inside `Qwen3DecoderLayer`.
4. Keep `jax.checkpoint(loss_fn)` in `train.py` WITHOUT any policy.
5. Set `batch_size=4` (per chip).
6. Use Tokamax CE and Splash Attention.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=4`, `seqlen=8192`. This will establish the baseline MFU.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).
Cluster: `charles-v6e`

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
