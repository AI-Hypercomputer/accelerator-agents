---
title: "qwen3-jax-v057-h-tokamax-block-remat"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "By grouping layers into blocks of 4 and applying @nnx.remat to the block, we can reduce the activation memory footprint to fit within 31.25 GB HBM while maintaining the computational efficiency of recomputing 4 layers at a time."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-g-tokamax-layer-ckpt-fix-splash
---

# qwen3-jax-v057-h-tokamax-block-remat

## Hypothesis under test

**Hypothesis**: In `v057-g`, checkpointing every individual layer resulted in 35.83 GB of HBM usage. This is because saving the inputs to all 28 layers consumes 15 GB of HBM. By introducing `Qwen3Block` to group layers into chunks of 4 and applying `@nnx.remat` on the block level, we only save the input to every 4th layer. This reduces the saved input memory from 15 GB to 3.75 GB. Although recomputing 4 layers at a time during the backward pass increases the peak activation memory from 10.8 GB to ~43 GB, wait.
*Correction*: The compiler will materialize the 4 layers one by one inside the block if it unrolls properly, but it will keep the outputs of the 4 layers alive for the backward pass of the block. If this fits, the total memory should drop below 31.25 GB. If it OOMs due to 43 GB activation memory, we will know block remat is memory-prohibitive.

**Mechanism**:
1. Modifed `modeling_qwen3.py` to wrap every 4 `Qwen3DecoderLayer`s in a `Qwen3Block` module.
2. Applied `@nnx.remat` to the `__call__` method of `Qwen3Block`.
3. Set `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` and Splash block sizes to `2048` to avoid tracing and VMEM compilation errors.

**Predicted signal**: The model will compile and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM OOM due to excessive activation materialization during the block's backward pass.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it).

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
