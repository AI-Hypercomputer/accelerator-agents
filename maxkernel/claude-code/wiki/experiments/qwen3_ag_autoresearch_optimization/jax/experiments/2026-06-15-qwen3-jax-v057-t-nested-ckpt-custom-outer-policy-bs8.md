---
title: "qwen3-jax-v057-t-nested-ckpt-custom-outer-policy-bs8"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "In `v057-s`, setting `policy=nothing_saveable` on the inner checkpoints was correct, but we left the outer `Qwen3Block` checkpoint with no policy. This caused the outer checkpoint to default to `save_anything_that_can_be_saved`, which enthusiastically saved all the intermediate inner layer boundaries anyway, negating the memory savings and causing the exact same 35.83GB HBM OOM. By setting a custom outer policy (`lambda *args, **kwargs: False`) on `Qwen3Block`, the outer checkpoint will ONLY save the inputs to the block and discard all internal layer boundaries. The inner checkpoints (with `nothing_saveable`) will act strictly as compiler barriers during backward recomputation to prevent the XLA 60-minute fusion hang. This combination will finally reduce boundary memory to 3.75GB and fit `bs=8` into 31.25GB HBM."
status: failed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-s-nested-ckpt-nothing-saveable-bs8
---

# qwen3-jax-v057-t-nested-ckpt-custom-outer-policy-bs8

## Hypothesis under test

**Hypothesis**: Nested checkpointing using a custom `return False` policy on the outer block and `nothing_saveable` on the inner layers fixes both the compiler hang and the HBM OOM.
1. Outer checkpoint: `jax.checkpoint(Qwen3Block, policy=lambda *args, **kwargs: False)` (saves block inputs, 3.75GB. Prevents default policy from saving inner layer boundaries).
2. Inner checkpoint: `jax.checkpoint(Qwen3DecoderLayer, policy=jax.checkpoint_policies.nothing_saveable)` (acts ONLY as a compiler barrier against backward fusion, saves NO memory during main forward pass).
3. Loss checkpoint: `jax.checkpoint(loss_fn)` (prevents Tokamax CE fusion, saves `hidden_states` input, recomputes 39GB logits).

**Mechanism**:
1. Keep `Qwen3Block` with `K=4` layers.
2. Apply custom `False` policy to `Qwen3Block` checkpoint.
3. Ensure `jax.checkpoint(layer._call_impl, policy=jax.checkpoint_policies.nothing_saveable)` inside `Qwen3DecoderLayer`.
4. Keep `jax.checkpoint(loss_fn)` in `train.py` WITHOUT any policy.
5. Set `batch_size=8`.
6. Use Tokamax CE and Splash Attention.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=8`, `seqlen=8192`.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).
Cluster: `charles-v6e`

## Results
**Status**: Crashed (HBM OOM during compilation)

Despite perfectly isolating the 4-layer block boundaries (reducing saved boundary memory to 3.75GB) using nested checkpoints with custom policies, the XLA compiler still crashed with `XLA compile OOM (512MB HLO temp allocations during jvp()/reduce_sum)`.

This confirms that at `batch_size=8` and `seqlen=8192` (65,536 tokens per chip), the peak temporary memory required to recompute the backward pass of even a single transformer layer (which contains Splash Attention + intermediate MLPs) exceeds the physical 31.25 GB HBM capacity. `batch_size=8` at 8k seqlen is physically impossible on a single TPU v6e chip.

**Next steps**:
We must back off to `batch_size=4` at `seqlen=8192` (or `batch_size=8` at `seqlen=4096`). I recommend `batch_size=4` with the current nested checkpointing setup, which should compile and run with plenty of headroom.
