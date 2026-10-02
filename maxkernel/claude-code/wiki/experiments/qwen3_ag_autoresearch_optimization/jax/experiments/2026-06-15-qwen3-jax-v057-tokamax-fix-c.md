---
title: "qwen3-jax-v057-tokamax-fix-c"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "The `v056` run crashed with a 40GB HBM OOM because `checkpoint_dots_with_no_batch_dims` conflicted with Tokamax CE's VJP shape tracking, materializing the `[65536, 151936]` logits tensor. We completely disabled selective rematerialization in the outer `loss_fn` in `v057-c`, but hit a 327 GB OOM due to missing internal boundaries. However, our recent discovery in `v057-i` showed that block rematerialization (chunking into blocks of 4) triggered a 60-minute compiler hang during Pallas autotuning for the Tokamax CE loss kernel itself. This implies that the shape of the inputs to `tokamax.linear_softmax_cross_entropy_loss` when passed through an `nnx.remat` block boundary is causing catastrophic Pallas trace blowup."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-i-tokamax-block-remat-fsdp-fix
---

# qwen3-jax-v057-tokamax-fix-c

## Hypothesis under test

**Hypothesis**: Block rematerialization (applying `nnx.remat` around `Qwen3Block`) caused the XLA compiler to hang for 60+ minutes while autotuning the Tokamax CE loss kernel (as seen by the `Autotuning cache miss for PallasMosaicTpuLinearSoftmaxCrossEntropyLoss` logs right before the hang). To break the impasse, we must revert to the baseline flat transformer structure (checkpointing individual `Qwen3DecoderLayer`s instead of blocks), which successfully compiled the Pallas CE kernel in <2 mins in previous runs. To fix the HBM OOM from saving 28 layer inputs (15 GB), we will test if `use_splash=False` combined with layer-wise `jax.checkpoint` reduces the overall peak memory enough to fit within 31.25 GB, or if we just have to drop `batch_size` back to 4.

Wait, our goal is `batch_size=8`, `seqlen=8k`, `use_splash=True`, `use_tokamax_ce=True`.
Let's trace back. In `v057-d`, layer-wise checkpointing + Tokamax CE + Splash Attention compiled and traced correctly until it hit a Splash tracing assertion (`bkv=512`).
In `v057-f`, fixing the Splash tracing assertion caused a VMEM OOM (`35.6M > 32M limit`).
In `v057-g`, increasing the VMEM limit compiled successfully, but hit a `35.83GB` HBM OOM!
We diagnosed the HBM OOM as being caused by 28 layers of saved inputs (`15 GB`). We tried `Qwen3Block` to fix it, but it hung the compiler for an hour!
Instead of `Qwen3Block`, what if we go back to layer-wise checkpointing, but use a more aggressive `jax.checkpoint` policy on the *entire* `train_step` to save memory, or use `jax.checkpoint(layer, policy=checkpoint_dots)`?
If layer-wise checkpointing without policies saves all inputs (15GB), we could apply `jax.checkpoint(layer, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)` to *only* save the dot product inputs (which are much smaller) and drop the attention intermediates.

Wait, the inputs to `Qwen3DecoderLayer` are just `hidden_states`, which is `[B, L, D]`.
For B=8, L=8192, D=4096, `bfloat16`:
`8 * 8192 * 4096 * 2 bytes = 536 MB`.
`536 MB * 28 layers = 15 GB`!
Yes! The very act of saving the `hidden_states` at the boundary of each of the 28 layers costs 15 GB!

If we don't save them, we have to recompute from the beginning.
Wait, if `Qwen3Block` caused a compiler hang, maybe there's a simpler way to do gradient checkpointing every 4 layers?
We can use a simple python `for` loop with a manual checkpoint wrapper that groups 4 layers!

Let's test `batch_size=4` just to see if we can get a clean baseline of `v057-g` (layer-wise checkpointing, Tokamax CE, Splash) to prove the functionality works!
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
