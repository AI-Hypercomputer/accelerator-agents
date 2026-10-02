---
title: "qwen3-jax-v057-k-tokamax-policy"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "In `v056`, we saw that `jax.checkpoint` on `loss_fn` with the policy `checkpoint_dots_with_no_batch_dims` caused a 40 GB HBM OOM because it forces the materialization of the full logits tensor. In `v057-c`, we tried removing the policy entirely, but hit a 327 GB OOM because that disabled all intermediate checkpointing. In `v057-h/i`, we tried Block Rematerialization to save memory, but it hung the Pallas compiler for 60 minutes. Therefore, the correct path is to use per-layer `jax.checkpoint` on the transformer layers (which successfully compiled in `v057-g`) and apply a custom checkpoint policy to the outer `loss_fn` that correctly saves dots but explicitely *excludes* the `lm_head` projection, or simply don't checkpoint `loss_fn` at all since the layers are already checkpointed."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-j-tokamax-layer-ckpt-bs4
---

# qwen3-jax-v057-k-tokamax-policy

## Hypothesis under test

Wait, the previous layer-wise checkpointing run (`v057-g`) compiled successfully but OOM'd at 35.83 GB!
If we do `bs=8` with layer-wise checkpointing, it uses 15 GB of HBM just for the 28 layer boundary inputs (`8 * 8192 * 4096 * 2 bytes * 28 layers`).
If the physical limit is 31.25 GB, and we need 10 GB for weights/grads/opt + 10 GB for peak activation + 15 GB for boundaries = 35 GB.
We cannot fit `bs=8` with layer-wise checkpointing!

Wait! We don't have to save *all* the inputs at the boundaries! If we apply the `checkpoint_dots_with_no_batch_dims` policy *to the layer*, we don't need to save the 15 GB of `hidden_states`. We only save the small attention and MLP dot inputs.

Let's verify this!
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
