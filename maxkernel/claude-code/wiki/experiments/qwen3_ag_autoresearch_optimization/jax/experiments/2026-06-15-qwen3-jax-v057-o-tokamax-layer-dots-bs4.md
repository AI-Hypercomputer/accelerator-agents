---
title: "qwen3-jax-v057-o-tokamax-layer-dots-bs4"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "This is a run to check if the `checkpoint_dots_with_no_batch_dims` policy on `loss_fn` reduces the memory enough to fit `bs=4`. The `v057-l` experiment crashed with the exact same `35.83GB` HBM OOM, implying that the policy we added did not successfully prevent the layer boundary intermediate states from being saved! However, we know `bs=4` will reduce the intermediate states by half. Let's see if `bs=4` fits with the current layer-wise checkpointing."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-n-tokamax-layer-dots-bs8
---

# qwen3-jax-v057-o-tokamax-layer-dots-bs4

## Hypothesis under test

Wait! `v057-l` (bs=4) crashed with `35.83 GB` HBM OOM!
Ah! `v057-l` wasn't `bs=4`. I submitted `v057-l` with `batch_size=8`! And `v057-j` was the `bs=4` run that we never got a result for!
Wait, did `v057-j` (bs=4) succeed? Let me check its subagent logs!
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
