---
title: "Qwen3 JAX current v055 - remove attention projection checkpoint names"
type: experiment
hypothesis: "Removing unused attention projection checkpoint_name wrappers may reduce artificial HLO barriers on the current no-scan frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v055-remove-attn-proj-names
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, attention, checkpoint-name, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v055 - remove attention projection checkpoint names

## Hypothesis under test

**Hypothesis**: The carried v036/v037 runtime uses layer remat with a
`nothing_saveable` policy, so attention projection names (`query_proj`,
`key_proj`, `value_proj`, `out_proj`) are not used for saving or offload.
Removing those name primitives may reduce HLO scheduling barriers or copy
pressure around the attention projection and Splash boundary without changing
Qwen3 math.

**Mechanism**: Derive a thin image from
`qwen3-8b-jax:v258-mlp-up-first` and change only
`Qwen3Attention.__call__` so q/k/v/o projections are not wrapped in
`checkpoint_name`. Projection order, QK norm, RoPE, Splash dispatch, output
reshape, MLP code, decoder-layer input naming, optimizer, and loss remain
unchanged.

Keep the v037 current-root runtime stack fixed: no-scan, remat with
`nothing_saveable`, Tokamax Splash max-logit 30, MaxText/T5X CE, activation
sharding, batch size 4, seq8192, RS3+AR2 SparseCore offload, selective
resources, scoped VMEM 100352, and shared-memory limit 90.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or a
same-band result with materially lower attention custom-call/copy/fusion time.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO with no throughput/profile gain, or clean completion below the v036/v037
frontier.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v055-noattnnames`
- Base image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v055-remove-attn-proj-names`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v055-v037-remove-attn-proj-names`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Results

The workload completed cleanly with finite loss.

- Worker0 summary: **58,977 tok/s**, **7,372 tok/s/chip**,
  **42.3% old MFU**
- Worker1 summary: **58,954 tok/s**, **7,369 tok/s/chip**,
  **42.3% old MFU**

XProf:

- Run:
  `2026-06-12-qwen3-jax-v055-v037-remove-attn-proj-names/2026_06_12_12_11_04`
- Step time: **4463.0 ms**
- MXU utilization: **66.4%**
- Peak HBM: **29.03 GiB** of **31.25 GiB**; free **2.2202 GiB**
- Op profile: convolution fusion **28,204.246 ms / 52.7%**,
  custom-call **14,713.865 ms / 27.5%**, loop fusion
  **5,334.778 ms / 10.0%** on the one-host profile view

Optimized train-step HLO:

- Local copy:
  `/tmp/qwen3-v055-hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `a090c01fb3b2871541ebdb5f19e89c64f50a73f7575e3ebf927251ae13e42e59`
- Size: **19,915,660 bytes / 119,349 lines**
- Text counts: `all-gather=6997`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=394`, `async-done=394`,
  `convolution=3081`, `custom-call=3849`, `copy=22117`,
  `fusion=30896`, `dot_general=7302`, `splash=2861`, `tokamax=2`,
  `jvp_jit_silu__=576`

## Verdict

Refuted. Removing attention projection names is slower than the current-cluster
v036/v037 frontier (**58,977 tok/s / 4463.0 ms** versus
**59,330 tok/s / 4433.0 ms**) and does not reduce HBM. It also moves the HLO
text-count profile back toward the larger old-v273 family (`custom-call=3849`,
`copy=22117`, `fusion=30896`) instead of the leaner v037 family
(`custom-call=2256`, `copy=9918`, `fusion=22711`). The attention projection
`checkpoint_name` wrappers are useful compiler steering and should be kept.
