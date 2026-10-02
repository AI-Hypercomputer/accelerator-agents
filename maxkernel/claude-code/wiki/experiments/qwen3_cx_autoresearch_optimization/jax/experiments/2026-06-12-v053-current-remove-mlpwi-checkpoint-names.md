---
title: "Qwen3 JAX current v053 - remove MLP checkpoint names"
type: experiment
hypothesis: "Removing unused MLP checkpoint_name wrappers may let XLA fuse the hot MLP gate/up/SwiGLU region more cleanly."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v053-remove-mlpwi-names
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, checkpoint-name, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v053 - remove MLP checkpoint names

## Hypothesis under test

**Hypothesis**: The current v036/v037 profile is dominated by MLP/dense
convolution fusion. The v258 source-order change helped historically, but the
MLP input projections are still wrapped in `checkpoint_name(..., "mlpwi")`.
Under the carried runtime (`use_remat=True`, `offload_remat=False`) the layer
remat policy is `nothing_saveable`, so the `mlpwi` names are not used for
saving or offload. Removing those name primitives may reduce artificial HLO
barriers in the hot MLP gate/up/SwiGLU region while preserving Qwen3 math.

**Mechanism**: Derive a thin image from `qwen3-8b-jax:v258-mlp-up-first` and
change only `Qwen3MLP.__call__`:

```python
u = self.up_proj(x)
g = self.gate_proj(x)
return self.down_proj(jax.nn.silu(g) * u)
```

Keep the v037 current-root runtime stack fixed: no-scan, remat with
`nothing_saveable`, Tokamax Splash max-logit 30, MaxText/T5X CE, activation
sharding, batch size 4, seq8192, RS3+AR2 SparseCore offload, selective
resources, scoped VMEM 100352, and shared-memory limit 90.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or
same-band throughput with materially lower convolution-fusion time or reduced
HLO/fusion count.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO with no throughput/profile gain, or clean completion below the v036/v037
frontier.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v053-nomlpwi`
- Base image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v053-remove-mlpwi-names`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v053-v037-remove-mlpwi-names`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Results

The workload completed cleanly with finite loss and no runtime correctness
signal. Worker summaries:

- Worker0: **58,956 tok/s**, **7,369 tok/s/chip**, **42.3% old MFU**
- Worker1: **58,976 tok/s**, **7,372 tok/s/chip**, **42.3% old MFU**

XProf:

- Run: `2026-06-12-qwen3-jax-v053-v037-remove-mlpwi-names/2026_06_12_11_50_32`
- Step time: **4463.7 ms**
- MXU utilization: **66.7%**
- Peak HBM: **29.03 GiB** of **31.25 GiB**; free **2.2202 GiB**
- Op profile: convolution fusion **57,024.949 ms / 53.2%**,
  custom-call **29,412.429 ms / 27.5%**, loop fusion
  **10,715.727 ms / 10.0%**

Optimized train-step HLO:

- Local copy:
  `/tmp/qwen3-v053-hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `bed1332e27eab6acbdd164754cc1914618bbd1848c10cc7357282c7d968bef8b`
- Size: **19,915,660 bytes / 119,349 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`,
  `dot_general=7122`, `splash=2211`, `tokamax=2`,
  `jvp_jit_silu__=576`

## Verdict

Refuted. Removing the unused `mlpwi` `checkpoint_name` wrappers changes the HLO
hash but not the HLO size, line count, memory footprint, or instruction-family
counts. Runtime regressed versus the current-cluster v036/v037 frontier:
**58,976 tok/s / 4463.7 ms** versus **59,330 tok/s / 4433.0 ms**. Convolution
fusion time also moved in the wrong direction versus v037. Do not carry this
source-layout change.
