---
title: "Qwen3 JAX current v061 - MLP multiply operand order"
type: experiment
hypothesis: "Swapping the SwiGLU elementwise multiply operands from silu(gate) * up to up * silu(gate) may perturb XLA's MLP fusion schedule without changing model semantics or the logits/CE memory envelope."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v061-mlp-mul-order
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, fusion, operand-order, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v061 - MLP multiply operand order

## Hypothesis under test

**Hypothesis**: The current frontier spends most device time in MLP-related
convolution fusions. A semantics-preserving source-level operand-order change
inside SwiGLU may change the compiler's producer/consumer schedule enough to
recover some of the v273/v277-to-current environment gap without increasing the
full-logits CE live range.

**Mechanism**: Build a thin image from `qwen3-8b-jax:v258-mlp-up-first` and
change only `Qwen3MLP.__call__`:

- baseline: `down_proj(silu(gate_proj(x)) * up_proj(x))`;
- v061: `down_proj(up_proj(x) * silu(gate_proj(x)))`;
- keep checkpoint names, projection evaluation order, parameter names,
  sharding, remat policy, CE, attention, and all backend flags fixed.

The multiplication is commutative for this bf16 elementwise op, so the output
distribution should be unchanged at fixed seed. This is intentionally a
schedule probe, not a mathematical optimization.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or a
near-tie with a lower MLP convolution-fusion profile bucket and no HBM
regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM/load
failure, or clean completion below v036/v037 without useful HLO/profile
evidence.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v061-mlpmul`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v061-mlp-mul-order`
- Image digest: `sha256:cf397f1bebed6bd46466e0c0cfa0f05e694db4e4ca6d7d519cd881f3b87c2813`
- Base image: `qwen3-8b-jax:v258-mlp-up-first`
- Base digest: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v061-v037-mlp-mul-order`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.
- Runtime stack: v037 frontier flags, including no-scan/remat, activation
  sharding, Tokamax Splash max-logit 30, MaxText CE entrypoint, scoped VMEM
  100352, selective resources, rerun3, SparseCore RS/AR offload, RS3+AR2, and
  shared-memory limit 90.
- Local CPU equivalence: exact on a small MLP config (`max_abs_diff=0.0`,
  `exact_equal=True`).

## Result

**Verdict: refuted.**

The workload completed cleanly with finite loss but regressed below the current
v036/v037 frontier:

- worker summary: **58,948 tok/s**, **7,368 tok/s/chip**, **42.3% old MFU**;
- XProf:
  `2026-06-12-qwen3-jax-v061-v037-mlp-mul-order/2026_06_12_13_18_44`;
- XProf step time: **4460.5 ms** average, **0.4 ms** stddev;
- MXU utilization: **66.7%**;
- peak HBM: **29.03 GiB / 31.25 GiB**, with **2.2202 GiB** free.

The optimized train-step HLO was copied to `/tmp/qwen3-v061-hlo`:

- SHA256: `ad313e21729d6b4a8c360f50b4ab5fc9b2a17e69779a8c004c486a655a72ef06`;
- size: **19,915,775 bytes / 119,351 lines**;
- counts: `all-gather=6997`, `all-reduce=165`, `reduce-scatter=2774`,
  `async-start=394`, `async-done=394`, `convolution=3081`,
  `custom-call=3849`, `copy=22117`, `fusion=30896`, `dot_general=7302`,
  `splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

The source derivation was clean: the copied model file differs from the base
image only by the MLP multiply line, and local CPU equivalence was exact.
However, that one-line source perturbation expanded the optimized HLO into the
larger old-v273-style structure and slowed the schedule. Do not carry the
operand-order swap.
