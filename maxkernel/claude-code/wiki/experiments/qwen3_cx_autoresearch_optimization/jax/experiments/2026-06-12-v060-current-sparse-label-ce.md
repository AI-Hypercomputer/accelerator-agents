---
title: "Qwen3 JAX current v060 - sparse-label MaxText CE"
type: experiment
hypothesis: "Replacing dense one-hot targets in the MaxText-style CE custom VJP with sparse labels may reduce CE memory/loop-fusion work without changing logits, lm-head, or softmax math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v060-sparse-label-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, ce, cross-entropy, sparse-labels, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v060 - sparse-label MaxText CE

## Hypothesis under test

**Hypothesis**: The current frontier still materializes dense one-hot targets
inside the MaxText-style full-logits CE path. A sparse-label custom VJP can keep
the same full-logits/lm-head math while replacing `softmax - one_hot` with a
scatter subtract in the backward. This may reduce CE-related loop-fusion and
memory traffic without entering the previously fragile Tokamax streamed-CE path.

**Mechanism**: Build a thin image from `qwen3-8b-jax:v258-mlp-up-first` and
change only `train.py`:

- add `cross_entropy_with_logits_sparse_labels(logits, labels)` as a
  `jax.custom_vjp`;
- forward computes `logsumexp(logits) - logits[label]`;
- backward computes softmax probabilities and subtracts the incoming gradient at
  the sparse label index with indexed scatter-add;
- keep the old MaxText/T5X dense-target custom VJP as the default;
- enable the new path only with `USE_SPARSE_LABEL_CE=1`.

Local CPU equivalence against the existing MaxText-style CE passed:
`loss_max_abs_diff=0.0`, `grad_max_abs_diff=1.49e-08`.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or a
same-band result with a materially smaller CE/loop-fusion footprint and no MXU
or HBM regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM/load
failure, or clean completion below v036/v037 without a useful CE/HLO reduction.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v060-sparsece`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v060-sparse-label-ce`
- Image digest: `sha256:1cd8855704ed9b72399bd9cd28775868a417999601258be5e0f19cf89a269b22`
- Base image: `qwen3-8b-jax:v258-mlp-up-first`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v060-v037-sparse-label-ce`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.
- Runtime stack: v037 frontier flags plus `USE_SPARSE_LABEL_CE=1`, including
  no-scan/remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE
  entrypoint, scoped VMEM 100352, selective resources, rerun3, SparseCore
  RS/AR offload, RS3+AR2, and shared-memory limit 90.

## Result

**Verdict: refuted / invalid for the current v037 memory envelope.**

The workload failed at TPU compile before step 0 with HBM exhaustion:

- required HBM: **39.19 GiB** program memory on a **31.25 GiB** device;
- exceeded capacity by **7.95 GiB**;
- largest allocation: **18.55 GiB** `f32[4,8192,151936]` in the final
  lm-head/logits path;
- second-largest allocation: **9.27 GiB** `bf16[4,8192,151936]` in the same
  fused logits computation.

This means the sparse-label custom VJP did not reduce the CE footprint in the
way needed for this graph. It instead exposed or extended the full-logits live
range, losing the current frontier's tight HBM fit. Do not carry this sparse CE
path forward unless paired with a real streamed/fused lm-head+CE implementation
that avoids materializing full `[batch, seq, vocab]` logits.
