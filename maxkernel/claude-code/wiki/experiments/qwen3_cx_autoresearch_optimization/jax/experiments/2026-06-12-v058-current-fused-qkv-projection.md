---
title: "Qwen3 JAX current v058 - fused QKV projection"
type: experiment
hypothesis: "Combining the attention Q/K/V input projections into one concatenated dot may reduce convolution-fusion launch/scheduling overhead while preserving Qwen3 math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v058-fused-qkv-proj
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, attention, qkv, projection, splash, tokamax-splash, maxtext-ce, sparsecore, collectives, scheduler, shared-memory, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v058 - fused QKV projection

## Hypothesis under test

**Hypothesis**: The current v037/v056/v057 profiles remain dominated by
convolution fusion and Splash custom-call time. The attention block still emits
three independent input projection dots (`q_proj`, `k_proj`, `v_proj`) from the
same hidden-state tensor. Combining those into one concatenated projection may
reduce dot launch/scheduling overhead or select a better HLO schedule.

**Mechanism**: Build a thin image from `qwen3-8b-jax:v258-mlp-up-first` and
change only `Qwen3Attention.__call__`:

- concatenate `q_proj`, `k_proj`, and `v_proj` weights on the output axis;
- run one `dot_general`;
- split the result back into Q, K, and V;
- keep the existing `checkpoint_name(..., "query_proj"|"key_proj"|"value_proj")`
  tags on the split tensors;
- keep QK norm, RoPE, Splash dispatch, output projection, MLP, CE, sharding, and
  all runtime flags fixed.

Local CPU equivalence on a small Qwen3 attention config passed exactly:
`max_abs_diff=0.0`, `allclose0=True`.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or a
same-band result with a materially smaller convolution-fusion bucket or HLO dot
count that motivates a follow-up.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM/load
failure from the concatenated projection, or clean completion below v036/v037
without a useful HLO/profile reduction.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v058-qkv`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v058-fused-qkv-proj`
- Image digest: `sha256:ac966dd97b52f72295baf87ff049aa4b94fe0119d40f83e9e540cf59e91d2a83`
- Base image: `qwen3-8b-jax:v258-mlp-up-first`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v058-v037-fused-qkv-proj`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.
- Runtime stack: v037 frontier flags, including no-scan/remat, activation
  sharding, Tokamax Splash max-logit 30, MaxText/T5X CE, scoped VMEM 100352,
  selective resources, rerun3, SparseCore RS/AR offload, RS3+AR2, and shared
  memory limit 90.

## Results

Workload `alekseyv-qwen3-cc5-jax-v058-qkv` completed cleanly, but the fused
QKV projection regressed the current-cluster frontier. The worker summary
reported **56,604 tok/s**, **7,075 tok/s/chip**, and **40.6% old MFU** over
18 measured steps. This corresponds to roughly **46.2% MaxText-style MFU** by
the lane's normalized formula, below v036/v037's roughly **48.4%** and below
the historical v273/v277 band near **49.2%**.

XProf
`2026-06-12-qwen3-jax-v058-v037-fused-qkv-proj/2026_06_12_12_53_00` reports
**4654.1 ms** step time, **64.0% MXU**, and peak HBM **28.97 GiB** with
**2.2749 GiB** free on the available host trace. The op profile still spends
most time in convolution fusion (**28,593 ms / 51.2%**) and Splash custom calls
(**14,725 ms / 26.4%**).

Optimized train-step HLO:
`/tmp/qwen3-v058-hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
with SHA
`52604a87685c61374f8f1f8a31a5941ce05ad9f930184129171b59681e70960f`,
**17,098,781 bytes / 101,543 lines**. Text counts:
`all-gather=5067`, `all-reduce=165`, `reduce-scatter=2005`,
`async-start=289`, `async-done=289`, `convolution=2173`,
`custom-call=2916`, `copy=22621`, `fusion=24875`, `dot_general=5188`,
`splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

## Verdict

**Refuted.** Full QKV fusion reduced optimized-HLO size and dot/convolution
counts, but selected a slower schedule with lower MXU utilization. Do not carry
full QKV projection fusion forward. A narrower K/V-only projection fusion is
still worth one probe because it may remove the two smaller GQA projections
without disturbing the larger Q path's layout.
