---
title: "Qwen3 JAX current v059 - fused K/V projection"
type: experiment
hypothesis: "Combining only the attention K/V input projections may reduce two small GQA projection dots without perturbing the larger Q projection schedule that full QKV fusion regressed."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v059-fused-kv-proj
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, attention, kv, projection, splash, tokamax-splash, maxtext-ce, sparsecore, collectives, scheduler, shared-memory, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v059 - fused K/V projection

## Hypothesis under test

**Hypothesis**: v058 showed that full QKV projection fusion reduces HLO size but
hurts layout/scheduling enough to regress from **59,330 tok/s** to
**56,604 tok/s**. A narrower K/V-only fusion may preserve the Q projection
schedule while removing one launch/scheduling unit for the two smaller GQA
projections.

**Mechanism**: Build a thin image from `qwen3-8b-jax:v258-mlp-up-first` and
change only `Qwen3Attention.__call__`:

- keep `q_proj(hidden_states)` exactly as in the base image;
- concatenate `k_proj` and `v_proj` weights on the output axis;
- run one `dot_general` for K/V;
- split the result back into K and V;
- keep the existing `checkpoint_name(..., "key_proj"|"value_proj")` tags on
  the split tensors;
- keep QK norm, RoPE, Splash dispatch, output projection, MLP, CE, sharding, and
  all runtime flags fixed.

Local CPU equivalence on a small Qwen3 attention config passed exactly:
`max_abs_diff=0.0`, `allclose0=True`.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or a
same-band result with a smaller attention projection footprint that motivates
carrying K/V fusion into another supported mechanism.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM/load
failure from the concatenated K/V projection, or clean completion below v036/v037
without a useful HLO/profile reduction.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v059-kv`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v059-fused-kv-proj`
- Image digest: `sha256:0fa2dee61296f1c8d0c9d7618cc9b5554350668d3889df0dbb0624ccb8ca3d0f`
- Base image: `qwen3-8b-jax:v258-mlp-up-first`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v059-v037-fused-kv-proj`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.
- Runtime stack: v037 frontier flags, including no-scan/remat, activation
  sharding, Tokamax Splash max-logit 30, MaxText/T5X CE, scoped VMEM 100352,
  selective resources, rerun3, SparseCore RS/AR offload, RS3+AR2, and shared
  memory limit 90.

## Results

Workload `alekseyv-qwen3-cc5-jax-v059-kv` completed cleanly. Early steady
steps briefly matched or slightly exceeded the current-cluster frontier
(step 2 **59,394 tok/s**, step 3 **59,340 tok/s**), but the full run settled
below v036/v037. The worker summary reported **59,177 tok/s**,
**7,397 tok/s/chip**, and **42.4% old MFU** across 18 measured steps, roughly
**48.2% MaxText-style MFU** by the lane's normalized formula.

XProf
`2026-06-12-qwen3-jax-v059-v037-fused-kv-proj/2026_06_12_13_01_55` reports
**4444.5 ms** step time, **66.7% MXU**, and peak HBM **29.01 GiB** with
**2.2382 GiB** free. The op profile still spends **53.1%** in convolution
fusion and **27.6%** in Splash custom calls.

Optimized train-step HLO:
`/tmp/qwen3-v059-hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
with SHA
`cbfbfbccc6607450269118ae0e6f16476412af89fc3b562cc709d46aca720d95`,
**18,723,015 bytes / 111,915 lines**. Text counts:
`all-gather=6137`, `all-reduce=165`, `reduce-scatter=2378`,
`async-start=394`, `async-done=394`, `convolution=2609`,
`custom-call=3561`, `copy=22566`, `fusion=28019`, `dot_general=6295`,
`splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

## Verdict

**Refuted / close neutral.** K/V-only fusion is much less harmful than full QKV
fusion and produces a smaller HLO than the v056/v057 high-side schedule, but it
does not beat v036/v037's **59,330 tok/s / 4433.0 ms** frontier. Close the
attention input-projection fusion axis for now; remaining opportunities should
target CE/lm-head, MLP/source scheduling, or collective/layout behavior rather
than further Q/K/V concatenation variants.
