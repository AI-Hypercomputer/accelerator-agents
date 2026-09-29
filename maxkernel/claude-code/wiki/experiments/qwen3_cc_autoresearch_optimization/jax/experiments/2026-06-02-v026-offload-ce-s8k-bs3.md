---
title: "Qwen3-8B jax v026 — offload-remat + tokamax-CE + bs3 @ seq8192 (full MaxText-analog stack)"
type: experiment
hypothesis: qwen3-jax-host-offload-bs3-seq8192
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-offload-2026-06-02
status: invalid
verdict: invalid
tags: [qwen3-cc, jax, offload-remat, tokamax-ce, splash, remat, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v026 — offload-remat + tokamax-CE + bs3 @ seq8192

## Hypothesis under test

**Hypothesis**: The combination that mirrors MaxText's bs3@seq8192 recipe — **tokamax streamed CE**
(removes the `[B,L,V]` logit wall that OOM'd [v025](2026-06-02-v025-offload-remat-s8k-bs2.md): 18.54G of
f32 logits/log_softmax) **+ offload-remat** (parks attention/MLP activations in host DRAM) — frees enough
HBM to fit **bs3** at seq8192 and reach toward the MaxText ceiling (45.3% / 6,942 tok/s/chip), beating the
v009 bs1 frontier (30.4% / 5,305).

**Mechanism**: `--use_remat --use_splash --offload_remat --use_tokamax_ce --tokamax_ce_impl=mosaic_tpu
--batch_size=3 --seqlen=8192` + the v018 7-flag scheduler bundle, image `qwen3-8b-jax:v025-offload` (no
rebuild — CE + offload code both present). Two complementary *memory* levers, both required to reach bs3:
CE drops the logit materialization, offload drops the activation HBM. (v025 = offload-only OOM'd; v016 =
CE-only fit bs2 but didn't beat bs1 — neither alone reaches bs3.)

**Predicted signal**: bs3 fits (CE removes 18.54G logits, offload frees activations); MFU > 30.4% /
tok-s-chip > 5,305 (v009), ideally approaching MaxText's 6,942; loss ~12.07 stable, no NaN; exit 0.

**Falsification criterion**: OOM at bs3 (combined levers still insufficient), or MFU ≤ 30.4% (the qwen3
jax stack doesn't amortize batch at seq8192 the way MaxText does — v016's bs2 regression hinted at this;
the residual gap would then be MaxText's matmul/layer-fusion efficiency, not batch, and seq8192 is at our
stack's practical ceiling).

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v025-offload`.
- **Launch**: `--use_remat --use_splash --offload_remat --use_tokamax_ce --tokamax_ce_impl=mosaic_tpu --batch_size=3 --seqlen=8192` + 7 LIBTPU scheduler flags. All three levers confirmed active in log (offload policy, splash, CE autotune).

## Results

**Compile-time HBM OOM, pre-step-0** — but **only +2.34G over** (33.59G of 31.25G), vs v025's +7.55G:
- program buffer **27.87G** + arguments **5.72G** = 33.59G.
- **CE worked**: the f32 `[B,L,V]` logit + log_softmax materialization (18.54G in v025) is GONE — tokamax streams the loss.
- **offload worked**: no-batch-dim attn/MLP dots offloaded to host.
- **Marginal blocker now**: the top-20 temps are all `bf16[3,8192,4096]` (192 MB each) — *batch-dim* dot_general/reduce_sum outputs in the backward jvp, which `offload_dot_with_no_batch_dims` deliberately does **not** offload. Plus the CE backward Pallas kernel's `w=f32[4096,151936]` (2.49G) weight shard.

## Profile

No profile/HLO train-step module (crash at compile, pre-step-0). HLO dump dir created but empty.

## Verdict

**Invalid** (compile-time OOM, no steps). But this is the **most informative OOM of the exploration**: the
two-lever stack (CE + offload-remat) shrank the bs3 overage from a logit-dominated +7.55G ([v025](2026-06-02-v025-offload-remat-s8k-bs2.md))
to a **marginal +2.34G**, and pinpointed the last blocker: the **batch-dim layer activations**
(`bf16[3,8192,4096]`) that the no-batch-dim offload policy skips. That is exactly the tensor MaxText's
`decoder_layer_input=offload` targets — the one piece of the MaxText recipe not yet ported. We are within
2.34G of fitting bs3 (the MaxText shape, 45.3% / 6,942 ceiling).

## Next hypotheses

- [offload the decoder-layer input/activations (batch-dim) too](../../../../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md) — extend the offload to the `bf16[B,8192,4096]` layer activations via `jax.ad_checkpoint.checkpoint_name` + `save_and_offload_only_these_names` (the precise MaxText `decoder_layer_input=offload` analog). Shaves >2.34G → should fit bs3. **Dispatching as v027** (code change).
- Fallback if bs3 still won't fit or doesn't beat v009 (30.4%): the seq8192 gap to MaxText is **per-token efficiency** (matmul/layer-fusion), not batch-fitting — confirmed by the v016 bs2<bs1 trend — and seq8192 is at our stack's practical ceiling. Document and close.
