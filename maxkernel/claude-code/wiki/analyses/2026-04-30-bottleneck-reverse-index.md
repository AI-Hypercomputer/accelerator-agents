---
title: "Bottleneck reverse-index — profile bucket → candidate levers"
type: analysis
tags: [bottleneck, profile, hypothesis-generation, reverse-index, methodology]
created: 2026-04-30
updated: 2026-04-30
---

A reverse index keyed by xprof profile bucket. Given a profile that shows category X dominant or anomalous, jump straight to: (a) the concepts/techniques known to move that bucket on TPU, (b) hypotheses already tried in this wiki for that bucket and the verdicts, (c) open hypotheses tagged for it. The point is to short-circuit the rediscovery tax — never re-propose Pallas RMSNorm if the HLO check already showed XLA fuses it.

> **⚠ Not exhaustive — and not a fence.**
> The buckets and levers below capture *what this wiki has measured so far* (as of 2026-04-30, anchored on the [Llama 3 8B converged-stack bottleneck breakdown](../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md)). They are a starting shortlist, not a closed set. Novel optimizations land on TPU regularly — new Pallas kernels, new XLA passes, new SparseCore primitives, new HW generations — and bottleneck patterns shift as the stack converges. **Always pair this index with first-principles profile reading and HLO inspection.** If you propose a lever that isn't on this page and it pans out, fold it back here as part of the experiment writeup. Treat absence-from-this-list as "haven't tried it here yet," not "won't work."

## How to use this page

1. Capture or open the profile. Identify the dominant 1–3 buckets by `% of step time` (canonical category names in the [bottleneck-breakdown observation](../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md)).
2. Look up each bucket below.
3. Read **Tried & refuted** first — saves writing a hypothesis already disproved.
4. Pick from **Standing levers** + **Open hypotheses** to seed the experiment's `## Next hypotheses` section.
5. If the dominant bucket isn't listed below, add a section for it as part of your writeup.

---

## Buckets

### Convolution fusion (matmul) — `kind=kOutput` Mosaic kernels

The big rectangle of step time when the model is compute-healthy. On a saturated stack this is 50–60 % and rising; the rest is non-matmul tax. Diagnostic split is *time-share* vs *tile efficiency*:

- **Time-share too small** → not enough matmul work per non-matmul work. Levers: increase per-chip B·L (raise batch, raise seq), reduce non-matmul tax in other buckets.
- **Tile-efficiency low** (per-matmul achieved/peak < 0.5) → tile shapes don't match the MXU. Levers: dimension-alignment, custom XLA tile-shape flags, MaxText-style XLA stack.

**Standing levers**:
- [dimension-alignment](../concepts/dimension-alignment.md) — batch/hidden divisibility for the MXU (128×128 v5e/v5p; 256×256 v6e).
- [dtype-strategy](../concepts/dtype-strategy.md) — bf16 compute is non-negotiable; fp32 master only on weights/optimizer state.
- [int8-quantization](../concepts/int8-quantization.md) — shifts the critical batch; per [2026-04-27 log entry](../log.md), the only meaningful per-chip throughput lever above bf16-MXU saturation.
- [xla-flags](../concepts/xla-flags.md) + [latency-hiding-scheduler](../concepts/latency-hiding-scheduler.md) — MaxText's `xla_flags_library.py` HOST_OFFLOAD scheduler bundle was the breakthrough lever (+10 % alone) on Llama 3 8B.
- Custom Pallas matmul prologue — fuse bf16 weight cast / RMSNorm into the matmul; speculative; not yet attempted in this wiki.

**Tried & refuted**:
- *Recipe XLA flags at small per-chip B·L* — flat ±noise at B·L = 4,096; needs ≳ 10,000 to pay. See [llama3-8b exp 7](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp7-splash-xla-bs4-rejected.md).

**Open hypotheses**:
- [int8 weight-only quantization (AQT/qwix)](../hypotheses/llama3-jax-int8-weight-quantization.md) — +15-30 % step time, medium confidence, L effort.

---

### Attention (splash MHA fwd + bwd)

Typically 15–25 % on a healthy stack. Both directions matter — bwd is often ~1.5× the fwd cost on TPU because dQ/dK/dV is two separate matmuls. The dkv bwd kernel is usually the single largest custom-call cost.

**Standing levers**:
- [splash-attention](../concepts/splash-attention.md) + [attention-block-sizes](../concepts/attention-block-sizes.md) — `block_q`/`block_kv` autotune. For Llama 3 8B / v6e-8, `bkv=2048` matches MaxText (+0.7 %); kernel-only sweep gave +30-32 % on fwd+bwd vs production defaults at seq=1024 ([exp 8](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp8-splash-kernel-autotune-potential.md)).
- [tokamax-shipped splash variant](../sources/2026-tokamax-splash-attention.md) with perf knobs — `use_base2_exp=True`, `fuse_reciprocal=True`, `max_logit_const=30`. +4.4 % on Llama 3 8B vs jax-experimental ([exp 72a](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md)).
- [base2-softmax](../concepts/base2-softmax.md) — exploit TPU's native base-2 exp.
- [ring-attention](../concepts/ring-attention.md) — long-seq path; relevant when single-chip can't hold KV. Pallas references in [pallas-kernel directory](2026-04-23-pallas-kernel-directory.md): maxdiffusion, haoliuhl, ejkernel.
- Faster custom dkv kernel — speculative; MaxText reference at 44.6 % MFU likely has a more efficient bwd; pull in for comparison.

**Tried & refuted**:
- *Splash block-size sweeps post-converged-stack* — bkv=2048 already optimal for Llama 3 8B at seq=8192 (exp 67 / exp 68 / exp 80 sweeps).

**Open hypotheses**: *None filed.* Faster splash bwd is a deep-rewrite candidate — Pallas authoring effort, not a config change.

---

### Cross-entropy loss (fwd + bwd)

Pre-tokamax-CE this was 7.5 % of step time on Llama 3 8B. With the right kernel it collapses to ~0 %. **Asymmetry warning**: bwd is typically ~7× the fwd cost on naïve impls (logit recomputation in VMEM).

**Standing levers**:
- [memory-efficient-cross-entropy](../concepts/memory-efficient-cross-entropy.md) — fused linear+log-softmax+NLL, avoids `[B, V]` logits materialization (~6 GiB savings on Llama 3 8B at bs=3 seq=8192).
- tokamax CE `chunked_xla` impl ([tokamax — supported ops](../sources/2026-tokamax-supported-ops.md)) — XLA-matmul chunks; out-performed `mosaic_tpu` impl on Llama 3 8B by collapsing CE bwd to ~0 % ([exp 62b](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md)).
- Persistent logit-block residuals on bwd — speculative; save `(B*L_block, V_block)` from fwd to skip recompute.

**Tried & refuted**:
- *bf16-native CE on Llama 3 8B* — `invalid` (loss diverges); fp32 cast at the CE boundary required, except for kernels that accept bf16 grad output.
- *levanter fused CE with logit soft-cap on gemma4* (exp 47) — refuted; same boundary-tax mechanism as Pallas-replacing an op XLA already fuses well.

**Open hypotheses**: *None.*

---

### Loop fusion (RMSNorm / silu / residual / mul)

The vector-op tail. Typically 13–15 % of step on a converged stack, **HBM-BW utilization 0.75–0.77 — already at hardware limit**. No throughput recoverable from this layer except by *reducing how much vector work hits HBM in the first place*.

**Standing levers**:
- [xla-fusion](../concepts/xla-fusion.md) — XLA already inlines RMSNorm into Q/K/V matmul prologues (`kind=kOutput` Mosaic kernels) and SwiGLU into the down_proj matmul. Verified via HLO inspection on Llama 3 8B exp 28b ([log 2026-04-27](../log.md)).
- MaxText-style layer fusion — fuse residual + norm + projection across layers; not yet tried in this wiki.

**Tried & refuted**:
- *Pallas RMSNorm + matmul-prologue fusion* — REFUTED 2026-04-27 by HLO inspection. XLA already does it. See [hypothesis page](../hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md).
- *Pallas SwiGLU + down_proj fusion* — REFUTED 2026-04-27 by HLO inspection. XLA already does it. See [hypothesis page](../hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md).
- *pallas-forge SwiGLU on v5e* — loses 35 % to XLA. See [gemma4 retrospective](2026-04-24-gemma4-jax-ceiling-and-process-retrospective.md).
- *Pallas RMSNorm on gemma4* (exp 33) — −8.1 % rejected; XLA-fusion sufficient.

**Open hypotheses**: *None — bucket considered closed for naïve Pallas-replacement.* Reopen only with a layer-fusion-across-layers idea, not a single-op replacement.

---

### Collectives (all-gather / reduce-scatter / all-reduce)

FSDP and TP traffic. On v6e-8 single-slice with FSDP, all three collectives traverse the TensorCore by default. The Llama 3 8B breakthrough (+3.4 %) was offloading **all three** to the SparseCore.

**Standing levers**:
- [SparseCore offload of FSDP collectives](../concepts/sparsecore.md) — `xla_tpu_enable_sparse_core_collective_offload_{all_reduce,reduce_scatter,all_gather}=true`. All three together is the win; each individually is small. See [exp 27/28b](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md).
- [async-collectives](../concepts/async-collectives.md) + [latency-hiding-scheduler](../concepts/latency-hiding-scheduler.md) — XLA scheduling pass that overlaps collectives with compute.
- [send-recv-done](../concepts/send-recv-done.md) async pattern.
- FSDP↔TP↔CP rebalance — when ICI utilization is the bottleneck, shifting axes can help.

**Tried & refuted**:
- *TP=2 mix on gemma4* — −14 % vs FSDP-only; TP doesn't pay below ICI-island fanout.

**Open hypotheses**: *None filed.* SC offload is fully on; further wins likely require multi-slice (DCN) regime change.

---

### HBM peak / OOM (compile-time)

Not a step-time bucket but a *fitting* constraint shaping which (bs, seq) regimes are reachable. Reduce peak HBM and you unlock larger per-chip B·L, which then moves matmul time-share up.

**Standing levers**:
- [rematerialization](../concepts/rematerialization.md) — selective remat; `nothing_saveable` is too aggressive.
- [scan-over-layers](../concepts/scan-over-layers.md) — `jax.lax.scan` pattern; collapses N-layer compile time and HBM staging.
- [host-offload](../concepts/host-offload.md) — async host↔HBM transfer for opt state, activations, layer inputs. MaxText `decoder_layer_input: offload` + per-projection offload is the recipe-tuned pattern.
- [memory-efficient-cross-entropy](../concepts/memory-efficient-cross-entropy.md) — avoids `[B, V]` logit buffer; ~6 GiB savings on Llama 3 8B / bs=3 / seq=8192.
- [training-memory-budget](../concepts/training-memory-budget.md) accounting first; don't optimize blind.

**Tried & refuted**:
- *`nothing_saveable` remat at fitting shape* (torchax exp 11) — 52 % step-time tax; doesn't fix non-fitting shapes' peak. **Discovery**: torchax `make_train_step`'s `remat_policy` arg is silently ignored upstream. Recorded in the [index ledger](../index.md) under llama3-8b experiments.

**Open hypotheses**: *None filed.* Llama 3 8B currently fits at bs=4 seq=8192 with full SC offload; bs=4 is the sweet-spot per [exp 28b](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md).

---

### Data formatting / copy / bitcast / reshape

Usually <5 % on a healthy stack, HBM-BW saturated. When it grows, symptomatic of layout mismatches between consumer and producer ops.

**Standing levers**:
- [dimension-alignment](../concepts/dimension-alignment.md) — pad shapes to MXU multiples; XLA inserts reshapes when divisibility fails.
- HLO inspection — `bitcast` chains in the post-optimization HLO point to layout-assignment fights.

**Tried & refuted**: *None recorded.*

**Open hypotheses**: *None filed.*

---

### Buckets seen but not yet measured here

Keep these in mind when profiles from new programs / hardware land:

- **Megascale / DCN collectives** — multi-slice regime; not present at v6e-8 single-slice. See [megascale-stats](../sources/2026-xprof-megascale-stats.md).
- **Embedding-grad / SparseCore embedding** — likely meaningful in MoE / large-vocab regimes.
- **Host overhead / Python tracing** — [tpu-duty-cycle](../concepts/tpu-duty-cycle.md) < 100 % flags this. Levers: `jax.jit`, persistent compile cache, `donate_argnums`.
- **VMEM spill** — kernel-config-dependent; surface via [llo-utilization](../concepts/llo-utilization.md). When a Pallas kernel exceeds [vmem-budget](../concepts/vmem-budget.md) (v6e = 96 MiB, v7 = 48 MiB), the runtime spills to HBM and tile-efficiency collapses.

---

## How this page is maintained

- **Bootstrap data source**: [llama3-8b-torchax converged-stack bottleneck breakdown](../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md) (xprof exp 61b + exp 79).
- **Update trigger**: any new observation that introduces a bucket not listed; any experiment whose verdict changes the *Tried & refuted* roster; any hypothesis whose status flips.
- **Future automation**: when concept and hypothesis frontmatter gain a `bottleneck_bucket:` field, this page can be regenerated as a LINT subtask. Not yet wired up.

## See also

- [SCHEMA](../../SCHEMA.md) — operating contract; the experiment template's `## Next hypotheses` section is the primary downstream consumer of this page.
- [llama3-8b converged-stack bottleneck breakdown](../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md) — quantitative bucket %s from the data source.
- [Pallas kernel directory](2026-04-23-pallas-kernel-directory.md) — function→kernel cross-reference for the kernels named above.
- [llama3-8b day summary](2026-04-26-llama3-8b-torchax-day-summary.md) — narrative version of the climb that populated several buckets here.
- [gemma4 process retrospective](2026-04-24-gemma4-jax-ceiling-and-process-retrospective.md) — generalizable lessons feeding the *Tried & refuted* lists.

## Sources

- `wiki/observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md` — canonical bucket %s.
- xprof MCP tools (`get_overview`, `get_op_profile`, `get_top_hlo_ops`).

