---
title: "Exp 13 — Per-layer gradient_checkpoint (ACCEPTED at OOM-shapes; bs=4 seq=2048 + bs=8 seq=1024 unlocked)"
type: experiment
tags: [llama3, torchax, remat, gradient-checkpoint, accepted]
hypothesis: llama3-torchax-per-layer-remat
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp13-scan (image hf-v5)"
branched_from: v6e8-llama3-8b-torchax-20260425-exp9-splash-autotuned-bs4
verdict: supported
---

After [exp 11's](2026-04-25-exp11-remat-rejected.md) finding that an
**outer-loss-only** `gradient_checkpoint` with `nothing_saveable` doesn't
reduce the compile-time HBM peak, this experiment patches `LlamaModel.forward`
to wrap **each `LlamaDecoderLayer` call** in its own
`interop.gradient_checkpoint` scope. The 32 distinct checkpoint scopes
force XLA to schedule layer-by-layer; the recompute graph for one layer is
materialized at a time. **This is the architectural unlock that gemma4-jax
exp 49-51 found via scan-over-layers, achieved here via 30 lines of
LlamaModel-forward monkey-patching instead of a full scan rewrite.**

Result: **bs=4 seq=2048 (B·L=8192) and bs=8 seq=1024 (B·L=8192) — both
previously OOM by 7 GiB — now fit in HBM and run end-to-end.** The cost
is a 12-34 % step-time tax (less than `nothing_saveable`'s 52 %); the
benefit is doubling the per-chip compute density.

## Hypothesis under test

> Wrapping each `LlamaDecoderLayer.forward` in `interop.gradient_checkpoint`
> with `policy=nothing_saveable` creates 32 distinct checkpoint scopes; XLA
> can no longer schedule the full unrolled graph as one giant scheduling
> problem and must serialize the per-layer recompute work. Compile-time
> HBM peak should drop enough to fit `bs=4 seq=2048` and `bs=8 seq=1024`
> (which OOM'd by 7-7.5 GiB without remat AND with the outer-loss-only
> remat).

Falsification: at least one of {bs=4 seq=2048, bs=8 seq=1024} still OOMs
at compile.

## Setup

- Image: **`hf-v5`**. Adds a `--use_per_layer_remat` flag to `train.py`
  that monkey-patches `model.model.forward` (LlamaModel.forward) to wrap
  each `decoder_layer(...)` call in
  `torchax.interop.gradient_checkpoint(call, kwargs={"policy":
  nothing_saveable})`. ~30 lines diff vs `hf-v3`. Other components
  unchanged: autotuned splash from [exp 9](2026-04-25-exp9-splash-autotuned-bs4-accepted.md);
  `axis_types=(Auto, Auto)` mesh; `bf16` weights + `bf16` `mu/nu` adamw.
- Three workloads submitted in parallel:
  - **`exp 13a`**: `bs=4 seq=1024` — fitting shape, measure remat tax.
  - **`exp 13b`**: `bs=4 seq=2048` — was OOM in exp 6 + exp 11b.
  - **`exp 13c`**: `bs=8 seq=1024` — was OOM in exp 4 + exp 11c.
- Each runs `--train_steps=15`, profile at step 7, autotuned splash on.

## Results

| Sub-exp | Shape | per-chip B·L | Step | per-chip TPS | Aggregate TPS | MFU | vs exp 9 |
|---|---|---|---|---|---|---|---|
| baseline (exp 9) | bs=4 seq=1024 | 4096 | 560 ms | 7,225 | 57,799 | **36.1 %** | — |
| **13a** | bs=4 seq=1024 | 4096 | **752 ms** | 5,452 | 43,548 | **27.2 %** | −9 pp (refuted at fitting shape) |
| **13b** | bs=4 seq=2048 | 8192 | **1,212 ms** | 6,757 | 54,085 | **32.9 %** | **UNLOCKED** (was OOM 7.32 GiB) |
| **13c** | bs=8 seq=1024 | 8192 | **1,161 ms** | 7,058 | 56,455 | **35.2 %** | **UNLOCKED** (was OOM 7.48 GiB), only −1 pp |

vs exp 11 (outer-loss `nothing_saveable`):
| Sub-exp | exp 11 (outer) | exp 13 (per-layer) | Δ |
|---|---|---|---|
| 11a / 13a (bs=4 seq=1024) | 853 ms / 24.0 % MFU | **752 ms / 27.2 % MFU** | −12 % step time, +3.2 pp MFU |
| 11b / 13b (bs=4 seq=2048) | OOM 7.32 GiB | **fits, 32.9 % MFU** | UNLOCKED |
| 11c / 13c (bs=8 seq=1024) | OOM 7.48 GiB | **fits, 35.2 % MFU** | UNLOCKED |

### Compile-time HBM peak (for the OOM-shapes)

Inferred from successful compile (no OOM error in either case). Both shapes
fit in 31.25 GiB / chip with margin (no exact peak number printed; XLA only
emits a number on OOM).

## Profile

- `exp 13c` profile pulled to `/tmp/llama3_run/exp13c_profile_r{0,1}.tgz`
  (not yet uploaded to GCS — defer until exp 14 settles whether to optimize
  further at this shape).
- **xprof browser URL** (when uploaded): `http://localhost:8791/?run=2026-04-25-llama3-8b-exp13c-layer-remat-bs8-seq1k`.
- **Steps captured**: 7.

## Observations

### O1. Per-layer remat **does** reduce compile-time HBM peak; outer-loss remat **does not**.

Same `nothing_saveable` policy, same trainer, same shape — only the scope
of the `gradient_checkpoint` wrapper differs:

- **Outer (exp 11b)**: one `jax.checkpoint` around the whole loss
  function. XLA sees the 32-layer unrolled graph as one scheduling
  problem; no serialization help; OOM unchanged.
- **Per-layer (exp 13b)**: 32 separate `jax.checkpoint` calls, one per
  decoder layer. XLA must schedule each layer's recompute as a
  semantically-isolated subgraph; live-buffer footprint at any moment is
  bounded by one layer's worth.

This generalizes: **`jax.checkpoint` scope granularity is the actual lever**
for compile-time HBM, not the policy. `nothing_saveable` at outer scope =
no-op for HBM; `nothing_saveable` at per-layer scope = ~7-GiB HBM headroom
freed (this experiment).

### O2. Step-time tax is much smaller than outer-loss `nothing_saveable`.

| Wrapping scope | bs=4 seq=1024 step time | Tax vs exp 9 |
|---|---|---|
| None (exp 9) | 560 ms | — |
| Per-layer (exp 13a) | 752 ms | +34 % |
| Outer loss (exp 11a) | 853 ms | +52 % |

The outer-loss wrap re-runs the entire fwd of all 32 layers at backward
time. The per-layer wrap re-runs only one layer's fwd at a time during its
own bwd. In practice both should re-execute the same total compute, but
the per-layer scope lets XLA fuse intra-layer ops (fwd+bwd of one layer)
more aggressively. We measure ~18 pp lower tax — XLA's intra-scope fusion
appears to dominate.

### O3. Equal per-chip compute density (B·L=8192) → 13c (bs=8) is ~7 % faster than 13b (bs=4 seq=2k).

13b and 13c have the same per-chip token count (B·L = 8192). 13c's bigger
batch dimension parallelizes nicely; 13b's longer sequence pays the
attention `O(L²)` flops (splash makes attention activation memory `O(L)`
but the FLOPs still grow). Net: 13c is the better B·L=8192 shape for
throughput.

### O4. exp 13c's MFU (35.2 %) is essentially equal to exp 9's (36.1 %) — at 2× the per-chip token count.

This is the key signal that the unlocked shapes are competitive. For the
program target (fp32 master weights + seq=8192), per-layer remat is the
mandatory enabler — without it no fp32+seq=8k shape will fit.

## Verdict + reasoning

**Supported (accepted).** Per-layer remat **graduates** to the production
path for any shape that doesn't fit without it. Decision rule:

- **bs=4 seq=1024** stays on exp 9 (no remat, MFU 36.1 %) — 13a is slower
  there.
- **bs=4 seq=2048** uses 13b (was infeasible; now MFU 32.9 %).
- **bs=8 seq=1024** uses 13c (was infeasible; now MFU 35.2 %, the new
  high-density frontier).
- **All future seq=8192 / fp32-master experiments** start from this
  per-layer-remat base.

This **invalidates exp 11's "remat is refuted" verdict in the limited
sense that the policy was right, the scope was wrong**. Exp 11's outer-loss
wrap was necessary to test the canonical `make_train_step` line-fix, but
the actual fix here is per-layer.

## Next hypotheses generated

1. **`bs=2 seq=4096`** at per-layer remat — same B·L=8192, attention cost
   between 13b and 13c. Effort: S (env-only). Expected MFU ~33-34 %.
2. **`bs=1 seq=8192`** at per-layer remat — the **program target seq**.
   B·L=8192. With splash (already autotuned per [exp 10](2026-04-25-exp10-splash-autotune-multishape-potential.md))
   + per-layer remat, this *should* fit. Effort: S. **Highest priority** —
   the next concrete step on the program-target chain.
3. **`bs=2 seq=8192`** if (2) fits — would double per-chip B·L to 16384.
4. **fp32 master + bf16 compute (AMP)** at bs=4 seq=1024 baseline first
   — the simpler shape to validate the AMP plumbing. Then climb the seq
   ladder. Effort: M.
5. **Selective remat `dots_saveable`** at OOM-shapes — keeps matmul outputs
   (the big GiB-scale tensors) and recomputes only pointwise/norm/silu.
   Should reduce the 12-34 % tax we measured here while keeping the HBM
   savings. Effort: S (just policy swap).

## See also

- [Exp 9 (current best at fitting shape)](2026-04-25-exp9-splash-autotuned-bs4-accepted.md).
- [Exp 11 (outer-loss remat REFUTED)](2026-04-25-exp11-remat-rejected.md) — the experiment that taught us scope matters.
- [Gemma4 jax-exp49 (scan-over-layers)](../../../gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp49-jax-scan-layers-potential.md) — the alternative architecture for the same memory unlock; this experiment shows per-layer remat achieves it without the full scan refactor.
- [`train.py` on this branch](../train.py) — the `--use_per_layer_remat` patch (~50 lines including reconstructed mask + position_embeddings).

## Sources

- Workloads (XPK):
  - `llama3-8b-exp13a-layer-remat-bs4-seq1k`
  - `llama3-8b-exp13b-layer-remat-bs4-seq2k`
  - `llama3-8b-exp13c-layer-remat-bs8-seq1k`
- Image: `<your-registry>/test/llama3-8b-torchax-container:hf-v5`.
- Branch: `v6e8-llama3-8b-torchax-20260425-exp13-scan`.
