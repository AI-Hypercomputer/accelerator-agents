---
title: "MaxText gap-closing phase — summary (v024–v032): what was ported, what closed, what remains"
type: analysis
tags: [qwen3-cc5, jax, maxtext, v6e-8, phase-summary, gap-analysis]
created: 2026-06-12
updated: 2026-06-12
---

# MaxText gap-closing phase — summary

User directive: *"find maxtext recipe for similar size qwen3, run it and
analyze its performance, then find a way to match its performance on jax
lane."* This phase ran the reference
([mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md):
MaxText qwen3-8b @ bs3/seq8192 = **6,953 tok/s/chip**), ported its
mechanisms through nine experiments (v024–v032), moved the jax-lane
frontier **5,898 → 6,040 tok/s/chip (+2.4%, 33.6% → 34.6% MFU)**, and
terminated with the residual −13.1% gap **fully attributed and quantified**.

## Frontier movement

| Milestone | Config | tok/s/chip | MFU |
|---|---|---|---|
| Phase start (v019) | bs1, unrolled, save_attn+skip5, vmem flag | 5,898 | 33.6% |
| v026 | bs3, scan + offload_attn + **completed 25-flag stack** | 5,933 | 34.0% |
| **v027 (standing frontier)** | **bs4, same** | **6,040** | **34.6%** |
| MaxText reference | bs3, flash 2048 fused, custom remat+offload, 24 usable flags | 6,953 | ≈39.6% (our accounting) |

Recipe: trunk `174efd7`, image `qwen3-8b-jax:latest`, launch flags in the
[model page](../models/qwen3-cc5-jax.md) How-to-run.

## What the port established (mechanism scoreboard vs MaxText)

| MaxText ingredient | Port outcome |
|---|---|
| Host-offload remat (activations → pinned host) | **Matched** — but only the full triple works: offload alone OOMs unrolled (v024: liveness), fits-but-slow under scan alone (v025: 416 ms DMA stalls), and becomes free (0.7 ms) only WITH the completed flag stack (v026). bs3–bs5 all fit (v014's bs3-OOM wall eliminated). |
| Recipe flag stack | **Matched+** — 13 unadopted flags landed +10.3% (v026); windowed-einsum disable alone −341 ms. 2 SparseCore flags compile-fatal on this libtpu (mt-v001 attempt 1); 1 pipeliner-fusion flag retired. |
| Attention kernel (flash 2048 fused-bwd) | **We are FASTER** — splash beats their flash by 89 ms/step at the identical shape despite 2× kernel invocations (their residuals-save costs more than our remat re-runs). Block sweeps triple-refuted (v016/v022/v028). |
| Batch (bs3) | **Exceeded** — our optimum is bs4 (v027); bs5 flat (v029), bs6 OOM. Batch was NOT the gap driver (+1.8% at bs3→bs4 measured vs the +17.9% the original mt-v001 analysis attributed to it — corrected with a contradiction marker). |
| Scanned decoder with overlapped weight AGs | **NOT matched — the entire residual gap.** See below. |

## The residual gap: −13.1%, one cause

[Bucket-diff analysis](2026-06-12-maxtext-vs-jax-bucket-diff.md) at the
shape-identical bs3 point: of MaxText's 607 ms advantage, **+451 ms (74%)
is exposed FSDP all-gather wait** — they overlap layer N+1's weight gather
under layer N's compute inside their scanned decoder (~9 ms exposed); our
scan exposes 460–791 ms (bs3→bs5, async-done bucket). Identical
reduce-scatter bytes — pure scheduling. Secondary: +177 ms conv/MXU
(largely stall feedback), +70 ms chunked-CE (structural: XLA-native CE
cannot fit our vocab at these batches).

**The flag-level path is exhausted** (hypothesis
[qwen3-jax-scan-ag-overlap](../hypotheses/qwen3-jax-scan-ag-overlap.md)
refuted): `while_loop_double_buffering` absent from this libtpu (v030);
`collective_pipeliner` + `ici_ag_pipelining` fire-and-hurt at −1.8% with a
new 138 ms synchronous-AG bucket (v031, profiled); `pipelined_loop_unrolling`
a silent no-op — HLO unchanged, step flat (v032). The unrolled-loop escape
hatch is structurally dead with offload (59.67 G/67.67 G OOMs, v024/v030-B).

**Surviving route**:
[qwen3-jax-manual-weight-prefetch](../hypotheses/qwen3-jax-manual-weight-prefetch.md)
(L effort, open) — explicit double-buffered weight gathering inside the
scan body, MaxText-style. Expected ceiling ≈ +600 ms/step ≈ +11%, which
would land ~6,700 tok/s/chip — within ~4% of MaxText, with the remainder
being their MXU/layout edge.

## Experiment ledger (this phase)

| Exp | What | Verdict | Headline |
|---|---|---|---|
| [mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md) | MaxText reference | supported | 6,953 tok/s/chip; SC flags compile-fatal |
| [v024](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v024-offload-bs3.md) | offload, unrolled, bs3 | refuted | 59.41 G liveness OOM; mechanism proven (S(5), parity) |
| [v025](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v025-offload-scan-bs3.md) | offload + scan, bs3 | refuted | FITS (27.34 G) but 5,381 — DMA stalls 416 ms; batch not the driver |
| [v026](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v026-mt-flags-bs3.md) | + 13 recipe flags | **supported, frontier** | 5,933 (+10.3%); windowed-einsum off; DMA hidden; merged `174efd7` |
| [v027](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v027-bs4-mtfl.md) | bs4 | **supported, frontier** | **6,040 / 34.6% MFU**; HBM 23.72 G |
| [v028](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v028-splash-blocks-bs4.md) | splash block sweep | refuted | sym-1024 −2.5%; all-2048 flat; 3rd flat result — kernel closed |
| [v029](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v029-bs5-probe.md) | bs5 | refuted | fits (27.80 G) but −0.5%; batch closed at bs4; bs6=OOM |
| [v030](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v030-ag-overlap.md) | dbl-buffer flag + unrolled | inconclusive | flag absent; unrolled 67.67 G OOM (structural) |
| [v031](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v031-ag-pipeline-flags.md) | pipeliner flags | refuted | fire-and-hurt −1.8%; +138 ms sync-AG bucket |
| [v032](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v032-loop-unroll.md) | loop-unroll flag | inconclusive | silent no-op (HLO unchanged); flat in 7 steady steps; spot-exhausted |

Supporting analyses:
[bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md);
observation
[host-offload-unrolled-backward-liveness](../observations/host-offload-unrolled-backward-liveness.md).

## Lane state at phase close

- **Frontier: v027 — 6,040 tok/s/chip / 34.6% MFU / 5,425 ms @ bs4/seq8192.**
  Session total: 20.5% → 34.6% MFU (baseline seq2048 → target seq8192).
- Open queue: manual-weight-prefetch (**L**, the gap route);
  vocab-sharded CE (L, deprioritized). The S/M queue is **dry** — per
  program discipline the lane moves to **holding** pending a human decision
  on committing L-effort to the scan-body prefetch port.
- Spot-reclaim pressure on the cluster was high at phase end (5 preemptions
  across v031–v032).

## Phase 2 addendum (v044–v051) — MAXTEXT MATCHED AND EXCEEDED

The post-retrospective inventory execution closed the remaining −9.9% in
seven experiments:

| Exp | What | Verdict | Result |
|---|---|---|---|
| v044/v045 | splash context checkpointing (kernel saves out+lse → no bwd fwd-rerun; offloaded) | **supported, frontier** | 6,715 / ≈38.6% (+7.3%) |
| v046 | fused gate+up MLP | refuted | neutral (weight-BW dominated) |
| v047/v048 | context residuals in HBM | **supported, frontier** | 6,772 / ≈38.9% (+0.85%) |
| v049 | activation sharding constraints | refuted | fired-and-useless (XLA propagation already optimal) |
| v050/v051 | **(in,out) weight layout** (transpose-free x@W; lm_head excluded) | **supported, frontier** | **6,959 / 39.9% MFU (+2.76%)** |

**Final scoreboard @ bs2/seq8192, identical Qwen3-8B architecture:**

| Stack | tok/s/chip | Step | Notes |
|---|---|---|---|
| **qwen3-jax (ours, v051-certified)** | **6,959** (median; 6,969 in the 20-step run) | 2,354 ms | 39.9% MFU (our accounting) |
| MaxText reference (mt-v001) | 6,953 | 3,535 ms (bs3) | 45.4% their accounting / ≈39.6% ours |

**The user's directive — "find a way to match its performance on jax
lane" — is complete: matched and slightly exceeded.** Day total: 20.5% →
**39.9% MFU** (+19.4 pp, 8 frontier shifts, 51 experiments). Recipe (trunk
`b991d9f`, image `latest`): bs2/seq8192 + scan + save_qkv_ctx_hbm +
SPLASH_RESIDUAL_CKPT_NAME=context + JAX_WEIGHT_LAYOUT=io + splash
(bq2048/bkv1024 fused) + tokamax chunked CE f32-x + 21-flag stack +
JAX_SCAN_UNROLL=1. On record for later: int8-AQT (numerics waiver
required), upstream XLA S(5) bug fix (re-opens unroll×offload), libtpu
updates.

## Sources

- All experiment pages above; profiles + HLO under
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v0{24..32}*/`
  and `.../2026-06-12-qwen3-maxtext-v001-base/`.

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [Bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md) · [Lane retrospective (v019 era)](2026-06-12-qwen3_cc5-jax-retrospective.md)
