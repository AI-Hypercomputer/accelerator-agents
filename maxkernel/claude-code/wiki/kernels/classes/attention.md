---
title: "Kernel class: attention (streaming-softmax + KV structure)"
type: concept
tags: [kernel-class, attention, pallas, briefs-tier]
created: 2026-07-22
---

**Route here when** K1 shows: a large score/logits materialization scaling with S² (or its kOutput-fused absence — check!), softmax/masking structure, KV operands with sharing (GQA/MLA) or indirection (paged/ragged). Secondary page for paged/ragged variants: [grouped-ragged-indirection](grouped-ragged-indirection.md).
Universal rules (measurement, parity, evidence, platform) are in [BRIEFS](../../kernel_experiments/BRIEFS.md) — always loaded alongside; this page holds only what is attention-specific.

## K1 signatures
- The naive's score matrix (S×S per head) either round-trips HBM (streaming lever real) or is already inside one `kOutput` fusion / collapsed by a scoped-vmem probe (BRIEFS §2/§7 — probe the flag BEFORE authoring).
- Mask structure decides work-skipping: causal ⇒ sub-triangular tiles only; block-sparse ⇒ grid folding.
- MXU operand-feed bandwidth (BRIEFS §2 fourth floor) binds KV-streaming decode before the HBM floor.
- After streaming lands, MXU idle correlated with exp phases in the LLO fit = the VPU transcendental wall (BRIEFS §2) → route to const-shift + compute pipelining, not another tile sweep.

## Utilization yardstick (when are you done?)

Compute the op-point's **useful**-FLOP roofline at K1 (causal halves useful FLOPs — count only
live blocks) and re-state frontier utilization against it at every K7. Class experience:
streaming/online softmax alone typically lands near ~15–20% of peak — that is the STARTING
point, not a result; +const-shift/split-loop reaches ~45–50%; +compute pipelining, trace-level
diagonal skip, and keeping all MXUs fed, 85%+ is achievable on MXU-bound causal shapes.
**Below ~70% util with any of those three untried, "at ceiling" is not a claimable verdict** —
pick the next lever instead. Decode/KV-streaming shapes (q_len ≈ 1) are bandwidth-bound: the
yardstick there is achieved bytes/s vs the K1 bandwidth floor — MXU utilization is the wrong
denominator.

## Levers (verified)
- **Streaming / online softmax** — the core move: tile the KV axis, keep running max/sum in VMEM, never materialize the score matrix. Accumulate fp32. This is the class's largest lever and the one cold runs historically miss.
- **CONST-SHIFT SOFTMAX** (3-arm validated): replace running-max with a constant shift after MEASURING the overflow margin (max |QK| vs fp32 range) — pre-register the margin; running-max is the general fallback.
- **Causal diagonal subtiling ≥ the MXU tile edge**: on the diagonal block, subtile pairs with key_subtile > query_subtile are fully masked — never trace them (Python-level loop, not runtime cond); large verified matmul-work skip. Computing masked subtiles and zeroing them via predicates is NOT this lever — the subtile must never be emitted; verify by counting emitted matmuls in the lowered kernel, not by reading your source. NEVER subtile below the MXU edge (256 on v6e) — pipeline-fill bubbles collapse it ~3.5x.
- **Split-loop / unmasked-prefix** (catalog class 1): mask-free hot loop over fully-visible blocks + one explicit masked diagonal step; verify by hot-loop op census (cmpi/select/iota = 0) + no-skip coverage probes.
- **Grid folding via scalar prefetch** (catalog class 2): enumerate only live blocks (causal/block-sparse) via prefetched index arrays; wins scale with the dead-traffic fraction.
- **Heads-per-program is OP-POINT-DEPENDENT — sweep hp ∈ {1,2,4}, don't assume**: a lowering failure on one route (e.g. vmap-over-heads gather → NotImplementedError) does not rule this out — head-grouping via the grid dimension and batch-in-block are the standard alternatives; rule out only after a grid-dim attempt. Large per-head VMEM footprints lose (forces smaller sequence blocks; prefer batch-in-block / head-in-grid); small footprints amortize per-program overhead and have won repeatedly. Both directions verified; the footprint decides.
- **Group-shared operands (GQA/MLA)**: load the shared K/V once per group (group rides inside the block — one K/V DMA per group, measured superior to heads-in-grid on multiple families). First try the algebraic route: folding a `repeat` into the contraction is an xla-rewrite, not a kernel.
- **Latent/low-rank absorption — shape-conditional; decide by FLOP algebra**: absorbing K/V up-projections into the attention contractions moves the score contraction from head dim to latent dim. FLOP trade: saves expansion (∝ S_kv·latent·d_head), adds score work (∝ S_q·S_kv·(latent−d_qk)) — wins iff S_q·(latent−d_qk) < latent·d_head (decode/small-S_q; FLOP-negative at prefill scale — a legitimate mechanism rule-out, measured both ways). The cheaper formulation's FLOP count is the roofline comparator. Must land INSIDE the streaming kernel — a standalone graph rewrite materializes the S² scores (route failure, not lever refutation). Contraction dims change either way: re-check tile alignment per generation; composite head dims (nope+rope) pad to the tile edge — split per-component contractions on the direct route.
- **Shared-operand expansion-in-kernel** *(shape-conditional — measured)*: traffic-side complement where absorption is FLOP-negative — many heads share one compressed K/V operand ⇒ load it once per block, expand in VMEM; FLOPs unchanged, expansion round-trips eliminated. Measured at prefill-scale S_q: fires exactly, parity-clean, but LOSES to a tuned direct kernel (expansion compute displaces MXU time the direct path overlaps). Decode/small-S_q remains its candidate regime — same shape-condition family as absorption.
- **Compact causal grid (live-pair enumeration)**: replace the rectangular (q_block, kv_block) grid with a flat live-pair grid — a scalar-prefetched, statically row-grouped live-pair table drives every BlockSpec index map, so masked upper-triangular programs are never launched at all (grid COMPACTION vs grid folding's index redirection). Verified as a further step on top of split-loop + pipelining. *(worker-discovered, receipt-verified, adopted 2026-07-24)*
- **Per-item software pipelining** when the wall is a serial MXU→VPU→MXU dependency chain (verified −24% on a paged-class kernel) — here that chain is QK^T→exp→PV; Pallas auto-double-buffering covers DMA only and never rules this out (BRIEFS §7); **cross-item prefetch at grid boundaries** (program i issues item i+1's first DMA during its tail, slot parity in SMEM — verified −12%/−7%).

## Class-specific parity traps
- **Phantom/padded rows**: parity on padded/sub-slot regions can be structurally wrong and inherit bit-exactly across candidates — probe kv_len < slot inputs explicitly (BRIEFS §4 has the general rule; it was earned here).
- **Degenerate masks are READ, not modeled**: an all-true mask makes the baseline do something specific (measured: uniform average over the full read pool) — replicate the measured behavior.
- **bq == bk validity**: block-index causal skip `j <= i` is valid ONLY when bq == bk; position-aware predicates survive bq != bk. A sweep winner needs its own parity run at exactly its config (BRIEFS §3 Sweeps).

## Refute honestly when
- The op-point sits at the dispatch floor (single-token decode, tiny per-call work) — the kernel loses to XLA's fused path regardless of inner-loop quality.
- A scoped-vmem probe already collapsed the materialization (a flag win is a WIN — file flag-class, not refute).
