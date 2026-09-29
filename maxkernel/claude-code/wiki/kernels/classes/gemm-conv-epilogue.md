---
title: "Kernel class: GEMM/Conv + epilogue chain (fusion scope, commutation, flags)"
type: concept
tags: [kernel-class, gemm, conv, epilogue, pallas, briefs-tier]
created: 2026-07-22
---

**Route here when** K1 shows a matmul/conv producer followed by pointwise activation(s) and/or reduction(s)/norm-stats — the dominant benchmark-suite shape. The strategy is decided by ONE HLO test, not by the op names.
Universal rules in [BRIEFS](../../kernel_experiments/BRIEFS.md); this page holds the class decision tree and its levers.

## The decision tree (K1, in order)
1. **Commutation check FIRST** (BRIEFS §2): any linear reduction/pooling over the GEMM output commutes onto the weight operand, eliding work by the reduction factor — the dominant legitimate win class on these suites. Check every epilogue chain before any kernel work. Implementation of a commuted pool is op-point-dependent (MXU-pool matmul when inside the epilogue — the reshape-mean form of identical algebra runs the pool as a cross-lane reduction ~1.85x slower there; reshape+mean when pre-pooling a weight offline — a full-contraction matmul just to average a few columns dominates the residual compute). Sweep both.
2. **Two applicability tests BEFORE any fusion attempt**: (a) *weight-re-read check* — fusing to keep an intermediate in VMEM can force re-reading large weight tensors across the batch/sequence axis; compute weight-bytes × reuse-factor vs intermediate-bytes saved (observed: GB-scale weights made every fused architecture memory-bound); (b) *op-point FLOP check* — absorption that shrinks memory can grow FLOPs severalfold at the wrong op-point (a prefill-shaped fusion can be decode-oriented). Fusion applicability is op-point-specific, not op-specific.
3. **The epilogue-fusion precondition test**: grep the naive HLO — is there a **separate `reduce` reading the matmul output back at a fusion boundary**? One fusion ⇒ the lever is ABSENT (single-pass reduce is already inside the matmul's kOutput fusion; a split would REINTRODUCE the round-trip) ⇒ flag-or-refute. Matmul-fusion + separate reduce ⇒ the lever is REAL.
4. **If the lever is real, fuse ONLY the epilogue** — matmul stays with XLA:
   - 2-pass / cross-element reductions (softmax, logsumexp tails): XLA cannot collapse them; a 1-pass online reduction in VMEM wins on touch-count (4→3) even when the buffer still round-trips once. Verify which mechanism via device-time attribution.
   - Multi-reduce / norm-stats chains (Sum + Group/BatchNorm stats re-reading a materialized activation): one Pallas kernel co-computing ALL the reduction stats eliminates the materialization (verified; confirm via buffer-assignment peak ≫ output size).
   - NEVER hand-write the matmul itself on a deeply compute-bound GEMM — a hand dot loses 4-5x to XLA (measured 3.4 ms vs 0.66 ms). Boundary refinement: a fused Pallas GEMM HAS beaten XLA (92.2% vs 90.2% of peak) when the fusion ALSO removes epilogue traffic — the loss cases removed nothing.
   - **LAYOUT TRAP**: a correctly-fused reduction kernel can still LOSE because operand_layout_constraints force relayout copies bigger than the savings (~3.3M cycles shuffle vs ~360K saved, observed). Check candidate HLO for `%copy` around the custom-call; the fix is a BlockSpec/grid restructure to consume XLA's natural post-matmul layout — and a refute whose candidates share one fixable layout flaw is INCONCLUSIVE.
5. **Full-fusion of GEMM→act→GEMM→reduce** beats flags only when BOTH hold: multiple kernel launches to collapse (dispatch reduction) AND the inter-matmul intermediate genuinely on the critical path (naive wall well above compute floor). Verified 1.136x (9 HLO fusions → 1 custom-call), beating both the xla-rewrite (0.98x) and the flag (~1.07x, which collapses only the FIRST round-trip).
6. **Pointwise-only chains** (no reduction): usually already fusion-complete — the honest path is the scoped-vmem probe (BRIEFS §7: 49152-class values recur as ~1.03–1.05x wins on large dense matmul + epilogue op-points) and then refute. **A bar-clearing flag is a WIN (flag-class), not a refute** (BRIEFS §5 — cite `flag_only_ratio`).
7. **"Compute-hidden" must be MEASURED** (naive-wall vs compute floor + static epilogue cycles), never assumed — on AI ≫ ridge ops, eliminating GB of overlapped traffic buys ~nothing (BRIEFS §2 overlap decomposition; three confirmations).

## Conv-specific (small channels)
- NCHW↔NHWC transpose-sandwiches are already folded by XLA layout assignment (`grep -c 'transpose(' after_optimizations` = 0); a layout-native rewrite compiles byte-identical. The residual O(tensor) `%copy` is XLA's intrinsic conv relayout — not removable.
- Below the MXU-fillable channel count the bound is the C_in/MXU_edge utilization ceiling; every hand formulation underutilizes worse (im2col ~6-9x slower; even a candidate that eliminated the ENTIRE layout-copy tax stayed ~6x slower on compute). Don't author; refute on the ceiling.

## Class-specific traps
- Workload ALGEBRA degeneracy (a self-cancelling chain outputs ≡ 0 for any input — random weights don't catch it) and zero/constant shipped weights (vacuous oracles, XLA const-fold fake wins) — BRIEFS §2/§4 probes are mandatory here; this suite class is where they were earned.
- Sub-ms op-points: wall bars measure host dispatch — dual-framing rules (BRIEFS §1) decide the verdict.

- **Flag sweep (scoped-vmem tile window)**: BRIEFS §7 probe is a required lever row on this class.
