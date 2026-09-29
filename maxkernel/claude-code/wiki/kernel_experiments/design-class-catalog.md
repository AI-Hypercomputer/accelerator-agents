---
title: "Pallas kernel design-class catalog — problem-invariant mechanisms"
type: concept
tags: [pallas, kernel-optimization, design-classes, mechanisms, tpu]
created: 2026-07-15
updated: 2026-07-15
---

The generative core distilled from multi-arm kernel campaigns: mechanism classes that transfer to NEW kernels, phrased problem-invariantly, each with the physics of when it applies and verified evidence. This page (with BRIEFS.md's trap catalog) is what an optimization run reads FIRST. Cardinal rule for reuse: the mechanism and its applicability test are the knowledge — the specific block sizes, flags, and ratios are op-point-bound and must be re-derived.

## Compute-shape classes

### 1. Split-loop / unmasked-prefix (masked iteration structures)
**Mechanism**: when a loop applies a mask that is trivial for most iterations (e.g. causal attention: fully-visible prefix blocks + one partial diagonal block), split it — a mask-free hot loop over the trivial region plus one explicit masked step. Removes per-iteration `iota/cmpi/select` from the hot path.
**Applies when**: mask is block-structured and degenerate on most blocks. **Doesn't**: fine-grained irregular masks.
**Evidence**: verified +15% over the masked loop (Mosaic-confirmed 0 mask ops in the hot body); independently load-bearing in two more attention-class kernels.
**Verify by**: hot-loop op census in Mosaic IR (cmpi/select/iota = 0) + no-skip coverage probes (a split loop can silently drop or double-count a block — probe every sub-range).

### 2. Grid folding via scalar prefetch (dead-iteration elimination)
**Mechanism**: when a fraction of grid steps does no useful work but still DMAs its windows (`pl.when` gates compute, NOT pipeline copies), enumerate only the live blocks via prefetched index arrays (`PrefetchScalarGridSpec`) and index block maps through them. Grid shrinks (e.g. S²→lower-triangle N(N+1)/2); dead DMA disappears.
**Applies when**: liveness is known at launch (causal/block-sparse patterns). **Physics**: wins scale with the dead-traffic fraction (observed: 6/16 grid steps dead ⇒ ~+12%).
**Evidence**: verified +11.9% from shrinking the grid to live blocks only (bandwidth win, not FLOPs).
**Verify by**: `iteration_bounds` + `scalar_prefetch` args in Mosaic; coverage probe on every remaining block.

### 3. Layout-native outputs (transpose-sandwich elimination)
**Mechanism**: emit the kernel's output directly in the layout its consumer needs (choose grid/BlockSpec so the written tiles land in consumer order) instead of writing flat and letting XLA reshape/transpose. Kills large intermediate materializations.
**Applies when**: a reshape→transpose chain of O(tensor-size) sits between two ops in HLO. **Cost**: usually free — it is an index-map change.
**Evidence**: verified +35% from emitting the consumer-native output layout (~600MB×2 materialization removed; parity bit-identical to the prior kernel proves pure-layout).
**Verify by**: custom_call output shape in HLO + bitcast/copy count drop; bit-identical parity vs the pre-change kernel proves pure-layout.

### 4. Fusion-scope selection (what to put inside one kernel)
**Mechanism**: fuse ops that share large intermediates (gate+up projections; norm+projection epilogues); do NOT fuse across an intermediate that is already compute-hidden.
**Physics test before attempting**: is the traffic you would eliminate actually on the critical path? (Observed twice: removing 2.1GB of logits traffic bought only 6.6% — hidden under the matmul; a direct-consumption contraction bought 0% for the same reason.)
**Evidence**: verified epilogue fusion +3–6%; a gate+up projection fusion lineage; the two null results above are equally load-bearing.
**Two more applicability tests**: (a) *weight-re-read check*: fusing to keep an intermediate in VMEM can force re-reading large weight tensors across the batch/sequence axis — compute weight-bytes × reuse-factor vs intermediate-bytes saved before attempting (observed: GB-scale weights made every fused architecture memory-bound). (b) *op-point FLOP check*: absorption that shrinks memory can grow FLOPs — a projection fusion at a prefill op-point can cost several times the core FLOPs (materialized length ≫ compressed reps) while the same fusion is decode-oriented. Fusion applicability is op-point-specific, not op-specific.

## Numerics classes

### 5. Minimal-fp32-tail parity engineering
**Mechanism**: when a parity gate fails under full-bf16, place fp32 only where the error is created (accumulators, final projection, softmax stats) — not everywhere. Find the site by binary-searching the dtype boundary against the gate.
**Anti-pattern (measured twice)**: dtype conversions have first-class cost — a bf16 exp that saves VPU exp cycles can lose to its own f32↔bf16 cast round-trip over big tiles (observed: −7.8% despite −44% exp ops); Kahan does nothing when the error originates in multiplication.
**Evidence**: an fp32 down-projection tail passed the gate at zero cost; a separate case proved the defect intrinsic to the factorization (wide-dynamic-range intermediates forced to bf16 pre-MXU — no accumulator trick can fix that class).
**Verify by**: element-type audit of each dot/reduce in HLO; parity vs the reference's own noise floor, never absolute zero.

### 6. Physics ceilings — recognize saturation and stop
**Mechanism**: compute the binding roofline FIRST (bytes-moved for streaming ops; MXU utilization of the dense-equivalent for grouped ops). If the incumbent is at it, further kernel work is waste; file the tie.
**Evidence**: a single-pass streaming op at ~68% of HBM peak — three independent arms tied within 1%; a uniform-group GMM class where the dense-equivalent einsum runs ~68% MXU and nothing ragged beat it (even the library ragged primitive was severalfold slower). >1.5x-of-roofline results are measurement-suspect by rule.

### 7. The exclusion-chain method (proving a local optimum)
**Mechanism**: when refuting candidates, structure the sequence so each refutation eliminates a resource hypothesis with mechanism evidence: restrict operand A's traffic (if ~0% wall change, A wasn't binding) → cut MXU issue ops (if wall-flat, not MXU-bound) → attack the remaining unit (if the lever loses to its own overhead, the incumbent is locally optimal). The product is a *proof*, reusable across the family, of why the frontier is where it is.
**Evidence**: one full verified chain (operand-DMA restriction ~1% → wall-flat at −25% MXU issue ops → VPU-bound → the cast tax beats the exp savings ⇒ the incumbent is the local optimum).

## Configuration classes

### 8. Scoped-VMEM sweep — both directions
**Mechanism**: `--xla_tpu_scoped_vmem_limit_kib` changes XLA's tiling/double-buffering schedule; the optimum is frequently BELOW the value that seems generous, and the curve is non-monotonic. Sweep {default, 49152, 65536, 114688, 131072, 262144}-class values with fresh processes per value and per-value compile caches.
**Evidence**: one op-point's optimum sat mid-range with larger values worse; the ranking was decided only by a low-jitter re-measure (σ 0.01 ms vs the author's 0.35 ms).
**Verify by**: HLO backend_config diff (window/iteration bounds + estimated_cycles change ⇒ flag consumed; zero diff + zero delta ⇒ inert flag).

### 9. Block-size / tile heuristics are not answers
Optimal blocks are op-point-, VMEM-, and layout-bound: one family's ceiling was 1024×1024 (2048 OOMed); another re-tuned its query block after a layout change; a library's default tile cost 10x. Re-derive per op-point via bounded sweep; record the VMEM wall in the ledger. A library's/default's tile choice is a hypothesis, not a fact (one audited library shipped TODO-stub heuristics and an empty autotune space — check the tuner has a search space before trusting "autotuned").

## Measurement invariants (preconditions for any of the above being real)
Co-measure every leg in one process, interleaved, both orderings; only ratios transfer across rigs/chips; fresh naive per filing; parity judged against the reference's own noise floor at the full claimed shape, with degeneracy + sensitivity probes; mechanism confirmed structurally (HLO/Mosaic) — a speedup whose mechanism can't be found in IR is an artifact until proven otherwise. These are BRIEFS §1/§4/§5; campaign forensics showed they are not self-enforcing — budget for independent verification.

## See also
- [BRIEFS.md](BRIEFS.md) — the full trap catalog + binding discipline rules
