---
title: "type inference — verifiers and shape functions"
type: concept
tags: [stablehlo, type-inference, verifier, shape-function, reference]
created: 2026-07-04
updated: 2026-07-04
---

**Type inference** — how StableHLO computes and verifies op result types/shapes. Under the reference lens this is background rather than a perf knob, but it explains *why the spec's Constraints tables are authoritative*: those constraints are enforced by ODS traits, verifiers, and shape functions, so a shape in a dump is guaranteed consistent with them, and a malformed op fails at build time rather than producing a bad dump.

## Where constraints live

Per [type-inference](src:docs/type_inference.md#type-inference) and the [proposal](src:docs/type_inference.md#proposal):

- **[(P1) Spec is source of truth](src:docs/type_inference.md#p1-use-the-stablehlo-spec-as-the-source-of-truth)** — verifiers/shape functions must match [spec.md](../sources/spec.md); where the spec is silent, XLA's `shape_inference.cc` / `hlo_verifier.cc` govern (neither covers unbounded [dynamism](dynamism.md)).
- **[(P2) ODS traits](src:docs/type_inference.md#p2-make-the-most-of-ods)** — constraints guaranteed by ODS (e.g. `SameOperandsAndResultShape`) need no extra code.
- **[(P3) Split](src:docs/type_inference.md#p3-maintain-verification-code-in-verifiers-and-shape-functions)** — `Op::verify()` vs `Op::inferReturnTypes()`. Most region-free ops put logic in shape functions; ops that can't infer return types (`ReshapeOp`, `BroadcastInDimOp`) or have regions (`ReduceOp`, `IfOp`, `CaseOp`, `MapOp`) need explicit verifiers.
- **[(P4) Testing](src:docs/type_inference.md#p4-establish-testing-guidelines)** — `ops_stablehlo.mlir` (positive+negative) and `infer_stablehlo.mlir` (shape-function existence). See [what-to-do](src:docs/type_inference.md#what-to-do).

> [!inferred]
> For the loop, the practical takeaway: the Inputs/Outputs/Constraints tables on each op in [spec.md](../sources/spec.md) are not aspirational — they're mechanically verified, so you can trust them when reasoning about the shapes and dtypes an op in a dump must have. Ops with regions (reduce, scatter, if/case) are exactly the ones whose result types depend on an inner computation.

## See also

- [dynamism](dynamism.md) (shape refinement builds on inference), [dot_general](dot_general.md) / [reduce-and-windowing](reduce-and-windowing.md) (constraint-heavy ops)
- Source: [type_inference.md](../sources/type_inference.md)

## Sources

- `raw/code/stablehlo/docs/type_inference.md`
