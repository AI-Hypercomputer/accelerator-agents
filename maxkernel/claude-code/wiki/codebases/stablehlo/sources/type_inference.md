---
title: "type_inference.md — verifier & shape-function guidelines"
type: source
tags: [stablehlo, docs, type-inference, verifier, shape-function, reference]
created: 2026-07-04
updated: 2026-07-04
---

Contributor guidelines for StableHLO **type inference** — verifiers and shape functions. Under the reference lens: this explains where an op's shape/type constraints are enforced (ODS traits vs `verify()` vs `inferReturnTypes()`), which is why a malformed op fails at build time and why the spec's Constraints tables are authoritative for what shapes an op can produce in a dump.

## Key points

- **[(P1)](src:docs/type_inference.md#p1-use-the-stablehlo-spec-as-the-source-of-truth)** The [spec](spec.md) is the source of truth for all verifiers/shape functions; where the spec is silent, XLA's `shape_inference.cc` / `hlo_verifier.cc` fill in (they don't cover unbounded dynamism).
- **[(P2)](src:docs/type_inference.md#p2-make-the-most-of-ods)** Constraints already guaranteed by ODS traits need no verification code.
- **[(P3)](src:docs/type_inference.md#p3-maintain-verification-code-in-verifiers-and-shape-functions)** Prefer putting checks in shape functions; ops that can't infer return types (`ReshapeOp`, `BroadcastInDimOp`) or that have regions (`ReduceOp`, `IfOp`, `CaseOp`, `MapOp`) need explicit verifiers.
- **[(P4)](src:docs/type_inference.md#p4-establish-testing-guidelines)** Testing split between `ops_stablehlo.mlir` (positive + negative) and `infer_stablehlo.mlir` (shape-function existence). See [proposal](src:docs/type_inference.md#proposal) and [what-to-do](src:docs/type_inference.md#what-to-do).

## Feeds these topics

- [type-inference](../topics/type-inference.md)
- [dynamism](../topics/dynamism.md)

## Sources

- `raw/code/stablehlo/docs/type_inference.md`
