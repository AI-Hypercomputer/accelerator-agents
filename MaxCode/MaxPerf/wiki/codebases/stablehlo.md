---
title: "StableHLO"
type: codebase
tags: [compiler, mlir, ir, hlo, reference]
commit: ce5d23016461f3c47f92519aa79e27a18ceea4ab
created: 2026-04-22
updated: 2026-04-22
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->


StableHLO is the MLIR dialect and operation set that acts as a portability layer between ML frameworks (JAX, PyTorch, TensorFlow) and ML compilers (XLA, IREE). For this wiki it is primarily a **reference** for interpreting HLO op names, semantics, and compiler-pass vocabulary when reading XLA dumps and xprof traces — not an optimization target. We do not modify StableHLO.

## Overview

StableHLO evolved from the MHLO dialect and adds serialization (MLIR bytecode) plus forward/backward compatibility guarantees. The repo contains: the op-set specification, an MLIR C++ implementation of the `stablehlo`, `chlo`, and `vhlo` dialects, a reference interpreter, and a collection of transformation passes (legalizations, canonicalizations, folders, simplifiers). In a typical TPU pipeline, frameworks lower to StableHLO, then XLA consumes it and lowers through HLO -> LLO -> TPU. When inspecting an `xla_dump` or profile, the op names (`stablehlo.dot_general`, `stablehlo.convolution`, `stablehlo.reduce`, `stablehlo.all_gather`, ...) and their semantics come from this repo's spec.

## Architecture

Top-level layout of `raw/code/stablehlo/`:

- `stablehlo/dialect/` — C++ and TableGen definitions for the three dialects: `StablehloOps.td` / `.cpp` (the main op-set), `ChloOps.td` (high-level "client" ops that decompose to StableHLO), `VhloOps.td` / `VhloDialect.td` (versioned HLO for serialization/compatibility).
- `stablehlo/transforms/` — MLIR passes: legalizations between dialects, shape refinement, compatibility expansion, quant/QDQ lowering.
- `stablehlo/transforms/optimization/` — target-independent optimization passes (aggressive folder, aggressive simplification).
- `stablehlo/reference/` — reference interpreter used for spec verification.
- `stablehlo/conversions/`, `stablehlo/integrations/` — lowerings to Linalg, TOSA, and Python/TF bindings.
- `stablehlo/tools/`, `stablehlo/tests/`, `stablehlo/testdata/` — CLI tools and lit tests.
- `docs/` — specification and generated pass documentation (not ingested in later waves).

## Key abstractions

- **StableHLO op-set** — the ~100 ops defined in [stablehlo/dialect/StablehloOps.td](../../raw/code/stablehlo/stablehlo/dialect/StablehloOps.td) whose semantics are frozen by the spec. These are the names that show up in HLO dumps.
- **CHLO (Client HLO)** — higher-level convenience ops in [stablehlo/dialect/ChloOps.td](../../raw/code/stablehlo/stablehlo/dialect/ChloOps.td) that decompose to StableHLO via `chlo-legalize-to-stablehlo`.
- **VHLO (Versioned HLO)** — a stable, versioned mirror used for serialization; enables forward/backward compatibility across compiler/framework versions.
- **Passes** — MLIR transformation passes registered in [stablehlo/transforms/Passes.td](../../raw/code/stablehlo/stablehlo/transforms/Passes.td) and [stablehlo/transforms/optimization/Passes.td](../../raw/code/stablehlo/stablehlo/transforms/optimization/Passes.td).
- **Spec** — [docs/spec.md](../../raw/code/stablehlo/docs/spec.md) is the authoritative op-by-op semantic reference.

## Entry points

Not a primary build target for this wiki, but for completeness:

- `stablehlo-opt` — lit/CLI driver for running passes (`stablehlo/tools/`).
- `stablehlo-translate` — serialization round-trip tool.
- Python bindings: `import mlir.dialects.stablehlo` after a Python-enabled build.

XLA and IREE link StableHLO as a library; end users rarely invoke this repo directly.

## Dependencies

- LLVM / MLIR (pinned in `build_tools/llvm_version.txt`).
- Standard MLIR infrastructure (TableGen, bytecode, Shape dialect).

## Notable files

- [stablehlo/dialect/StablehloOps.td](../../raw/code/stablehlo/stablehlo/dialect/StablehloOps.td) — op definitions (the source of truth for op names seen in HLO dumps).
- [stablehlo/dialect/ChloOps.td](../../raw/code/stablehlo/stablehlo/dialect/ChloOps.td) — CHLO op definitions.
- [stablehlo/dialect/VhloDialect.td](../../raw/code/stablehlo/stablehlo/dialect/VhloDialect.td) — version log for the dialect.
- [stablehlo/transforms/Passes.td](../../raw/code/stablehlo/stablehlo/transforms/Passes.td) — pass registry (canonicalize dynamism, shape refinement, legalizations).
- [stablehlo/transforms/optimization/Passes.td](../../raw/code/stablehlo/stablehlo/transforms/optimization/Passes.td) — `stablehlo-aggressive-folder`, `stablehlo-aggressive-simplification`, `stablehlo-target-independent-optimization`.
- [docs/spec.md](../../raw/code/stablehlo/docs/spec.md) — full op semantics.

## Performance-relevant surfaces

Intentionally minimal — this codebase is a reference, not a knob-bearing target.

- **Op-set semantics** — [docs/spec.md](../../raw/code/stablehlo/docs/spec.md). Use when an HLO dump shows an op whose meaning you need to confirm (e.g., `stablehlo.dot_general` contraction/batch dim conventions, `stablehlo.scatter` update semantics, collective ops' replica-group layout).
- **Pass catalog** — [docs/generated/stablehlo_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_passes.md) and [docs/generated/stablehlo_optimization_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md). Useful when an HLO stage name in an XLA dump corresponds to one of these passes (e.g., `stablehlo-aggressive-simplification`, `stablehlo-canonicalize-dynamism`).
- **C++ dialect implementation** — [stablehlo/dialect/](../../raw/code/stablehlo/stablehlo/dialect/) and [stablehlo/transforms/](../../raw/code/stablehlo/stablehlo/transforms/), kept at a high level. Drill in only if an XLA-side behavior points back to specific folding/verification logic here.

No StableHLO-side flags are expected to land in hypotheses or experiments in this wiki; optimization knobs live in XLA and the frameworks that emit StableHLO.

## Connections

- Serves as a reference for any future HLO-dump observations or `concepts/` pages (e.g., fusion, layout, collectives) that cite StableHLO op names.
- Upstream producers: JAX, PyTorch/XLA, TensorFlow.
- Downstream consumers: XLA (the main TPU compiler), IREE.
- `docs/` will not be ingested in later waves — this page is the single stop for StableHLO context.

## See also

- (none yet — no concept or observation pages reference StableHLO ops at ingest time)

## Sources

- [raw/code/stablehlo/README.md](../../raw/code/stablehlo/README.md)
- [raw/code/stablehlo/docs/spec.md](../../raw/code/stablehlo/docs/spec.md)
- [raw/code/stablehlo/docs/generated/stablehlo_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_passes.md)
- [raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md)
- [raw/code/stablehlo/stablehlo/dialect/](../../raw/code/stablehlo/stablehlo/dialect/)
- [raw/code/stablehlo/stablehlo/transforms/](../../raw/code/stablehlo/stablehlo/transforms/)
