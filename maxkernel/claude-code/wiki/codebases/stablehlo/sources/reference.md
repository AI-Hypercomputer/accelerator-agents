---
title: "reference.md — StableHLO reference interpreter design"
type: source
tags: [stablehlo, docs, interpreter, reference]
created: 2026-07-04
updated: 2026-07-04
---

Design of the **reference interpreter** — the executable ground-truth for op semantics. Under the reference lens: the interpreter is the authority that pins down what each op in a dump *actually computes*, independent of the TPU backend.

## Key points

- **[Data model](src:docs/reference.md#data-model)**: tensors are `Tensor` objects over major-to-minor contiguous buffers; elements are `APInt`/`APFloat`.
- **[How it works](src:docs/reference.md#how-the-interpreter-works)**: `eval(func, args)` walks ops in SSACFG order, dispatching a per-op `eval` (e.g. `AddOp`) that implements the spec semantics. See [interpreter-design](src:docs/reference.md#interpreter-design).
- **[Constant folding](src:docs/reference.md#using-the-interpreter-for-constant-folding)**, and [testing guidelines](src:docs/reference.md#testing-guidelines) / [testing the interpreter](src:docs/reference.md#testing-the-stablehlo-interpreter).

## See also

- [spec.md](spec.md) — the semantics the interpreter implements
- [interpreter_status.md](interpreter_status.md) — coverage

## Sources

- `raw/code/stablehlo/docs/reference.md`
