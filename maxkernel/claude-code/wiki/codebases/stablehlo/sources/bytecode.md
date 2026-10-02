---
title: "bytecode.md — StableHLO bytecode / MLIR Bytecode Format"
type: source
tags: [stablehlo, docs, bytecode, serialization, reference]
created: 2026-07-04
updated: 2026-07-04
---

StableHLO's on-disk **serialization format**. Under the reference lens: portable artifacts (`.mlir.bc`) you might find alongside a compiled model are MLIR bytecode.

## Key points

- **[MLIR bytecode format](src:docs/bytecode.md#mlir-bytecode-format)**: StableHLO serializes with the [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/), chosen for serialization speed/size, mmap, and easier versioning.
- Bytecode wasn't built to make MLIR stable, but StableHLO layers its compatibility guarantees on top of it (see [bytecode.md](src:docs/bytecode.md#stablehlo-bytecode) and [compatibility.md](compatibility.md)).

## Feeds these topics

- [compatibility-and-versioning](../topics/compatibility-and-versioning.md)

## See also

- [compatibility.md](compatibility.md)
- [vhlo.md](vhlo.md)

## Sources

- `raw/code/stablehlo/docs/bytecode.md`
