---
title: "vhlo.md — the VHLO (Versioned StableHLO) dialect"
type: source
tags: [stablehlo, docs, vhlo, versioning, serialization, reference]
created: 2026-07-04
updated: 2026-07-04
---

The **VHLO dialect** — the mechanism behind StableHLO's compatibility guarantees. Under the reference lens: VHLO is what you'd see inside a serialized portable artifact (`vhlo.exponential_v2`, versioned ops/types/attrs); the StableHLO dialect itself only ever carries the latest op versions.

## Key points

- **[What is VHLO](src:docs/vhlo.md#what-is-the-vhlo-dialect)**: an **add-only** dialect with versioned ops/types/attributes — once added, a feature can't change semantics; any change adds a new version (`VHLO_MyOpV1` valid `0.9.0→0.10.0`, `VHLO_MyOpV2` `0.11.0→current`).
- **[Why useful](src:docs/vhlo.md#why-is-vhlo-useful)**: forward compat = convert to VHLO + downgrade ops to a target version (fails on the producer if unsupported features are used); backward compat = upgrade VHLO ops to latest, then to StableHLO. Producers only target StableHLO ops; consumers only support the latest — VHLO conversion is hidden behind serialization.
- **[MLIR bytecode versions](src:docs/vhlo.md#mlir-bytecode-format-versions)**: each StableHLO version maps to an MLIR Bytecode Format version. See [the-vhlo-dialect](src:docs/vhlo.md#the-vhlo-dialect).

## Feeds these topics

- [compatibility-and-versioning](../topics/compatibility-and-versioning.md)

## See also

- [compatibility.md](compatibility.md)
- [bytecode.md](bytecode.md)

## Sources

- `raw/code/stablehlo/docs/vhlo.md`
