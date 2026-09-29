---
title: "compatibility.md — StableHLO compatibility guarantees"
type: source
tags: [stablehlo, docs, compatibility, versioning, serialization, reference]
created: 2026-07-04
updated: 2026-07-04
---

The **compatibility contract**: portable artifacts serialized by one libStablehlo version can be deserialized by another within a window. Under the reference lens: explains the `#.#.#` version strings and `.mlir.bc` portable artifacts you may encounter around a compiled model, and why an op you emitted may be expanded/downgraded for an older consumer.

## Key points

- **[Guarantees](src:docs/compatibility.md#guarantees)**: 5 years backward, 2 years forward compatibility for portable artifacts (semantics defined by [the spec](spec.md)).
- **[Versions](src:docs/compatibility.md#versions)**: minor bumped on opset/serialization changes, patch on downstream XLA integration.
- **[APIs](src:docs/compatibility.md#apis)**: [`stablehlo-translate --serialize/--deserialize`](src:docs/compatibility.md#stablehlo-translate), plus [C++](src:docs/compatibility.md#c) and [Python](src:docs/compatibility.md#python) `serializePortableArtifact` / `deserializePortableArtifact` with a `CompatibilityRequirement` (NONE / WEEK_4 / WEEK_12 / MAX).
- **[Tests](src:docs/compatibility.md#tests)** roundtrip a VHLO op compendium across all supported versions.
- **[Out of scope](src:docs/compatibility.md#out-of-scope)**: non-portable artifacts, unspecced features/attributes, bug-compatibility, and **numerical accuracy** (explicitly not guaranteed). See also [future-work](src:docs/compatibility.md#future-work) and [stablehlo-compatibility](src:docs/compatibility.md#stablehlo-compatibility).

## Feeds these topics

- [compatibility-and-versioning](../topics/compatibility-and-versioning.md)

## See also

- [vhlo.md](vhlo.md) — the versioned dialect that implements these guarantees
- [bytecode.md](bytecode.md) — the serialization format

## Sources

- `raw/code/stablehlo/docs/compatibility.md`
