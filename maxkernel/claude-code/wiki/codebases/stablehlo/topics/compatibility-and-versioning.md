---
title: "compatibility & versioning — VHLO, bytecode, portable artifacts"
type: concept
tags: [stablehlo, compatibility, versioning, vhlo, bytecode, serialization, reference]
created: 2026-07-04
updated: 2026-07-04
---

**Compatibility & versioning** — how a StableHLO program is serialized to a stable, forward/backward-compatible artifact. Under the reference lens this is peripheral to perf, but it explains the `#.#.#` version strings, `.mlir.bc` files, and `vhlo.*` ops you may encounter around a compiled/exported model, and why an op you emitted can be expanded or downgraded for a different consumer.

## The guarantee

Portable artifacts carry **5 years backward / 2 years forward** compatibility ([guarantees](src:docs/compatibility.md#guarantees)); semantics are defined by [spec.md](../sources/spec.md). The current version lives in `Version.h`; minor bumps on opset/serialization changes, patch on XLA integration ([versions](src:docs/compatibility.md#versions)). Explicitly **out of scope**: non-portable artifacts, unspecced attributes, and **numerical accuracy** ([out-of-scope](src:docs/compatibility.md#out-of-scope)).

## The three layers

1. **Portable-artifact APIs** — [`stablehlo-translate --serialize/--deserialize`](src:docs/compatibility.md#stablehlo-translate), or C++/Python `serializePortableArtifact(module, targetVersion)` / `deserializePortableArtifact` with a `CompatibilityRequirement` (NONE / WEEK_4 / WEEK_12 / MAX) ([apis](src:docs/compatibility.md#apis)).
2. **VHLO (Versioned StableHLO) dialect** — the mechanism. An **add-only** dialect with versioned ops/types/attrs; each op carries an inclusive version range (`VHLO_MyOpV1` `0.9.0→0.10.0`, `VHLO_MyOpV2` `0.11.0→current`) ([what-is-the-vhlo-dialect](src:docs/vhlo.md#what-is-the-vhlo-dialect)). Forward compat = convert to VHLO + **downgrade** ops to a target version (fails on the producer if unsupported); backward compat = **upgrade** VHLO ops to latest, then to StableHLO ([why-is-vhlo-useful](src:docs/vhlo.md#why-is-vhlo-useful)). Producers only target StableHLO; consumers only support the latest — VHLO conversion hides behind serialization.
3. **MLIR Bytecode Format** — the on-disk encoding. Each StableHLO version maps to an [MLIR bytecode version](src:docs/vhlo.md#mlir-bytecode-format-versions); StableHLO layers its guarantees on top of the [MLIR Bytecode Format](src:docs/bytecode.md#mlir-bytecode-format).

## The passes

[`-stablehlo-legalize-to-vhlo`](optimization-passes.md) / `-vhlo-legalize-to-stablehlo` / `-vhlo-to-version` implement the conversions; [`-stablehlo-compatibility-expander`](../sources/generated-stablehlo_passes.md) opt-in decomposes newer ops into older-version-equivalents (e.g. `tan → sine/cosine/divide`) to maximize compatibility at the cost of possibly-less-optimal compilation.

> [!inferred]
> Relevance to the loop: `vhlo.*` ops only appear inside serialized artifacts, not in a normal TPU HLO dump. But the "no numerical-accuracy guarantee" clause is load-bearing — it's the spec's blessing for backends (and versions) to differ numerically, which is exactly why the autoresearch loop must run its own semantics/parity check rather than trusting cross-version bit-equality.

## See also

- [optimization-passes](optimization-passes.md) (the legalize-to/from-vhlo passes), [quantization](quantization.md) (also accuracy-unspecified)
- Sources: [compatibility.md](../sources/compatibility.md), [vhlo.md](../sources/vhlo.md), [bytecode.md](../sources/bytecode.md)

## Sources

- `raw/code/stablehlo/docs/compatibility.md`
- `raw/code/stablehlo/docs/vhlo.md`
- `raw/code/stablehlo/docs/bytecode.md`
