---
slug: jaxite
commit: a2a5fe0b0eaba43c6280bb00e76f9f7ee4f9be24
scip_tool: scip-python
updated: 2026-07-03
---

# jaxite internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| jaxite-jaxite_bool-bool_params | [jaxite-jaxite_bool-bool_params](concepts/jaxite-jaxite_bool-bool_params.md) | fresh |
| jaxite-jaxite_bool | [jaxite-jaxite_bool](concepts/jaxite-jaxite_bool.md) | fresh |
| jaxite-jaxite_cggi-bootstrap | [jaxite-jaxite_cggi-bootstrap](concepts/jaxite-jaxite_cggi-bootstrap.md) | fresh |
| jaxite-jaxite_cggi-decomposition | [jaxite-jaxite_cggi-decomposition](concepts/jaxite-jaxite_cggi-decomposition.md) | fresh |
| jaxite-jaxite_cggi-encoding | [jaxite-jaxite_cggi-encoding](concepts/jaxite-jaxite_cggi-encoding.md) | fresh |
| jaxite-jaxite_cggi-lwe | [jaxite-jaxite_cggi-lwe](concepts/jaxite-jaxite_cggi-lwe.md) | fresh |
| jaxite-jaxite_cggi-parameters | [jaxite-jaxite_cggi-parameters](concepts/jaxite-jaxite_cggi-parameters.md) | fresh |
| jaxite-jaxite_cggi-random_source | [jaxite-jaxite_cggi-random_source](concepts/jaxite-jaxite_cggi-random_source.md) | fresh |
| jaxite-jaxite_cggi-rgsw | [jaxite-jaxite_cggi-rgsw](concepts/jaxite-jaxite_cggi-rgsw.md) | fresh |
| jaxite-jaxite_cggi-rlwe | [jaxite-jaxite_cggi-rlwe](concepts/jaxite-jaxite_cggi-rlwe.md) | fresh |
| jaxite-jaxite_cggi-types | [jaxite-jaxite_cggi-types](concepts/jaxite-jaxite_cggi-types.md) | fresh |
| jaxite-jaxite_ckks-encode | [jaxite-jaxite_ckks-encode](concepts/jaxite-jaxite_ckks-encode.md) | fresh |
| jaxite-jaxite_ckks-encrypt | [jaxite-jaxite_ckks-encrypt](concepts/jaxite-jaxite_ckks-encrypt.md) | fresh |
| jaxite-jaxite_ckks-mul | [jaxite-jaxite_ckks-mul](concepts/jaxite-jaxite_ckks-mul.md) | fresh |
| jaxite-jaxite_ckks-ntt | [jaxite-jaxite_ckks-ntt](concepts/jaxite-jaxite_ckks-ntt.md) | fresh |
| jaxite-jaxite_ckks-rescale | [jaxite-jaxite_ckks-rescale](concepts/jaxite-jaxite_ckks-rescale.md) | fresh |
| jaxite-jaxite_ckks-rns | [jaxite-jaxite_ckks-rns](concepts/jaxite-jaxite_ckks-rns.md) | fresh |
| jaxite-jaxite_ckks-types | [jaxite-jaxite_ckks-types](concepts/jaxite-jaxite_ckks-types.md) | fresh |
| jaxite_ec-algorithm-elliptic_curve | [jaxite_ec-algorithm-elliptic_curve](concepts/jaxite_ec-algorithm-elliptic_curve.md) | fresh |
| jaxite_ec-algorithm-finite_field | [jaxite_ec-algorithm-finite_field](concepts/jaxite_ec-algorithm-finite_field.md) | fresh |
| jaxite_ec-elliptic_curve_test | [jaxite_ec-elliptic_curve_test](concepts/jaxite_ec-elliptic_curve_test.md) | fresh |
| jaxite_ec-pippenger | [jaxite_ec-pippenger](concepts/jaxite_ec-pippenger.md) | fresh |
| jaxite_ec-pippenger_rns | [jaxite_ec-pippenger_rns](concepts/jaxite_ec-pippenger_rns.md) | fresh |
| jaxite_ec-util | [jaxite_ec-util](concepts/jaxite_ec-util.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **1804** across 89 modules
- deep (concept pages): **375** (20.8%)
- catalog-only: **1429**
- represented total: **1804** (100.0%)
- classes represented: **141/141**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
