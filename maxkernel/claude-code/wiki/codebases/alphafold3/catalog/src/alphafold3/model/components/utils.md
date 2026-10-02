---
title: 'Module: src/alphafold3/model/components/utils.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/components/utils.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.components.utils`/
symbols:
  bfloat16_context: bfloat16_context().
  remove_invalidly_typed_feats: remove_invalidly_typed_feats().
  mask_mean: mask_mean().
  VALID_DTYPES: VALID_DTYPES.
  bfloat16_getter: bfloat16_getter().
---
# Module: [`src/alphafold3/model/components/utils.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/utils.py)

## Functions
- `bfloat16_context()` — [`L55`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/utils.py#L55) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
- `bfloat16_getter(next_getter, value, context)` — [`L46`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/utils.py#L46) — Ensures that a bfloat16 parameter is provided by casting if necessary.
- `mask_mean(mask: jnp.ndarray, value: jnp.ndarray, axis: int | Sequence[int] | None = None, keepdims: bool = False, eps: float = 1e-10)` — [`L60`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/utils.py#L60) — Masked mean.
- `remove_invalidly_typed_feats(batch: features.BatchDict)` — [`L35`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/utils.py#L35) — Remove features of types we don't want to send to the TPU e.g. strings.

## Module values
- `VALID_DTYPES` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/utils.py#L32)

