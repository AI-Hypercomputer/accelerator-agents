---
title: 'Module: qwix/contrib/sparsity/sparsity_module.py'
type: catalog
provenance: extracted
module: qwix/contrib/sparsity/sparsity_module.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.sparsity.sparsity_module`/SparsityModule#
symbols:
  SparsityModule.__call__: __call__().
  SparsityModule.sparsity_rule: sparsity_rule.
  SparsityModule.mask_update: mask_update().
  SparsityModule._maybe_update_mask: _maybe_update_mask().
  SparsityModule.should_update_mask: should_update_mask().
  SparsityModule.step: step.
  SparsityModule.mask: mask.
  SparsityModule: ''
  SparsityModule.__init__: __init__().
  SparsityModule.no_mask_update: no_mask_update().
---
# Module: [`qwix/contrib/sparsity/sparsity_module.py`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py)

## Classes
### `SparsityModule`  ·  implements/extends Module
- def: [`qwix/contrib/sparsity/sparsity_module.py:22`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L22)
- doc: A stateful module for managing and applying structured sparsity in Flax NNX.
- signature: `class SparsityModule(nnx.Module):`
- members:
  - `_maybe_update_mask(self, weight: jax.Array, step: jax.Array)` — [`L55`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L55) — Updates the sparsity mask based on the current step and config.
  - `mask_update(w: jax.Array, mask_val: jax.Array)` — [`L65`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L65) — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
  - `no_mask_update(w, mask_val)` — [`L77`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L77)
  - `should_update_mask(step: jax.Array)` — [`L80`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L80)
  - `mask` — [`L41`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L41)
  - `sparsity_rule` — [`L49`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L49)
  - `step` — [`L40`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L40)
- protocol/private: `__call__`[`L103`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L103), `__init__`[`L43`](../../../../../../../raw/code/qwix/qwix/contrib/sparsity/sparsity_module.py#L43)
- uses (calls/refs, reference-scoped): [`get_sparsity_mask`](../../_src/core/sparsity.md#get_sparsity_mask), [`SparsityRule`](../../_src/core/sparsity.md#SparsityRule), [`weight_sparsity_m`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_m), [`weight_sparsity_n`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_n), [`activation_sparsity_m`](../../_src/core/sparsity.md#SparsityRule.activation_sparsity_m), [`weight_sparsity_order`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_order), [`weight_sparsity_start_step`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_start_step), [`activation_sparsity_n`](../../_src/core/sparsity.md#SparsityRule.activation_sparsity_n), [`eval_mode`](../../_src/core/sparsity.md#SparsityRule.eval_mode), [`weight_sparsity_block_size`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_block_size), [`weight_sparsity_offset`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_offset), [`weight_sparsity_update_step`](../../_src/core/sparsity.md#SparsityRule.weight_sparsity_update_step), [`activation_sparsity_block_size`](../../_src/core/sparsity.md#SparsityRule.activation_sparsity_block_size), [`activation_sparsity_offset`](../../_src/core/sparsity.md#SparsityRule.activation_sparsity_offset), [`activation_sparsity_order`](../../_src/core/sparsity.md#SparsityRule.activation_sparsity_order)
- used by: (4 test-only callers)

