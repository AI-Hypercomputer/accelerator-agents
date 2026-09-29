---
title: 'Module: qwix/_src/core/sparsity.py'
type: catalog
provenance: extracted
module: qwix/_src/core/sparsity.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.sparsity`/
symbols:
  get_sparsity_mask: get_sparsity_mask().
  SparsityRule: SparsityRule#
  SparsityRule.weight_sparsity_m: SparsityRule#weight_sparsity_m.
  SparsityRule.weight_sparsity_n: SparsityRule#weight_sparsity_n.
  prune_inputs_n_m: prune_inputs_n_m().
  apply_sparsity: apply_sparsity().
  SparsityRule.weight_sparsity_order: SparsityRule#weight_sparsity_order.
  SparsityRule.weight_sparsity_start_step: SparsityRule#weight_sparsity_start_step.
  SparsityRule.activation_sparsity_m: SparsityRule#activation_sparsity_m.
  get_sparsity_mask_unstructured: get_sparsity_mask_unstructured().
  SparsityRule.weight_sparsity_block_size: SparsityRule#weight_sparsity_block_size.
  SparsityRule.weight_sparsity_offset: SparsityRule#weight_sparsity_offset.
  SparsityRule.weight_sparsity_update_step: SparsityRule#weight_sparsity_update_step.
  SparsityRule.eval_mode: SparsityRule#eval_mode.
  SparsityRule.activation_sparsity_n: SparsityRule#activation_sparsity_n.
  SparsityRule.activation_sparsity_order: SparsityRule#activation_sparsity_order.
  SparsityRule.activation_sparsity_block_size: SparsityRule#activation_sparsity_block_size.
  SparsityRule.activation_sparsity_offset: SparsityRule#activation_sparsity_offset.
  get_sparsity_mask.block_score: get_sparsity_mask().block_score().
---
# Module: [`qwix/_src/core/sparsity.py`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py)

## Classes
### `SparsityRule`
- def: [`qwix/_src/core/sparsity.py:25`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L25) — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
- doc: Sparsity rules that match and configure the sparsity behavior.
- signature: `class SparsityRule:`
- members:
  - `activation_sparsity_block_size` — [`L47`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L47)
  - `activation_sparsity_m` — [`L45`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L45)
  - `activation_sparsity_n` — [`L44`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L44)
  - `activation_sparsity_offset` — [`L48`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L48)
  - `activation_sparsity_order` — [`L46`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L46)
  - `eval_mode` — [`L42`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L42)
  - `weight_sparsity_block_size` — [`L37`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L37)
  - `weight_sparsity_m` — [`L33`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L33) — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
  - `weight_sparsity_n` — [`L32`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L32) — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
  - `weight_sparsity_offset` — [`L38`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L38)
  - `weight_sparsity_order` — [`L36`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L36)
  - `weight_sparsity_start_step` — [`L39`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L39)
  - `weight_sparsity_update_step` — [`L40`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L40)
- used by: [`__call__`](../../contrib/sparsity/sparsity_module.md#SparsityModule.__call__), [`sparsify`](qarray.md#sparsify), [`mask_update`](../../contrib/sparsity/sparsity_module.md#SparsityModule.mask_update), [`should_update_mask`](../../contrib/sparsity/sparsity_module.md#SparsityModule.should_update_mask), [`sparsity_rule`](dot_general_qt.md#DotGeneralQtConfig.sparsity_rule), [`__init__`](../../contrib/sparsity/sparsity_module.md#SparsityModule.__init__)  (6 test-only)

## Functions
- `apply_sparsity(inputs: jax.Array, mask: jax.Array, is_channelwise: bool = False, pruned_value: Optional[jax.Array] = None)` — [`L51`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L51) — Returns sparsified inputs based on input mask.
- `block_score(inputs: jax.Array)` — [`L142`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L142) — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
- `get_sparsity_mask(inputs: jax.Array, n_sparsity: int = 0, m_sparsity: int = 0, order: str = 'R', block_size: int = 0, offset: int = 0)` — [`L81`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L81) — Returns sparsified inputs for n:m structured pruning. — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
- `get_sparsity_mask_unstructured(inputs: jax.Array, mask: jax.Array | None, prune_rate: jax.Array | float)` — [`L233`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L233) — Computes a sparisty mask to prune the required percentage of weights.
- `prune_inputs_n_m(inputs: jax.Array, *, n: int, m: int, order: str = 'R', offset: int = 0)` — [`L262`](../../../../../../../raw/code/qwix/qwix/_src/core/sparsity.py#L262) — Returns pruned array with N:M (structured) pruning. — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)

