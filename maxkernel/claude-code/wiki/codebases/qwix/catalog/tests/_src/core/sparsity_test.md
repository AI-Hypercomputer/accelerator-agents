---
title: 'Module: tests/_src/core/sparsity_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/sparsity_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.sparsity_test`/
symbols:
  BlockPruningFunctionalityTest.test_prune_inputs_n_m: BlockPruningFunctionalityTest#test_prune_inputs_n_m().
  BlockPruningFunctionalityTest.test_block_pruning: BlockPruningFunctionalityTest#test_block_pruning().
  PruningFunctionalityTest.test_prune_inputs_n_m: PruningFunctionalityTest#test_prune_inputs_n_m().
  PruningFunctionalityTest.test_n_m_pruning_mask: PruningFunctionalityTest#test_n_m_pruning_mask().
  PruningFunctionalityTest.test_pruning: PruningFunctionalityTest#test_pruning().
  PruningFunctionalityTest.test_apply_channelwise_mask: PruningFunctionalityTest#test_apply_channelwise_mask().
  dataclass: dataclass.
  PruningFunctionalityTest: PruningFunctionalityTest#
  BlockPruningFunctionalityTest: BlockPruningFunctionalityTest#
---
# Module: [`tests/_src/core/sparsity_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py)

## Classes
### `BlockPruningFunctionalityTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/sparsity_test.py:206`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L206)
- signature: `class BlockPruningFunctionalityTest(parameterized.TestCase):`
- members:
  - `test_block_pruning(self, order, inputs, exp_output, n_sparsity, m_sparsity, block_size)` — [`L407`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L407)
  - `test_prune_inputs_n_m(self, block_size)` — [`L213`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L213)
- uses (calls/refs, reference-scoped): [`get_sparsity_mask`](../../../qwix/_src/core/sparsity.md#get_sparsity_mask), [`apply_sparsity`](../../../qwix/_src/core/sparsity.md#apply_sparsity)

### `PruningFunctionalityTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/sparsity_test.py:29`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L29)
- signature: `class PruningFunctionalityTest(parameterized.TestCase):`
- members:
  - `test_apply_channelwise_mask(self, inputs, mask, exp_output)` — [`L199`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L199)
  - `test_n_m_pruning_mask(self)` — [`L50`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L50)
  - `test_prune_inputs_n_m(self)` — [`L31`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L31)
  - `test_pruning(self, order, inputs, exp_output, n_sparsity, m_sparsity)` — [`L172`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L172)
- uses (calls/refs, reference-scoped): [`get_sparsity_mask`](../../../qwix/_src/core/sparsity.md#get_sparsity_mask), [`prune_inputs_n_m`](../../../qwix/_src/core/sparsity.md#prune_inputs_n_m), [`apply_sparsity`](../../../qwix/_src/core/sparsity.md#apply_sparsity)

## Module values
- `dataclass` — [`L26`](../../../../../../../raw/code/qwix/tests/_src/core/sparsity_test.py#L26)

