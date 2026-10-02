---
title: 'Module: axlearn/common/ops/_optimization_barrier_test.py'
type: catalog
provenance: extracted
module: axlearn/common/ops/_optimization_barrier_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.ops._optimization_barrier_test`/OptimizationBarrierTest#
symbols:
  OptimizationBarrierTest: ''
  OptimizationBarrierTest.test_forward_optimization_barrier: test_forward_optimization_barrier().
  OptimizationBarrierTest.test_forward_optimization_barrier_grad: test_forward_optimization_barrier_grad().
  OptimizationBarrierTest.test_forward_optimization_barrier_vmap: test_forward_optimization_barrier_vmap().
---
# Module: [`axlearn/common/ops/_optimization_barrier_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier_test.py)

## Classes
### `OptimizationBarrierTest`
- def: [`axlearn/common/ops/_optimization_barrier_test.py:18`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier_test.py#L18)
- doc: Tests ops.optimization_barrier
- signature: `class OptimizationBarrierTest(TestCase):`
- members:
  - `test_forward_optimization_barrier(self)` — [`L21`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier_test.py#L21) — Test that constant folding happens without a barrier and does
  - `test_forward_optimization_barrier_grad(self)` — [`L56`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier_test.py#L56) — Tests that `forward_optimization_barrier` does not wrap new computations generated
  - `test_forward_optimization_barrier_vmap(self)` — [`L80`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier_test.py#L80) — Tests that `forward_optimization_barrier` works with vmap.

