---
title: 'Module: test/test_train.py'
type: catalog
provenance: extracted
module: test/test_train.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_train`/TrainTest#
symbols:
  TrainTest.test_train_step_can_run: test_train_step_can_run().
  TrainTest.setUp: setUp().
  TrainTest.test_scan_module: test_scan_module().
  TrainTest.model_fn: model_fn().
  TrainTest: ''
---
# Module: [`test/test_train.py`](../../../../../raw/code/torchax/test/test_train.py)

## Classes
### `TrainTest`  ·  implements/extends TestCase
- def: [`test/test_train.py:25`](../../../../../raw/code/torchax/test/test_train.py#L25)
- signature: `class TrainTest(unittest.TestCase):`
- members:
  - `model_fn(weight, buffers, args)` — [`L63`](../../../../../raw/code/torchax/test/test_train.py#L63)
  - `setUp(self)` — [`L26`](../../../../../raw/code/torchax/test/test_train.py#L26)
  - `test_scan_module(self)` — [`L30`](../../../../../raw/code/torchax/test/test_train.py#L30)
  - `test_train_step_can_run(self)` — [`L53`](../../../../../raw/code/torchax/test/test_train.py#L53)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`enable_accuracy_mode`](../torchax/__init__.md#enable_accuracy_mode)

