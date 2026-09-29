---
title: 'Module: test/test_shared_weights.py'
type: catalog
provenance: extracted
module: test/test_shared_weights.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_shared_weights`/SharedWeights
symbols:
  SharedWeightsTest.test_shared_weights: Test#test_shared_weights().
  SharedWeightsModel.head: Model#head.
  SharedWeightsModel.forward: Model#forward().
  SharedWeightsModel.emb: Model#emb.
  SharedWeightsTest: Test#
  SharedWeightsModel: Model#
  SharedWeightsModel.mlp: Model#mlp.
  SharedWeightsModel.__init__: Model#__init__().
---
# Module: [`test/test_shared_weights.py`](../../../../../raw/code/torchax/test/test_shared_weights.py)

## Classes
### `SharedWeightsModel`  ·  implements/extends Module
- def: [`test/test_shared_weights.py:10`](../../../../../raw/code/torchax/test/test_shared_weights.py#L10)
- signature: `class SharedWeightsModel(nn.Module):`
- members:
  - `forward(self, x)` — [`L22`](../../../../../raw/code/torchax/test/test_shared_weights.py#L22)
  - `emb` — [`L13`](../../../../../raw/code/torchax/test/test_shared_weights.py#L13)
  - `head` — [`L17`](../../../../../raw/code/torchax/test/test_shared_weights.py#L17)
  - `mlp` — [`L14`](../../../../../raw/code/torchax/test/test_shared_weights.py#L14)
- protocol/private: `__init__`[`L11`](../../../../../raw/code/torchax/test/test_shared_weights.py#L11)
- used by: (1 test-only callers)

### `SharedWeightsTest`  ·  implements/extends TestCase
- def: [`test/test_shared_weights.py:29`](../../../../../raw/code/torchax/test/test_shared_weights.py#L29)
- signature: `class SharedWeightsTest(base_test_util.TestCase):`
- members:
  - `test_shared_weights(self)` — [`L30`](../../../../../raw/code/torchax/test/test_shared_weights.py#L30)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`extract_jax`](../torchax/__init__.md#extract_jax), [`t2j_copy`](../torchax/tensor.md#Environment.t2j_copy), [`j2t_copy`](../torchax/tensor.md#Environment.j2t_copy)  (4 test-only)

