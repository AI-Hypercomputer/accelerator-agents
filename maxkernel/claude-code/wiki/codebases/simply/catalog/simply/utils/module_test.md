---
title: 'Module: simply/utils/module_test.py'
type: catalog
provenance: extracted
module: simply/utils/module_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.module_test`/E
symbols:
  EinsumLinearTest.test_dump: insumLinearTest#test_dump().
  EinsumLinearTest.test_gmoe_routing: insumLinearTest#test_gmoe_routing().
  EinsumLinearTest.test_classic_linear: insumLinearTest#test_classic_linear().
  EinsumLinearTest.test_classic_linear2: insumLinearTest#test_classic_linear2().
  EinsumLinearTest.test_gmoe_linear: insumLinearTest#test_gmoe_linear().
  EinsumLinearTest.test_gmoe_linear2: insumLinearTest#test_gmoe_linear2().
  EinsumLinearTest.test_mha_qkv: insumLinearTest#test_mha_qkv().
  EinsumLinearTest.test_combined_qkv: insumLinearTest#test_combined_qkv().
  EinsumLinearTest.test_mha_o: insumLinearTest#test_mha_o().
  EmbeddingLinearTest.test_embedding_linear_tied_embedding: mbeddingLinearTest#test_embedding_linear_tied_embedding().
  EmbeddingLinearTest.test_embedding_linear_untied_embedding: mbeddingLinearTest#test_embedding_linear_untied_embedding().
  EmbeddingLinearTest.test_embedding_linear_embed: mbeddingLinearTest#test_embedding_linear_embed().
  EmbeddingLinearTest.test_embedding_linear_apply: mbeddingLinearTest#test_embedding_linear_apply().
  EinsumLinearTest: insumLinearTest#
  EmbeddingLinearTest: mbeddingLinearTest#
---
# Module: [`simply/utils/module_test.py`](../../../../../../raw/code/simply/simply/utils/module_test.py)

## Classes
### `EinsumLinearTest`  ·  implements/extends TestCase
- def: [`simply/utils/module_test.py:25`](../../../../../../raw/code/simply/simply/utils/module_test.py#L25)
- signature: `class EinsumLinearTest(absltest.TestCase):`
- members:
  - `test_classic_linear(self)` — [`L53`](../../../../../../raw/code/simply/simply/utils/module_test.py#L53)
  - `test_classic_linear2(self)` — [`L71`](../../../../../../raw/code/simply/simply/utils/module_test.py#L71)
  - `test_combined_qkv(self)` — [`L156`](../../../../../../raw/code/simply/simply/utils/module_test.py#L156)
  - `test_dump(self)` — [`L27`](../../../../../../raw/code/simply/simply/utils/module_test.py#L27)
  - `test_gmoe_linear(self)` — [`L89`](../../../../../../raw/code/simply/simply/utils/module_test.py#L89)
  - `test_gmoe_linear2(self)` — [`L118`](../../../../../../raw/code/simply/simply/utils/module_test.py#L118)
  - `test_gmoe_routing(self)` — [`L201`](../../../../../../raw/code/simply/simply/utils/module_test.py#L201)
  - `test_mha_o(self)` — [`L183`](../../../../../../raw/code/simply/simply/utils/module_test.py#L183)
  - `test_mha_qkv(self)` — [`L138`](../../../../../../raw/code/simply/simply/utils/module_test.py#L138)
- uses (calls/refs, reference-scoped): [`dump`](pytree.md#dump), [`EinsumLinear`](module.md#EinsumLinear), [`NOT_ANNOTATED`](sharding.md#NOT_ANNOTATED), [`IdentityInit`](initializer.md#IdentityInit)

### `EmbeddingLinearTest`  ·  implements/extends TestCase
- def: [`simply/utils/module_test.py:238`](../../../../../../raw/code/simply/simply/utils/module_test.py#L238)
- signature: `class EmbeddingLinearTest(parameterized.TestCase):`
- members:
  - `test_embedding_linear_apply(self)` — [`L272`](../../../../../../raw/code/simply/simply/utils/module_test.py#L272)
  - `test_embedding_linear_embed(self, use_lookup)` — [`L262`](../../../../../../raw/code/simply/simply/utils/module_test.py#L262)
  - `test_embedding_linear_tied_embedding(self)` — [`L240`](../../../../../../raw/code/simply/simply/utils/module_test.py#L240)
  - `test_embedding_linear_untied_embedding(self)` — [`L250`](../../../../../../raw/code/simply/simply/utils/module_test.py#L250)
- uses (calls/refs, reference-scoped): [`EmbeddingLinear`](module.md#EmbeddingLinear)

