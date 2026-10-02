---
title: 'Module: aqt/jax/v2/flax/freezer_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/freezer_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.freezer_test`/
symbols:
  _CustomStructure: _CustomStructure#
  _CustomStructure.member: _CustomStructure#member.
  _CustomStructure.member_list: _CustomStructure#member_list.
  _CustomStructure.member_dict: _CustomStructure#member_dict.
  _CustomStructure.__eq__: _CustomStructure#__eq__().
  TestModel: TestModel#
  TestModel.freezer_mode: TestModel#freezer_mode.
  TestModel.axis_metadata_wrapper: TestModel#axis_metadata_wrapper.
  TestModel.setup: TestModel#setup().
  TestModel.__call__: TestModel#__call__().
  TestModel.freezer_get: TestModel#freezer_get().
  TestModel.freezer_set: TestModel#freezer_set().
  FreezerTest: FreezerTest#
  FreezerTest._create_custom_structure: FreezerTest#_create_custom_structure().
  FreezerTest._assert_same_tree_shape_dtype: FreezerTest#_assert_same_tree_shape_dtype().
  FreezerTest.test_freezer_get_set: FreezerTest#test_freezer_get_set().
---
# Module: [`aqt/jax/v2/flax/freezer_test.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py)

## Classes
### `FreezerTest`
- def: [`aqt/jax/v2/flax/freezer_test.py:71`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L71)
- signature: `class FreezerTest(parameterized.TestCase):`
- members:
  - `_assert_same_tree_shape_dtype(self, tree1, tree2)` — [`L81`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L81) — Checks if the two given pytrees have the same structure with the same leaves' shapes and dtypes.
  - `test_freezer_get_set(self)` — [`L95`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L95)
- protocol/private: `_create_custom_structure`[`L72`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L72)

### `TestModel`
- def: [`aqt/jax/v2/flax/freezer_test.py:47`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L47)
- signature: `class TestModel(nn.Module):`
- members:
  - `__call__(self, x)` — [`L58`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L58) — Emulates basic routine on how to use the freezer.
  - `freezer_get(self)` — [`L64`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L64)
  - `freezer_set(self, x)` — [`L67`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L67)
  - `setup(self)` — [`L51`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L51)
  - `axis_metadata_wrapper` — [`L49`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L49)
  - `freezer_mode` — [`L48`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L48)

### `_CustomStructure`
- def: [`aqt/jax/v2/flax/freezer_test.py:28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L28)
- signature: `class _CustomStructure:`
- members:
  - `member` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L29)
  - `member_dict` — [`L31`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L31)
  - `member_list` — [`L30`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L30)
- protocol/private: `__eq__`[`L37`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer_test.py#L37)

