---
title: 'Module: simply/utils/pytree_test.py'
type: catalog
provenance: extracted
module: simply/utils/pytree_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.pytree_test`/
symbols:
  _B: _B#
  _A: _A#
  PyTreeTest.test_save_and_load_pytree: PyTreeTest#test_save_and_load_pytree().
  PyTreeTest.test_load: PyTreeTest#test_load().
  PyTreeTest.test_concatenate_pytrees: PyTreeTest#test_concatenate_pytrees().
  _C: _C#
  PyTreeTest.test_dump: PyTreeTest#test_dump().
  PyTreeTest.test_enum: PyTreeTest#test_enum().
  _TestEnum: _TestEnum#
  _B.y: _B#y.
  PyTreeTest.test_tree_type: PyTreeTest#test_tree_type().
  PyTreeTest.test_traverse_tree: PyTreeTest#test_traverse_tree().
  PyTreeTest.test_load_dict: PyTreeTest#test_load_dict().
  PyTreeTest.test_ndarray: PyTreeTest#test_ndarray().
  _C.a: _C#a.
  _C.b: _C#b.
  _C.d: _C#d.
  PyTreeTest.test_tree: PyTreeTest#test_tree().
  PyTreeTest.test_set_tree: PyTreeTest#test_set_tree().
  PyTreeTest.test_tree_mapping_schema_check: PyTreeTest#test_tree_mapping_schema_check().
  PyTreeTest.test_tree_sequence_schema_check: PyTreeTest#test_tree_sequence_schema_check().
  PyTreeTest.test_trim_none: PyTreeTest#test_trim_none().
  PyTreeTest.test_to_flat_dict: PyTreeTest#test_to_flat_dict().
  PyTreeTest.test_tree_leaves_with_tag: PyTreeTest#test_tree_leaves_with_tag().
  PyTreeTest._output: PyTreeTest#_output().
  _A.x: _A#x.
  _TestEnum.VAL1: _TestEnum#VAL1.
  _TestEnum.VAL2: _TestEnum#VAL2.
  PyTreeTest: PyTreeTest#
---
# Module: [`simply/utils/pytree_test.py`](../../../../../../raw/code/simply/simply/utils/pytree_test.py)

## Classes
### `PyTreeTest`  ·  implements/extends TestCase
- def: [`simply/utils/pytree_test.py:51`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L51)
- signature: `class PyTreeTest(absltest.TestCase):`
- members:
  - `test_concatenate_pytrees(self)` — [`L260`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L260)
  - `test_dump(self)` — [`L214`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L214)
  - `test_enum(self)` — [`L246`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L246)
  - `test_load(self)` — [`L170`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L170)
  - `test_load_dict(self)` — [`L201`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L201)
  - `test_ndarray(self)` — [`L230`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L230)
  - `test_save_and_load_pytree(self)` — [`L282`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L282)
  - `test_set_tree(self)` — [`L67`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L67)
  - `test_to_flat_dict(self)` — [`L278`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L278)
  - `test_traverse_tree(self)` — [`L121`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L121)
  - `test_tree(self)` — [`L53`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L53)
  - `test_tree_leaves_with_tag(self)` — [`L288`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L288)
  - `test_tree_mapping_schema_check(self)` — [`L103`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L103)
  - `test_tree_sequence_schema_check(self)` — [`L113`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L113)
  - `test_tree_type(self)` — [`L93`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L93)
  - `test_trim_none(self)` — [`L271`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L271)
- protocol/private: `_output`[`L140`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L140)
- uses (calls/refs, reference-scoped): [`load`](pytree.md#load), [`dump`](pytree.md#dump), [`traverse_tree_with_path`](pytree.md#traverse_tree_with_path), [`set_tree_value`](pytree.md#set_tree_value), [`tree_is_sequence`](pytree.md#tree_is_sequence), [`_B`](pytree_test.md#_B), [`tree_is_mapping`](pytree.md#tree_is_mapping), [`concatenate_pytrees`](pytree.md#concatenate_pytrees), [`tree_value`](pytree.md#tree_value), [`trim_none`](pytree.md#trim_none), [`_A`](pytree_test.md#_A), [`load_pytree_from`](pytree.md#load_pytree_from), [`check_trees_match_mapping_keys`](pytree.md#check_trees_match_mapping_keys), [`save_pytree_to`](pytree.md#save_pytree_to), [`check_trees_match_sequence_length`](pytree.md#check_trees_match_sequence_length), [`to_flat_dict`](pytree.md#to_flat_dict), [`_C`](pytree_test.md#_C), [`_TestEnum`](pytree_test.md#_TestEnum), [`tree_leaves_with_tag`](pytree.md#tree_leaves_with_tag)

### `_A`
- def: [`simply/utils/pytree_test.py:27`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L27)
- signature: `class _A:`
- members:
  - `x` — [`L28`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L28)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry)
- used by: [`_B`](pytree_test.md#_B), [`test_save_and_load_pytree`](pytree_test.md#PyTreeTest.test_save_and_load_pytree), [`test_concatenate_pytrees`](pytree_test.md#PyTreeTest.test_concatenate_pytrees), [`test_load`](pytree_test.md#PyTreeTest.test_load), [`test_load_dict`](pytree_test.md#PyTreeTest.test_load_dict), [`a`](pytree_test.md#_C.a), [`d`](pytree_test.md#_C.d)

### `_B`
- def: [`simply/utils/pytree_test.py:33`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L33)
- signature: `class _B(_A):`
- members:
  - `y` — [`L34`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L34)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry), [`_A`](pytree_test.md#_A)
- used by: [`test_save_and_load_pytree`](pytree_test.md#PyTreeTest.test_save_and_load_pytree), [`test_concatenate_pytrees`](pytree_test.md#PyTreeTest.test_concatenate_pytrees), [`test_load`](pytree_test.md#PyTreeTest.test_load), [`test_dump`](pytree_test.md#PyTreeTest.test_dump), [`b`](pytree_test.md#_C.b)

### `_C`
- def: [`simply/utils/pytree_test.py:39`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L39)
- signature: `class _C:`
- members:
  - `a` — [`L40`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L40)
  - `b` — [`L41`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L41)
  - `d` — [`L42`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L42)
- uses (calls/refs, reference-scoped): [`_B`](pytree_test.md#_B), [`_A`](pytree_test.md#_A)
- used by: [`test_save_and_load_pytree`](pytree_test.md#PyTreeTest.test_save_and_load_pytree), [`test_concatenate_pytrees`](pytree_test.md#PyTreeTest.test_concatenate_pytrees), [`test_load`](pytree_test.md#PyTreeTest.test_load), [`test_dump`](pytree_test.md#PyTreeTest.test_dump)

### `_TestEnum`
- def: [`simply/utils/pytree_test.py:46`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L46)
- signature: `class _TestEnum(enum.Enum):`
- members:
  - `VAL1` — [`L47`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L47)
  - `VAL2` — [`L48`](../../../../../../raw/code/simply/simply/utils/pytree_test.py#L48)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry)
- used by: [`test_enum`](pytree_test.md#PyTreeTest.test_enum)

