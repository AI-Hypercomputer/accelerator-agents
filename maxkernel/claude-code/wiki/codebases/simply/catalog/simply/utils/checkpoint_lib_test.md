---
title: 'Module: simply/utils/checkpoint_lib_test.py'
type: catalog
provenance: extracted
module: simply/utils/checkpoint_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.checkpoint_lib_test`/
symbols:
  QwenFormatTest.test_restore_qwen3_moe_format: QwenFormatTest#test_restore_qwen3_moe_format().
  QwenFormatTest.test_restore_qwen2_format: QwenFormatTest#test_restore_qwen2_format().
  CheckpointFormatTest.test_restore_gemma2_format: CheckpointFormatTest#test_restore_gemma2_format().
  CheckpointFormatTest.test_restore_gemma2_transpose_format: CheckpointFormatTest#test_restore_gemma2_transpose_format().
  CheckpointFormatTest.test_restore_with_format: CheckpointFormatTest#test_restore_with_format().
  CheckpointFormatTest.test_restore_legacy_format: CheckpointFormatTest#test_restore_legacy_format().
  CheckpointFormatTest.expected_state: CheckpointFormatTest#expected_state.
  load_state: load_state().
  QwenFormatTest.expected_state: QwenFormatTest#expected_state.
  CheckpointFormatTest.expected_abstract_state: CheckpointFormatTest#expected_abstract_state.
  CheckpointLibTest.test_dump_format: CheckpointLibTest#test_dump_format().
  QwenFormatTest.qwen2_test_config: QwenFormatTest#qwen2_test_config().
  CheckpointFormatTest.lm_test_config: CheckpointFormatTest#lm_test_config().
  QwenFormatTest.qwen3_moe_test_config: QwenFormatTest#qwen3_moe_test_config().
  QwenFormatTest.expected_abstract_state: QwenFormatTest#expected_abstract_state.
  CheckpointFormatTest: CheckpointFormatTest#
  CheckpointFormatTest.setUp: CheckpointFormatTest#setUp().
  QwenFormatTest: QwenFormatTest#
  QwenFormatTest.setUp: QwenFormatTest#setUp().
  CheckpointLibTest: CheckpointLibTest#
---
# Module: [`simply/utils/checkpoint_lib_test.py`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py)

## Classes
### `CheckpointFormatTest`  ·  implements/extends TestCase
- def: [`simply/utils/checkpoint_lib_test.py:43`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L43)
- signature: `class CheckpointFormatTest(absltest.TestCase):`
- members:
  - `lm_test_config(self)` — [`L45`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L45)
  - `setUp(self)` — [`L58`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L58)
  - `test_restore_gemma2_format(self)` — [`L86`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L86)
  - `test_restore_gemma2_transpose_format(self)` — [`L107`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L107)
  - `test_restore_legacy_format(self)` — [`L66`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L66)
  - `test_restore_with_format(self)` — [`L128`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L128)
  - `expected_abstract_state` — [`L62`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L62)
  - `expected_state` — [`L60`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L60)
- uses (calls/refs, reference-scoped): [`TransformerLM`](../model_lib.md#TransformerLM), [`traverse_tree_with_path`](pytree.md#traverse_tree_with_path), [`load_checkpoint_from_dir`](checkpoint_lib.md#load_checkpoint_from_dir), [`get_raw_arrays`](common.md#get_raw_arrays), [`save_checkpoint`](checkpoint_lib.md#save_checkpoint), [`lm_test`](../config_lib.md#lm_test), [`get_abstract_params`](checkpoint_lib.md#get_abstract_params), [`load_state`](checkpoint_lib_test.md#load_state), [`Gemma2Format`](checkpoint_lib.md#Gemma2Format), [`LegacyFormat`](checkpoint_lib.md#LegacyFormat), [`Gemma2TransposeFormat`](checkpoint_lib.md#Gemma2TransposeFormat)

### `CheckpointLibTest`  ·  implements/extends TestCase
- def: [`simply/utils/checkpoint_lib_test.py:250`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L250)
- signature: `class CheckpointLibTest(absltest.TestCase):`
- members:
  - `test_dump_format(self)` — [`L252`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L252)
- uses (calls/refs, reference-scoped): [`dump`](pytree.md#dump), [`LegacyFormat`](checkpoint_lib.md#LegacyFormat)

### `QwenFormatTest`  ·  implements/extends TestCase
- def: [`simply/utils/checkpoint_lib_test.py:152`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L152)
- signature: `class QwenFormatTest(absltest.TestCase):`
- members:
  - `qwen2_test_config(cls)` — [`L155`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L155)
  - `qwen3_moe_test_config(cls)` — [`L167`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L167)
  - `setUp(self)` — [`L183`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L183)
  - `test_restore_qwen2_format(self)` — [`L191`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L191)
  - `test_restore_qwen3_moe_format(self)` — [`L218`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L218)
  - `expected_abstract_state` — [`L187`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L187)
  - `expected_state` — [`L185`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L185)
- uses (calls/refs, reference-scoped): [`TransformerLM`](../model_lib.md#TransformerLM), [`traverse_tree_with_path`](pytree.md#traverse_tree_with_path), [`load_checkpoint_from_dir`](checkpoint_lib.md#load_checkpoint_from_dir), [`get_raw_arrays`](common.md#get_raw_arrays), [`deepseek_qwen2_1p5b`](../config_lib.md#deepseek_qwen2_1p5b), [`save_checkpoint`](checkpoint_lib.md#save_checkpoint), [`qwen3_30b_a3b`](../config_lib.md#qwen3_30b_a3b), [`get_abstract_params`](checkpoint_lib.md#get_abstract_params), [`load_state`](checkpoint_lib_test.md#load_state), [`Qwen2Format`](checkpoint_lib.md#Qwen2Format), [`set_default_mesh_shape`](sharding.md#set_default_mesh_shape)

## Functions
- `load_state(json_file)` — [`L35`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib_test.py#L35)

