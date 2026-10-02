---
title: 'Module: test/test_checkpoint.py'
type: catalog
provenance: extracted
module: test/test_checkpoint.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_checkpoint`/CheckpointTest#
symbols:
  CheckpointTest.test_save_and_load_jax_style_checkpoint: test_save_and_load_jax_style_checkpoint().
  CheckpointTest.test_load_pytorch_style_checkpoint: test_load_pytorch_style_checkpoint().
  CheckpointTest.test_load_non_existent_checkpoint: test_load_non_existent_checkpoint().
  CheckpointTest: ''
---
# Module: [`test/test_checkpoint.py`](../../../../../raw/code/torchax/test/test_checkpoint.py)

## Classes
### `CheckpointTest`  ·  implements/extends TestCase
- def: [`test/test_checkpoint.py:30`](../../../../../raw/code/torchax/test/test_checkpoint.py#L30)
- signature: `class CheckpointTest(unittest.TestCase):`
- members:
  - `test_load_non_existent_checkpoint(self)` — [`L114`](../../../../../raw/code/torchax/test/test_checkpoint.py#L114)
  - `test_load_pytorch_style_checkpoint(self)` — [`L71`](../../../../../raw/code/torchax/test/test_checkpoint.py#L71)
  - `test_save_and_load_jax_style_checkpoint(self)` — [`L31`](../../../../../raw/code/torchax/test/test_checkpoint.py#L31)
- uses (calls/refs, reference-scoped): [`extract_jax`](../torchax/__init__.md#extract_jax), [`enable_globally`](../torchax/__init__.md#enable_globally), [`disable_globally`](../torchax/__init__.md#disable_globally), [`_to_jax`](../torchax/checkpoint.md#_to_jax), [`load_checkpoint`](../torchax/checkpoint.md#load_checkpoint), [`save_checkpoint`](../torchax/checkpoint.md#save_checkpoint), [`_to_torch`](../torchax/checkpoint.md#_to_torch)

