---
title: 'Module: tpu_inference/runner/block_table.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/block_table.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.block_table`/
symbols:
  MultiGroupBlockTable.block_tables: MultiGroupBlockTable#block_tables.
  BlockTable.num_blocks_per_row: BlockTable#num_blocks_per_row.
  BlockTable.block_table_cpu: BlockTable#block_table_cpu.
  BlockTable.append_row: BlockTable#append_row().
  BlockTable.get_cpu_tensor: BlockTable#get_cpu_tensor().
  BlockTable.add_row: BlockTable#add_row().
  BlockTable.move_row: BlockTable#move_row().
  BlockTable.swap_row: BlockTable#swap_row().
  BlockTable.commit: BlockTable#commit().
  BlockTable.clear: BlockTable#clear().
  MultiGroupBlockTable.append_row: MultiGroupBlockTable#append_row().
  MultiGroupBlockTable.add_row: MultiGroupBlockTable#add_row().
  MultiGroupBlockTable.move_row: MultiGroupBlockTable#move_row().
  MultiGroupBlockTable.swap_row: MultiGroupBlockTable#swap_row().
  MultiGroupBlockTable.commit: MultiGroupBlockTable#commit().
  MultiGroupBlockTable.clear: MultiGroupBlockTable#clear().
  MultiGroupBlockTable.__getitem__: MultiGroupBlockTable#__getitem__().
  BlockTable.max_num_blocks_per_req: BlockTable#max_num_blocks_per_req.
  BlockTable.block_table: BlockTable#block_table.
  BlockTable.get_device_tensor: BlockTable#get_device_tensor().
  BlockTable: BlockTable#
  MultiGroupBlockTable: MultiGroupBlockTable#
  logger: logger.
  BlockTable.__init__: BlockTable#__init__().
  BlockTable.max_num_reqs: BlockTable#max_num_reqs.
  BlockTable.max_num_batched_tokens: BlockTable#max_num_batched_tokens.
  BlockTable.pin_memory: BlockTable#pin_memory.
  MultiGroupBlockTable.__init__: MultiGroupBlockTable#__init__().
---
# Module: [`tpu_inference/runner/block_table.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py)

## Classes
### `BlockTable`
- def: [`tpu_inference/runner/block_table.py:13`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L13)
- signature: `class BlockTable:`
- members:
  - `add_row(self, block_ids: list[int], row_idx: int)` — [`L49`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L49)
  - `append_row(self, block_ids: list[int], row_idx: int)` — [`L37`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L37)
  - `clear(self)` — [`L71`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L71)
  - `commit(self, num_reqs: int)` — [`L67`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L67)
  - `get_cpu_tensor(self)` — [`L79`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L79) — Returns the CPU tensor of the block table.
  - `get_device_tensor(self)` — [`L75`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L75) — Ruturns the device tensor of the block table.
  - `move_row(self, src: int, tgt: int)` — [`L53`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L53)
  - `swap_row(self, src: int, tgt: int)` — [`L59`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L59)
  - `block_table` — [`L27`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L27)
  - `block_table_cpu` — [`L31`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L31)
  - `max_num_batched_tokens` — [`L24`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L24)
  - `max_num_blocks_per_req` — [`L23`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L23)
  - `max_num_reqs` — [`L22`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L22)
  - `num_blocks_per_row` — [`L35`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L35)
  - `pin_memory` — [`L25`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L25)
- protocol/private: `__init__`[`L15`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L15)
- used by: [`_precompile_mtp_helpers`](compilation_manager.md#CompilationManager._precompile_mtp_helpers), [`_precompile_eagle3_helpers`](compilation_manager.md#CompilationManager._precompile_eagle3_helpers), [`build_block_table`](compilation_manager.md#CompilationManager.build_block_table), `build_block_table_host`, [`block_tables`](block_table.md#MultiGroupBlockTable.block_tables), [`add_row`](block_table.md#MultiGroupBlockTable.add_row), [`append_row`](block_table.md#MultiGroupBlockTable.append_row), [`move_row`](block_table.md#MultiGroupBlockTable.move_row), [`swap_row`](block_table.md#MultiGroupBlockTable.swap_row), [`__getitem__`](block_table.md#MultiGroupBlockTable.__getitem__), [`clear`](block_table.md#MultiGroupBlockTable.clear), [`commit`](block_table.md#MultiGroupBlockTable.commit)

### `MultiGroupBlockTable`
- def: [`tpu_inference/runner/block_table.py:84`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L84)
- doc: The BlockTables for each KV cache group.
- signature: `class MultiGroupBlockTable:`
- members:
  - `__getitem__(self, idx: int)` — [`L126`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L126) — Returns the BlockTable for the i-th KV cache group.
  - `add_row(self, block_ids: list[list[int]], row_idx: int)` — [`L106`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L106)
  - `append_row(self, block_ids: list[list[int]], row_idx: int)` — [`L102`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L102)
  - `clear(self)` — [`L122`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L122)
  - `commit(self, num_reqs: int)` — [`L118`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L118)
  - `move_row(self, src: int, tgt: int)` — [`L110`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L110)
  - `swap_row(self, src: int, tgt: int)` — [`L114`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L114)
  - `block_tables` — [`L94`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L94)
- protocol/private: `__init__`[`L87`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L87)
- uses (calls/refs, reference-scoped): [`append_row`](block_table.md#BlockTable.append_row), [`add_row`](block_table.md#BlockTable.add_row), [`clear`](block_table.md#BlockTable.clear), [`commit`](block_table.md#BlockTable.commit), [`move_row`](block_table.md#BlockTable.move_row), [`swap_row`](block_table.md#BlockTable.swap_row), [`BlockTable`](block_table.md#BlockTable)
- used by: [`update_states`](persistent_batch_manager.md#PersistentBatchManager.update_states), [`add_request`](input_batch.md#InputBatch.add_request), [`condense`](input_batch.md#InputBatch.condense), [`swap_states`](input_batch.md#InputBatch.swap_states), [`block_table`](input_batch.md#InputBatch.block_table)

## Module values
- `logger` — [`L9`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/block_table.py#L9)

