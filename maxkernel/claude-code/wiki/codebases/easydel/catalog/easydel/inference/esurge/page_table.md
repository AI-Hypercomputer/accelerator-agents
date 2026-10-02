---
title: 'Module: easydel/inference/esurge/page_table.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/page_table.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.page_table`/
symbols:
  MultiGroupPageTable.page_tables: MultiGroupPageTable#page_tables.
  PageTable.clear_row: PageTable#clear_row().
  PageTable.move_row: PageTable#move_row().
  PageTable.clear: PageTable#clear().
  PageTable.page_table_cpu: PageTable#page_table_cpu.
  PageTable.append_row: PageTable#append_row().
  PageTable.num_pages_per_row: PageTable#num_pages_per_row.
  PageTable.swap_row: PageTable#swap_row().
  PAGE_TABLE_PADDING_VAL: PAGE_TABLE_PADDING_VAL.
  MultiGroupPageTable.add_row: MultiGroupPageTable#add_row().
  MultiGroupPageTable.clear_row: MultiGroupPageTable#clear_row().
  PageTable.add_row: PageTable#add_row().
  PageTable.commit: PageTable#commit().
  MultiGroupPageTable.move_row: MultiGroupPageTable#move_row().
  MultiGroupPageTable.swap_row: MultiGroupPageTable#swap_row().
  MultiGroupPageTable.clear: MultiGroupPageTable#clear().
  MultiGroupPageTable.append_rows_batch: MultiGroupPageTable#append_rows_batch().
  PageTable.page_table: PageTable#page_table.
  PageTable.cpu_version: PageTable#cpu_version.
  MultiGroupPageTable.append_row: MultiGroupPageTable#append_row().
  MultiGroupPageTable.commit: MultiGroupPageTable#commit().
  MultiGroupPageTable.__getitem__: MultiGroupPageTable#__getitem__().
  SLOT_MAPPING_PADDING_VAL: SLOT_MAPPING_PADDING_VAL.
  PageTable.get_cpu_tensor: PageTable#get_cpu_tensor().
  PageTable.get_device_tensor: PageTable#get_device_tensor().
  PageTable: PageTable#
  MultiGroupPageTable: MultiGroupPageTable#
  cdiv: cdiv().
  logger: logger.
  PageTable.__init__: PageTable#__init__().
  PageTable.max_num_reqs: PageTable#max_num_reqs.
  PageTable.max_num_pages_per_req: PageTable#max_num_pages_per_req.
  PageTable.max_num_batched_tokens: PageTable#max_num_batched_tokens.
  MultiGroupPageTable.__init__: MultiGroupPageTable#__init__().
---
# Module: [`easydel/inference/esurge/page_table.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py)

## Classes
### `MultiGroupPageTable`
- def: [`easydel/inference/esurge/page_table.py:260`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L260)
- doc: Multi-group page table for grouped-query attention.
- signature: `class MultiGroupPageTable:`
- members:
  - `__getitem__(self, idx: int)` — [`L418`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L418) — Get a specific group's page table.
  - `__init__(self, max_num_reqs: int, max_model_len: int, max_num_batched_tokens: int, page_sizes: list[int], sharding: jax.sharding.Sharding | None = None)` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L274) — Initialize a MultiGroupPageTable with page tables for each group.
  - `add_row(self, page_ids: list[list[int]], row_idx: int)` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L315) — Replace a row across all groups.
  - `append_row(self, page_ids: list[list[int]], row_idx: int)` — [`L301`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L301) — Append pages to a row across all groups.
  - `append_rows_batch(self, page_ids_per_req: list[list[list[int]]], req_indices: list[int])` — [`L387`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L387) — Batch append pages across all groups.
  - `clear(self)` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L378) — Clear all page tables across all groups.
  - `clear_row(self, row_idx: int)` — [`L328`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L328) — Clear a row across all groups, zeroing page IDs and counts.
  - `commit(self, num_reqs: int)` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L366) — Commit CPU modifications to GPU for all groups.
  - `move_row(self, src: int, tgt: int)` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L340) — Move a row across all groups.
  - `swap_row(self, src: int, tgt: int)` — [`L353`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L353) — Swap two rows across all groups.
  - `page_tables` — [`L291`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L291)
- uses (calls/refs, reference-scoped): [`clear_row`](page_table.md#PageTable.clear_row), [`clear`](page_table.md#PageTable.clear), [`move_row`](page_table.md#PageTable.move_row), [`append_row`](page_table.md#PageTable.append_row), [`swap_row`](page_table.md#PageTable.swap_row), [`add_row`](page_table.md#PageTable.add_row), [`commit`](page_table.md#PageTable.commit), [`PageTable`](page_table.md#PageTable), [`cdiv`](page_table.md#cdiv)
- used by: [`_update_states`](runners/model_runner.md#eSurgeRunner._update_states), [`remove_request`](runners/sequence_buffer.md#SequenceBuffer.remove_request), [`clear`](runners/sequence_buffer.md#SequenceBuffer.clear), [`swap_states`](runners/sequence_buffer.md#SequenceBuffer.swap_states), [`_move_request`](runners/sequence_buffer.md#SequenceBuffer._move_request), [`add_request`](runners/sequence_buffer.md#SequenceBuffer.add_request), [`page_table`](runners/sequence_buffer.md#SequenceBuffer.page_table)

### `PageTable`
- def: [`easydel/inference/esurge/page_table.py:52`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L52)
- doc: Manages page allocation for paged KV-cache layouts.
- signature: `class PageTable:`
- members:
  - `__init__(self, max_num_reqs: int, max_num_pages_per_req: int, max_num_batched_tokens: int, sharding: jax.sharding.Sharding | None = None)` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L75) — Initialize a PageTable with specified capacity.
  - `add_row(self, page_ids: list[int], row_idx: int)` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L148) — Replace a row with new page IDs.
  - `append_row(self, page_ids: list[int], row_idx: int)` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L109) — Append page IDs to a single row.
  - `clear(self)` — [`L223`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L223) — Clear all data in the page table.
  - `clear_row(self, row_idx: int)` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L135) — Zero out a row, resetting page IDs and count.
  - `commit(self, num_reqs: int)` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L208) — Commit CPU modifications to GPU.
  - `get_cpu_tensor(self)` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L248) — Get the CPU tensor of the page table.
  - `get_device_tensor(self)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L236) — Get the GPU device tensor of the page table.
  - `move_row(self, src: int, tgt: int)` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L164) — Move row content from source to target.
  - `swap_row(self, src: int, tgt: int)` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L186) — Swap two rows in the page table.
  - `cpu_version` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L107)
  - `max_num_batched_tokens` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L92)
  - `max_num_pages_per_req` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L91)
  - `max_num_reqs` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L90)
  - `num_pages_per_row` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L103)
  - `page_table` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L94)
  - `page_table_cpu` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L99)
- uses (calls/refs, reference-scoped): [`PAGE_TABLE_PADDING_VAL`](page_table.md#PAGE_TABLE_PADDING_VAL)
- used by: [`_execute_model_impl`](runners/model_runner.md#eSurgeRunner._execute_model_impl), [`get_compile_configurations`](runners/execution_manager.md#ExecutionManager.get_compile_configurations), [`page_tables`](page_table.md#MultiGroupPageTable.page_tables), [`add_row`](page_table.md#MultiGroupPageTable.add_row), [`clear_row`](page_table.md#MultiGroupPageTable.clear_row), [`append_rows_batch`](page_table.md#MultiGroupPageTable.append_rows_batch), [`clear`](page_table.md#MultiGroupPageTable.clear), [`move_row`](page_table.md#MultiGroupPageTable.move_row), [`swap_row`](page_table.md#MultiGroupPageTable.swap_row), [`__getitem__`](page_table.md#MultiGroupPageTable.__getitem__), [`append_row`](page_table.md#MultiGroupPageTable.append_row), [`commit`](page_table.md#MultiGroupPageTable.commit)

## Functions
- `cdiv(a: int, b: int)` — [`L30`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L30) — Compute ceiling division.

## Module values
- `PAGE_TABLE_PADDING_VAL` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L49)
- `SLOT_MAPPING_PADDING_VAL` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L48)
- `logger` — [`L27`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/page_table.py#L27)

