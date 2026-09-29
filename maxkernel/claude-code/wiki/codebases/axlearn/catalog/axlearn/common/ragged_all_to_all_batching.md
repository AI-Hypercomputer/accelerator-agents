---
title: 'Module: axlearn/common/ragged_all_to_all_batching.py'
type: catalog
provenance: extracted
module: axlearn/common/ragged_all_to_all_batching.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.ragged_all_to_all_batching`/
symbols:
  _create_ragged_all_to_all_with_axis.ragged_all_to_all_vmap_rule_fixed: _create_ragged_all_to_all_with_axis().ragged_all_to_all_vmap_rule_fixed().
  ragged_all_to_all_batched: ragged_all_to_all_batched().
  _create_ragged_all_to_all_with_axis: _create_ragged_all_to_all_with_axis().
  _create_ragged_all_to_all_with_axis.ragged_all_to_all_fixed_axis: _create_ragged_all_to_all_with_axis().ragged_all_to_all_fixed_axis().
  _create_ragged_all_to_all_with_axis.ragged_all_to_all_vmap_rule_fixed.loop_body: _create_ragged_all_to_all_with_axis().ragged_all_to_all_vmap_rule_fixed().loop_body().
---
# Module: [`axlearn/common/ragged_all_to_all_batching.py`](../../../../../../raw/code/axlearn/axlearn/common/ragged_all_to_all_batching.py)

## Functions
- `_create_ragged_all_to_all_with_axis(axis_name: str)` — [`L18`](../../../../../../raw/code/axlearn/axlearn/common/ragged_all_to_all_batching.py#L18) — Create a ragged_all_to_all function for custom_vmap.
- `loop_body(i, current_results)` — [`L74`](../../../../../../raw/code/axlearn/axlearn/common/ragged_all_to_all_batching.py#L74)
- `ragged_all_to_all_batched(inputs: Tensor, outputs: Tensor, input_offsets: Tensor, send_sizes: Tensor, output_offsets: Tensor, recv_sizes: Tensor, *, axis_name: str, axis_index_groups: Optional[Any] = None)` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/ragged_all_to_all_batching.py#L106) — Drop-in replacement for lax.ragged_all_to_all with batching support.
- `ragged_all_to_all_fixed_axis(inputs, outputs, input_offsets, send_sizes, output_offsets, recv_sizes)` — [`L22`](../../../../../../raw/code/axlearn/axlearn/common/ragged_all_to_all_batching.py#L22) — Custom vmap-compatible wrapper.
- `ragged_all_to_all_vmap_rule_fixed(axis_size, in_batched, inputs, outputs, input_offsets, send_sizes, output_offsets, recv_sizes)` — [`L38`](../../../../../../raw/code/axlearn/axlearn/common/ragged_all_to_all_batching.py#L38) — Custom batching rule for ragged_all_to_all.

