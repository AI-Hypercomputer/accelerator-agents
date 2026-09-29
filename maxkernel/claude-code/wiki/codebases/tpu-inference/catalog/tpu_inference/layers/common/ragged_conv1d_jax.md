---
title: 'Module: tpu_inference/layers/common/ragged_conv1d_jax.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/ragged_conv1d_jax.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.ragged_conv1d_jax`/
symbols:
  _fix_query_start_loc: _fix_query_start_loc().
  _get_boundary_indices: _get_boundary_indices().
  _get_state_update_indices: _get_state_update_indices().
  _depthwise_conv1d_loop_and_bias: _depthwise_conv1d_loop_and_bias().
  ragged_conv1d_mixed_prefill: ragged_conv1d_mixed_prefill().
  ragged_conv1d_decode_only: ragged_conv1d_decode_only().
  ragged_conv1d: ragged_conv1d().
---
# Module: [`tpu_inference/layers/common/ragged_conv1d_jax.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py)

## Functions
- `_depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias)` — [`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L76) — Depthwise 1D convolution using loops over kernel size.
- `_fix_query_start_loc(query_start_loc, num_valid_seqs)` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L36) — Fixes query_start_loc to be non-decreasing for invalid sequences.
- `_get_boundary_indices(starts, lengths, kernel_size, num_valid_seqs)` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L43) — Computes indices for boundary fixup.
- `_get_state_update_indices(query_start_loc, kernel_size, num_tokens)` — [`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L58) — Computes indices for updating the convolutional state.
- `ragged_conv1d(x: jax.Array, conv_state: jax.Array, conv_weight: jax.Array, conv_bias: jax.Array | None, query_start_loc: jax.Array, state_indices: jax.Array, distribution: jax.Array, has_initial_state: jax.Array, *, kernel_size: int)` — [`L250`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L250) — Applies 1D convolution over ragged sequences and updates state.
- `ragged_conv1d_decode_only(x, conv_state, conv_weight, conv_bias, query_start_loc, state_indices, distribution, has_initial_state, *, kernel_size)` — [`L193`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L193) — Apply conv1d for decode-only case (All valid reqs have seq_len=1).
- `ragged_conv1d_mixed_prefill(x, conv_state, conv_weight, conv_bias, query_start_loc, state_indices, distribution, has_initial_state, *, kernel_size)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/ragged_conv1d_jax.py#L106) — Applies 1D convolution, optimized for prefill.

