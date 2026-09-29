---
title: 'Module: python/sgl_jax/srt/speculative/relay_buffer.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/relay_buffer.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.relay_buffer`/
symbols:
  update_spec_relay_buffers: update_spec_relay_buffers().
  create_spec_relay_buffers: create_spec_relay_buffers().
  gather_spec_relay_buffers: gather_spec_relay_buffers().
  RELAY_STATE_SPEC: RELAY_STATE_SPEC.
  SpecRelayBuffers: SpecRelayBuffers#
  SpecRelayBuffers.topk_index: SpecRelayBuffers#topk_index.
  RELAY_ID_SPEC: RELAY_ID_SPEC.
  SpecRelayBuffers.hidden_states: SpecRelayBuffers#hidden_states.
  SpecRelayBuffers.verified_id: SpecRelayBuffers#verified_id.
  SpecRelayBuffers.new_seq_lens: SpecRelayBuffers#new_seq_lens.
  make_dp_valid_mask: make_dp_valid_mask().
---
# Module: [`python/sgl_jax/srt/speculative/relay_buffer.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py)

## Classes
### `SpecRelayBuffers`  ·  implements/extends NamedTuple
- def: [`python/sgl_jax/srt/speculative/relay_buffer.py:15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L15)
- signature: `class SpecRelayBuffers(NamedTuple):`
- members:
  - `hidden_states` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L17)
  - `new_seq_lens` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L19)
  - `topk_index` — [`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L16)
  - `verified_id` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L18)
- used by: [`update_spec_relay_buffers`](relay_buffer.md#update_spec_relay_buffers), [`create_spec_relay_buffers`](relay_buffer.md#create_spec_relay_buffers), [`gather_spec_relay_buffers`](relay_buffer.md#gather_spec_relay_buffers)

## Functions
- `create_spec_relay_buffers(mesh, req_to_token_pool, *, dp_size: int, num_steps: int, hidden_size: int, hidden_dtype)` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L22) — Create DP-local req-indexed buffers for cross-batch draft state relay.
- `gather_spec_relay_buffers(buffers: SpecRelayBuffers, future_indices, *, dp_size: int)` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L107) — Gather DP-padded draft state for the next batch.
- `make_dp_valid_mask(real_bs_per_dp, *, total_bs: int, per_dp_bs: int)` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L134)
- `update_spec_relay_buffers(buffers: SpecRelayBuffers, future_indices, valid_mask, topk_index, hidden_states, verified_id, new_seq_lens, *, dp_size: int)` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L56) — Write DP-padded draft state into relay buffers without touching padded rows.

## Module values
- `RELAY_ID_SPEC` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L12)
- `RELAY_STATE_SPEC` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/relay_buffer.py#L11)

