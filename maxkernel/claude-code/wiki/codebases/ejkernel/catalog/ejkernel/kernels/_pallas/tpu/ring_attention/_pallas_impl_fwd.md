---
title: 'Module: ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ring_attention._pallas_impl_fwd`/
symbols:
  _make_block_sizes: _make_block_sizes().
  ring_attention: ring_attention().
  ring_attention.single_batch_attention: ring_attention().single_batch_attention().
  _AttentionSinkMask.attention_sink_size: _AttentionSinkMask#attention_sink_size.
  _AttentionSinkMask.__init__: _AttentionSinkMask#__init__().
  _AttentionSinkMask.sink_mask_function: _AttentionSinkMask#sink_mask_function().
  _build_mask: _build_mask().
  _AttentionSinkMask.__eq__: _AttentionSinkMask#__eq__().
  _AttentionSinkMask.__hash__: _AttentionSinkMask#__hash__().
  _AttentionSinkMask: _AttentionSinkMask#
---
# Module: [`ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py)

## Classes
### `_AttentionSinkMask`  ·  implements/extends _ComputableMask
- def: [`ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py:99`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L99)
- doc: Attention mask that allows attending to initial KV positions (attention sinks).
- signature: `class _AttentionSinkMask(mask_lib._ComputableMask):`
- members:
  - `__eq__(self, other: object)` — [`L129`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L129) — Check equality based on shape, sink size, and query sequence.
  - `__hash__(self)` — [`L139`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L139) — Compute hash from shape, sink size, and query sequence bytes.
  - `__init__(self, *, shape: tuple[int, int], attention_sink_size: int, shard_count: int = 1)` — [`L113`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L113) — Initialize an attention sink mask.
  - `sink_mask_function(q_ids: np.ndarray, kv_ids: np.ndarray)` — [`L124`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L124)
  - `attention_sink_size` — [`L111`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L111)
- used by: [`_build_mask`](_pallas_impl_fwd.md#_build_mask)

## Functions
- `_build_mask(q_seq_len: int, kv_seq_len: int, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, attention_sink_size: int = 0, chunk_size: int | None = None)` — [`L151`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L151) — Build a composite attention mask from attention parameters.
- `_make_block_sizes(fwd_params: FwdParams | None, bwd_params: BwdParams | None)` — [`L203`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L203) — Create BlockSizes from FwdParams and BwdParams.
- `ring_attention(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, q_position_ids: Int[Array, "batch seq_len_q"] | None = None, kv_position_ids: Int[Array, "batch seq_len_k"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = False, logits_soft_cap: float | None = None, softmax_scale: float | None = None, axis_name: str | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, fused_backward: bool = False)` — [`L245`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L245) — Computes ring attention using Splash Attention kernels on TPU.
- `single_batch_attention(q, k, v, seg_ids, sinks_batch)` — [`L426`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py#L426) — Process single batch element.

