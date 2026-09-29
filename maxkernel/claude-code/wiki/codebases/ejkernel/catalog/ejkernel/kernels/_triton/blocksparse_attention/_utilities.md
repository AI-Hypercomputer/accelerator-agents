---
title: 'Module: ejkernel/kernels/_triton/blocksparse_attention/_utilities.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_triton/blocksparse_attention/_utilities.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._triton.blocksparse_attention._utilities`/
symbols:
  pad_to_block_size: pad_to_block_size().
  attention_pack_with_static_shape: attention_pack_with_static_shape().
  attention_pack_from_cu_static: attention_pack_from_cu_static().
  attention_unpack_with_static_shape: attention_unpack_with_static_shape().
  PADDING_SEGMENT_ID: PADDING_SEGMENT_ID.
  attention_pack_from_cu_static.body_b: attention_pack_from_cu_static().body_b().
  attention_unpack_with_static_shape.body_b: attention_unpack_with_static_shape().body_b().
  calc_bias_strides: calc_bias_strides().
  attention_pack_with_static_shape.process_token: attention_pack_with_static_shape().process_token().
  attention_pack_from_cu_static.body_b.body_p: attention_pack_from_cu_static().body_b().body_p().
  attention_unpack_with_static_shape.body_b.body_p: attention_unpack_with_static_shape().body_b().body_p().
  padded_load: padded_load().
  make_segment_mask: make_segment_mask().
  make_causal_mask: make_causal_mask().
  make_sliding_window_mask: make_sliding_window_mask().
  basic_attention_refrence: basic_attention_refrence().
---
# Module: [`ejkernel/kernels/_triton/blocksparse_attention/_utilities.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py)

## Functions
- `attention_pack_from_cu_static(x: Float[Array, "batch seq_max num_heads head_dim"], cum_seqlens: Int[Array, batch_plus_one], max_tokens: int | None = None)` — [`L418`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L418) — Pack variable-length batch using cumulative sequence lengths.
- `attention_pack_with_static_shape(x: Float[Array, "batch seq_len num_heads head_dim"], attention_mask: Bool[Array, "batch seq_len"], max_tokens: int | None = None)` — [`L167`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L167) — Pack attention tensor by removing padding based on attention mask.
- `attention_unpack_with_static_shape(x: Float[Array, "1 max_tokens num_heads head_dim"], cum_seqlens: Int[Array, batch_plus_one], batch_size: int, seqlen: int)` — [`L490`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L490) — Unpack contiguous tensor back to padded batch format.
- `basic_attention_refrence(q: Float[Array, "batch seq_len_q num_heads head_dim"], k: Float[Array, "batch seq_len_k num_heads_kv head_dim"], v: Float[Array, "batch seq_len_k num_heads_kv head_dim"], attn_bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, query_padding_mask: Bool[Array, "batch seq_len_q"] | None = None, key_padding_mask: Bool[Array, "batch seq_len_k"] | None = None, dropout_prob: float = 0, dropout_key: jax.Array | None = None, window_size: tuple[int, int] = (-1, -1), causal: bool = False, softcap: float = 0)` — [`L313`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L313) — Reference implementation of attention for testing and validation.
- `body_b(b, out_acc)` — [`L450`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L450) — Pack tokens for a single batch element into the output buffer.
- `body_b(b, out_acc)` — [`L520`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L520) — Unpack tokens for a single batch element from the packed buffer.
- `body_p(p, acc)` — [`L466`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L466) — Copy position p from batch element b if within sequence length.
- `body_p(p, acc)` — [`L536`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L536) — Copy position p from packed buffer to batch element b if valid.
- `calc_bias_strides(bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None, batch: int, nheads_q: int, QSeq: int, KSeq: int)` — [`L103`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L103) — Calculate memory strides for bias tensor with broadcasting support.
- `make_causal_mask(q_positions, kv_positions, transposed: tl.constexpr)` — [`L263`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L263) — Create causal (autoregressive) attention mask.
- `make_segment_mask(q_segment_ids, kv_segment_ids, transposed: tl.constexpr)` — [`L240`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L240) — Create segment mask for packed variable-length sequences.
- `make_sliding_window_mask(q_positions, kv_positions, window_left: tl.constexpr, window_right: tl.constexpr, transposed: tl.constexpr)` — [`L285`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L285) — Create sliding window mask.
- `pad_to_block_size(inputs: Sequence[Array] | None, indexs: Array | None, segment_ids: Array | None, block_size: int, pos_fill_value: int, transposed_inputs: bool = False)` — [`L560`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L560) — Pad tensors to be evenly divisible by block size.
- `padded_load(ptrs, offs_a, offs_b, PA0: tl.constexpr, PA1: tl.constexpr, LA0: tl.constexpr, LA1: tl.constexpr)` — [`L52`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L52) — Load data from memory with optional padding for boundary conditions.
- `process_token(i, packed_acc)` — [`L215`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L215) — Copy a single token from the padded input to its packed position.

## Module values
- `PADDING_SEGMENT_ID` — [`L48`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_utilities.py#L48)

