---
title: 'Module: ejkernel/xla_utils/utils.py'
type: catalog
provenance: extracted
module: ejkernel/xla_utils/utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.xla_utils.utils`/
symbols:
  identity_dtype_convert: identity_dtype_convert().
  prepare_chunk_indices: prepare_chunk_indices().
  prepare_chunk_offsets: prepare_chunk_offsets().
  prepare_position_ids: prepare_position_ids().
  prepare_lens: prepare_lens().
  prepare_cu_seqlens_from_mask: prepare_cu_seqlens_from_mask().
  prepare_sequence_ids: prepare_sequence_ids().
  prepare_token_indices: prepare_token_indices().
  cdiv: cdiv().
  prepare_lens_from_mask: prepare_lens_from_mask().
  identity_dtype_convert.identity_fn: identity_dtype_convert().identity_fn().
  identity_dtype_convert.identity_fn_fwd: identity_dtype_convert().identity_fn_fwd().
  identity_dtype_convert.identity_fn_bwd: identity_dtype_convert().identity_fn_bwd().
---
# Module: [`ejkernel/xla_utils/utils.py`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py)

## Functions
- `cdiv(a: Int[Array, ...], b: int)` — [`L52`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L52) — Compute ceiling division for integers in a JAX-compatible way.
- `identity_dtype_convert(dtype: jnp.dtype)` — [`L343`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L343) — Create an identity function that converts gradients to a specific dtype.
- `identity_fn(x)` — [`L365`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L365)
- `identity_fn_bwd(res, g)` — [`L371`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L371)
- `identity_fn_fwd(x)` — [`L368`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L368)
- `prepare_chunk_indices(cu_seqlens: Int[Array, num_seqs_plus_one], chunk_size: int)` — [`L265`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L265) — Generate (sequence_id, chunk_id) pairs for each chunk in the packed batch.
- `prepare_chunk_offsets(cu_seqlens: Int[Array, num_seqs_plus_one], chunk_size: int)` — [`L308`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L308) — Compute cumulative chunk offsets for packed sequences.
- `prepare_cu_seqlens_from_mask(mask: Bool[Array, "batch seq_len"], out_dtype: DTypeLike = jnp.int32)` — [`L133`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L133) — Create cumulative sequence lengths from a boolean attention mask.
- `prepare_lens(cu_seqlens: Int[Array, num_seqs_plus_one])` — [`L79`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L79) — Calculate individual sequence lengths from cumulative sequence lengths.
- `prepare_lens_from_mask(mask: Bool[Array, "batch seq_len"])` — [`L106`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L106) — Calculate sequence lengths from a boolean attention mask.
- `prepare_position_ids(cu_seqlens: Int[Array, num_seqs_plus_one])` — [`L166`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L166) — Generate position IDs for a batch of packed sequences.
- `prepare_sequence_ids(cu_seqlens: Int[Array, num_seqs_plus_one])` — [`L202`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L202) — Generate sequence IDs (0-indexed) for a batch of packed sequences.
- `prepare_token_indices(cu_seqlens: Int[Array, num_seqs_plus_one])` — [`L232`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/utils.py#L232) — Generate (sequence_id, position_id) pairs for each token in the packed batch.

