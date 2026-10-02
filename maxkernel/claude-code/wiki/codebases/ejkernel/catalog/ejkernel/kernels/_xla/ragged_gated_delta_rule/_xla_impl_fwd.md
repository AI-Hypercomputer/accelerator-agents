---
title: 'Module: ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.ragged_gated_delta_rule._xla_impl_fwd`/
symbols:
  _F32: _F32.
  _ragged_gdr_chunked_prefill: _ragged_gdr_chunked_prefill().
  _P: _P.
  _ragged_gdr_decode_only: _ragged_gdr_decode_only().
  _ragged_gdr_decode_only.scan_fn: _ragged_gdr_decode_only().scan_fn().
  _ragged_gdr_chunked_prefill.scan_body: _ragged_gdr_chunked_prefill().scan_body().
  pack_inputs_single_stream._pack: pack_inputs_single_stream()._pack().
  ragged_gated_delta_rule_dispatch: ragged_gated_delta_rule_dispatch().
  pack_inputs_single_stream: pack_inputs_single_stream().
  ragged_gated_delta_rule_dispatch.decode_fn: ragged_gated_delta_rule_dispatch().decode_fn().
  ragged_gated_delta_rule_dispatch.prefill_fn: ragged_gated_delta_rule_dispatch().prefill_fn().
  _ragged_gdr_chunked_prefill.to_chunk: _ragged_gdr_chunked_prefill().to_chunk().
  _ragged_gdr_chunked_prefill.to_chunk_scalar: _ragged_gdr_chunked_prefill().to_chunk_scalar().
  _ragged_gdr_step: _ragged_gdr_step().
---
# Module: [`ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py)

## Functions
- `_pack(data, fill=0)` — [`L232`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L232)
- `_ragged_gdr_chunked_prefill(query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices, chunk_size, use_qk_l2norm, compute_dtype=jnp.bfloat16)` — [`L261`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L261) — Chunked prefill forward path for ragged GDR with mixed-length sequences.
- `_ragged_gdr_decode_only(query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices, use_qk_l2norm)` — [`L96`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L96) — Decode-only forward path for ragged GDR (all sequences have length 1).
- `_ragged_gdr_step(query, key, value, g, beta, state)` — [`L53`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L53) — Execute one step of the GDR recurrence for a single token.
- `decode_fn(_)` — [`L472`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L472)
- `pack_inputs_single_stream(query, key, value, g, beta, query_start_loc, chunk_size)` — [`L175`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L175) — Pack ragged sequences into a chunk-aligned contiguous stream.
- `prefill_fn(_)` — [`L486`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L486)
- `ragged_gated_delta_rule_dispatch(query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices, *, chunk_size=64, use_qk_l2norm=True)` — [`L428`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L428) — Dispatch ragged GDR to decode-only or chunked prefill path.
- `scan_body(h, args)` — [`L397`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L397)
- `scan_fn(carry, xs)` — [`L142`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L142)
- `to_chunk(x)` — [`L345`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L345)
- `to_chunk_scalar(x)` — [`L348`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L348)

## Module values
- `_F32` — [`L50`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L50)
- `_P` — [`L49`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.py#L49)

