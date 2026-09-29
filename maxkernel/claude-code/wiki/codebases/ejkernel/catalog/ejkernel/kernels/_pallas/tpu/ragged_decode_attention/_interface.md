---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_decode_attention._interface`/ragged_decode_attention().
symbols:
  ragged_decode_attention: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_interface.py)

## Functions
- `ragged_decode_attention(query: Float[Array, "batch num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], sequence_start: Int[Array, batch], sequence_end: Int[Array, batch], softmax_scale: float | None = None, fwd_params: FwdParams | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, ...] | None = None)` — [`L79`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_interface.py#L79) — Ragged MQA decoding entry point with TPU-accelerated Flash Attention.

