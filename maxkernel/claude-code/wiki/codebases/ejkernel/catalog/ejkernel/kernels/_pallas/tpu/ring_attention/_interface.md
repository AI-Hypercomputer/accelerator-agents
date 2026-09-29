---
title: 'Module: ejkernel/kernels/_pallas/tpu/ring_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ring_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ring_attention._interface`/
symbols:
  ring_attention: ring_attention().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/ring_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_interface.py)

## Functions
- `ring_attention(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, q_position_ids: Int[Array, "batch seq_len_q"] | None = None, kv_position_ids: Int[Array, "batch seq_len_k"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = False, logits_soft_cap: float | None = None, softmax_scale: float | None = None, axis_name: str | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, fused_backward: bool = False)` — [`L32`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_interface.py#L32) — Computes ring attention using Splash Attention kernels on TPU.

## Module values
- `__all__` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_interface.py#L125)

