---
title: 'Module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.blocksparse_attention._interface`/
symbols:
  blocksparse_attention: blocksparse_attention().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_interface.py)

## Functions
- `blocksparse_attention(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], q_segment_ids: Int[Array, "batch seq_len"] | None = None, kv_segment_ids: Int[Array, "batch kv_len"] | None = None, q_positions: Int[Array, "batch seq_len"] | None = None, kv_positions: Int[Array, "batch kv_len"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None) = None, sequence_parallelism_mesh_axis_name: str | None = None, logits_soft_cap: float | None = None, qkv_layouts: tuple[SparseMask] | None = None, softmax_scale: float | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | Callable[[], SparseMask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = True, fused_backward: bool = False)` — [`L35`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_interface.py#L35) — Pallas TPU block-sparse (Splash) attention.

## Module values
- `__all__` — [`L156`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_interface.py#L156)

