---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_cross_entropy._pallas_impl_bwd`/_
symbols:
  _ce_bwd_pallas: ce_bwd_pallas().
  _pad_rows_1d: pad_rows_1d().
  _copy_rows_hbm_to_vmem: copy_rows_hbm_to_vmem().
  _pallas_out_shape_like: pallas_out_shape_like().
  _ce_bwd_kernel: ce_bwd_kernel().
  _pallas_out_shape: pallas_out_shape().
  _pad_rows_2d: pad_rows_2d().
  __all__: _all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py)

## Functions
- `_ce_bwd_kernel(logits_ref, lse_ref, targets_ref, weights_ref, dy_ref, dlogits_ref, logits_tile_ref, lse_scalar_ref, target_scalar_ref, weight_scalar_ref, dy_scalar_ref, dma_sem_ref, *, ignore_index: int, label_smoothing: float, z_loss: float, global_vocab_size: int, block_v: int, block_m: int)` — [`L141`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L141) — Write one cross-entropy gradient tile using global LSE and local target ids.
- `_ce_bwd_pallas(logits_2d, lse, targets_1d, weights_1d, dy, *, ignore_index, label_smoothing, z_loss, block_v, block_m, global_vocab_size=0)` — [`L254`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L254) — Launch the TPU Pallas cross-entropy backward kernel and return ``dlogits``.
- `_copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int)` — [`L115`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L115) — DMA-copy one contiguous row vector tile from HBM into VMEM scratch.
- `_pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L96`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L96) — Pad a per-row vector to match the padded logits row count.
- `_pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L79`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L79) — Pad a row-major logits/gradient matrix to a whole Pallas row block.
- `_pallas_out_shape(shape: tuple[int, ...], dtype: jnp.dtype)` — [`L45`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L45) — Build the ``out_shape`` for a Pallas call that is manual-sharding aware.
- `_pallas_out_shape_like(shape: tuple[int, ...], dtype: jnp.dtype, like: jax.Array)` — [`L69`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L69) — Build an output struct preserving an input value's manual-axis type.

## Module values
- `__all__` — [`L354`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L354)

