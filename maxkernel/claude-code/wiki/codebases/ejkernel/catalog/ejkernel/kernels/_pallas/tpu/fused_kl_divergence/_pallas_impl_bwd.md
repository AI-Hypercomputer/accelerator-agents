---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_kl_divergence._pallas_impl_bwd`/_
symbols:
  _kl_bwd_pallas: kl_bwd_pallas().
  _kl_bwd_kernel: kl_bwd_kernel().
  _pad_rows_1d: pad_rows_1d().
  _copy_rows_hbm_to_vmem: copy_rows_hbm_to_vmem().
  _pad_rows_2d: pad_rows_2d().
  _pallas_out_shape: pallas_out_shape().
  _copy_two_tiles: copy_two_tiles().
  __all__: _all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py)

## Functions
- `_copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int)` — [`L161`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L161) — DMA-copy one contiguous row vector tile from HBM into VMEM scratch.
- `_copy_two_tiles(student_ref, teacher_ref, s_tile_ref, t_tile_ref, sem_ref, row_start, col_start: int, block_m: int, size: int)` — [`L112`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L112) — DMA-copy aligned student and teacher vocab tiles into VMEM scratch.
- `_kl_bwd_kernel(student_ref, teacher_ref, lse_t_ref, lse_s_ref, acc_ref, weights_ref, dy_ref, dstudent_ref, student_tile_ref, teacher_tile_ref, lse_t_scalar_ref, lse_s_scalar_ref, acc_scalar_ref, weight_scalar_ref, dy_scalar_ref, dma_sem_ref, *, direction: str, temperature: float, block_v: int, block_m: int)` — [`L187`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L187) — Write one student-gradient tile for fused KL divergence.
- `_kl_bwd_pallas(student_2d, teacher_2d, lse_t, lse_s, acc, weights_1d, dy, *, direction, temperature, block_v, block_m)` — [`L292`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L292) — Launch the TPU Pallas KL backward kernel and return the student gradient.
- `_pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L95`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L95) — Pad a per-row vector to match the padded logit row count.
- `_pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L78`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L78) — Pad a row-major logit/gradient matrix to a whole Pallas row block.
- `_pallas_out_shape(shape: tuple[int, ...], dtype: jnp.dtype)` — [`L54`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L54) — Build the ``out_shape`` for a Pallas call that is manual-sharding aware.

## Module values
- `__all__` — [`L394`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L394)

