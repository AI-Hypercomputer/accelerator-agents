---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_kl_divergence._pallas_impl_fwd`/
symbols:
  fused_kl_divergence_pallas: fused_kl_divergence_pallas().
  _kl_fwd_pallas: _kl_fwd_pallas().
  _kl_tp_lse_stats_pallas: _kl_tp_lse_stats_pallas().
  _kl_tp_acc_pallas: _kl_tp_acc_pallas().
  _pallas_out_shape: _pallas_out_shape().
  _kl_tp_loss_and_aux: _kl_tp_loss_and_aux().
  _pad_rows_2d: _pad_rows_2d().
  _copy_two_tiles: _copy_two_tiles().
  _pad_rows_1d: _pad_rows_1d().
  _copy_rows_hbm_to_vmem: _copy_rows_hbm_to_vmem().
  _fused_kl_core_pallas: _fused_kl_core_pallas().
  _fused_kl_core_pallas_tp: _fused_kl_core_pallas_tp().
  _kl_fwd_kernel: _kl_fwd_kernel().
  _kl_tp_lse_stats_kernel: _kl_tp_lse_stats_kernel().
  _kl_tp_acc_kernel: _kl_tp_acc_kernel().
  _kl_core_fwd: _kl_core_fwd().
  _kl_core_bwd: _kl_core_bwd().
  _kl_core_tp_fwd: _kl_core_tp_fwd().
  _kl_core_tp_bwd: _kl_core_tp_bwd().
  _kl_fwd_kernel._compute_active_block: _kl_fwd_kernel()._compute_active_block().
  _kl_tp_lse_stats_kernel._compute_active_block: _kl_tp_lse_stats_kernel()._compute_active_block().
  _kl_tp_acc_kernel._compute_active_block: _kl_tp_acc_kernel()._compute_active_block().
  _default_block_v: _default_block_v().
  _default_block_m: _default_block_m().
  _flatten_logits: _flatten_logits().
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py)

## Functions
- `_compute_active_block()` — [`L246`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L246) — Stream KL tiles only for blocks with nonzero row weights.
- `_compute_active_block()` — [`L467`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L467) — Compute local teacher/student LSE pieces for active TP rows.
- `_compute_active_block()` — [`L587`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L587) — Accumulate this shard's KL contribution using global LSEs.
- `_copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int)` — [`L152`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L152) — DMA-copy one row-vector block from HBM into VMEM scratch.
- `_copy_two_tiles(student_ref, teacher_ref, s_tile_ref, t_tile_ref, sem_ref, row_start, col_start: int, block_m: int, size: int)` — [`L104`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L104) — DMA-copy matching student and teacher vocab tiles into VMEM scratch.
- `_default_block_m()` — [`L54`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L54) — Return the row tile size used by the sparse-row TPU KL kernels.
- `_default_block_v(vocab_size: int)` — [`L39`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L39) — Choose the default TPU vocab tile width for KL forward kernels.
- `_flatten_logits(logits: jax.Array)` — [`L97`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L97) — Flatten leading dimensions into rows while preserving output shape.
- `_fused_kl_core_pallas(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m)` — [`L753`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L753) — Differentiable replicated-vocab KL loss wrapper.
- `_fused_kl_core_pallas_tp(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m, vocab_parallel_axis)` — [`L937`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L937) — Differentiable TP-vocab KL wrapper backed by Pallas kernels.
- `_kl_core_bwd(direction, temperature, block_v, block_m, residual, dy)` — [`L816`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L816) — Backward rule for replicated-vocab KL using saved softmax stats.
- `_kl_core_fwd(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m)` — [`L785`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L785) — Forward rule for replicated-vocab KL, saving LSEs and KL aux.
- `_kl_core_tp_bwd(direction, temperature, block_v, block_m, vocab_parallel_axis, residual, dy)` — [`L1014`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L1014) — Backward rule for TP-vocab KL.
- `_kl_core_tp_fwd(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m, vocab_parallel_axis)` — [`L980`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L980) — Forward rule for TP-vocab KL, saving global LSEs and full-row KL aux.
- `_kl_fwd_kernel(student_ref, teacher_ref, weights_ref, loss_ref, lse_t_ref, lse_s_ref, acc_ref, student_tile_ref, teacher_tile_ref, weight_ref, dma_sem_ref, *, direction: str, temperature: float, block_v: int, block_m: int)` — [`L174`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L174) — Replicated-vocab KL forward kernel for one row block.
- `_kl_fwd_pallas(student_2d, teacher_2d, weights_1d, *, direction, temperature, block_v, block_m)` — [`L332`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L332) — Launch replicated-vocab TPU Pallas KL forward and trim padded rows.
- `_kl_tp_acc_kernel(student_ref, teacher_ref, weights_ref, lse_t_ref, lse_s_ref, acc_ref, student_tile_ref, teacher_tile_ref, weight_ref, lse_t_scalar_ref, lse_s_scalar_ref, dma_sem_ref, *, direction: str, temperature: float, block_v: int, block_m: int)` — [`L520`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L520) — Compute the local KL contribution using global teacher/student LSEs.
- `_kl_tp_acc_pallas(student_2d, teacher_2d, weights_1d, lse_t, lse_s, *, direction, temperature, block_v, block_m)` — [`L682`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L682) — Launch the second TP-vocab KL phase that computes local KL mass.
- `_kl_tp_loss_and_aux(student_2d, teacher_2d, weights_1d, *, direction, temperature, block_v, block_m, vocab_parallel_axis)` — [`L857`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L857) — Build TP-vocab KL loss from local Pallas phases and TP collectives.
- `_kl_tp_lse_stats_kernel(student_ref, teacher_ref, weights_ref, max_t_ref, max_s_ref, sum_t_ref, sum_s_ref, student_tile_ref, teacher_tile_ref, weight_ref, dma_sem_ref, *, temperature: float, block_v: int, block_m: int)` — [`L401`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L401) — Compute local teacher/student softmax stats for a TP vocab shard.
- `_kl_tp_lse_stats_pallas(student_2d, teacher_2d, weights_1d, *, temperature, block_v, block_m)` — [`L617`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L617) — Launch the first TP-vocab KL phase that computes local LSE stats.
- `_pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L90) — Pad a rank-1 row tensor with the value expected by the caller.
- `_pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L83`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L83) — Pad a rank-2 tensor along rows so Pallas grid blocks are rectangular.
- `_pallas_out_shape(shape: tuple[int, ...], dtype: jnp.dtype)` — [`L59`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L59) — Build a Pallas output spec tagged as varying over the current manual axes.
- `fused_kl_divergence_pallas(student_logits, teacher_logits, weights=None, *, reduction: str = "mean", direction: str = "forward", temperature: float = 1, beta: float = 0.5, vocab_parallel_axis: str | None = None, block_v: int = 0, block_m: int = 0)` — [`L1059`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L1059) — Run TPU Pallas fused KL divergence between student and teacher logits.

