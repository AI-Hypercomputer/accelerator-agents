---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_cross_entropy._pallas_impl_fwd`/
symbols:
  fused_cross_entropy_pallas: fused_cross_entropy_pallas().
  _ce_fwd_pallas: _ce_fwd_pallas().
  _ce_tp_stats_pallas: _ce_tp_stats_pallas().
  _pallas_out_shape: _pallas_out_shape().
  _fused_ce_loss_pallas: _fused_ce_loss_pallas().
  _ce_tp_loss_and_lse: _ce_tp_loss_and_lse().
  _fused_ce_loss_pallas_tp: _fused_ce_loss_pallas_tp().
  _pad_rows_1d: _pad_rows_1d().
  _copy_rows_hbm_to_vmem: _copy_rows_hbm_to_vmem().
  _ce_fwd_kernel: _ce_fwd_kernel().
  _ce_tp_stats_kernel: _ce_tp_stats_kernel().
  _ce_loss_fwd: _ce_loss_fwd().
  _ce_loss_bwd: _ce_loss_bwd().
  _ce_loss_tp_fwd: _ce_loss_tp_fwd().
  _ce_loss_tp_bwd: _ce_loss_tp_bwd().
  _copy_hbm_to_vmem: _copy_hbm_to_vmem().
  _ce_fwd_kernel._compute_active_block: _ce_fwd_kernel()._compute_active_block().
  _ce_tp_stats_kernel._compute_active_block: _ce_tp_stats_kernel()._compute_active_block().
  _pad_rows_2d: _pad_rows_2d().
  _default_block_v: _default_block_v().
  _default_block_m: _default_block_m().
  _flatten_logits: _flatten_logits().
  _per_token_weights: _per_token_weights().
  _ce_correct_jax: _ce_correct_jax().
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py)

## Functions
- `_ce_correct_jax(logits_2d, targets_1d, weights_1d, ignore_index: int)` — [`L603`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L603) — Compute sparse per-row accuracy outside the loss Pallas kernel.
- `_ce_fwd_kernel(logits_ref, targets_ref, weights_ref, loss_ref, lse_ref, logits_tile_ref, target_ref, weight_ref, dma_sem_ref, *, ignore_index: int, label_smoothing: float, z_loss: float, normalizing_constant: float, block_v: int, block_m: int)` — [`L145`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L145) — Replicated-vocab CE forward kernel for one row block.
- `_ce_fwd_pallas(logits_2d, targets_1d, weights_1d, *, ignore_index, label_smoothing, z_loss, block_v, block_m)` — [`L233`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L233) — Launch replicated-vocab TPU Pallas CE forward and trim padded rows.
- `_ce_loss_bwd(ignore_index, label_smoothing, z_loss, block_v, block_m, residual, dy)` — [`L452`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L452) — Backward rule for replicated-vocab CE using the saved global LSE.
- `_ce_loss_fwd(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m)` — [`L437`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L437) — Forward rule for replicated-vocab CE, saving tensors for backward.
- `_ce_loss_tp_bwd(ignore_index, label_smoothing, z_loss, block_v, block_m, vocab_parallel_axis, residual, dy)` — [`L578`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L578) — Backward rule for TP-vocab CE.
- `_ce_loss_tp_fwd(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m, vocab_parallel_axis)` — [`L552`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L552) — Forward rule for TP-vocab CE, saving global LSE and local targets.
- `_ce_tp_loss_and_lse(logits_2d, targets_1d, weights_1d, *, ignore_index, label_smoothing, z_loss, block_v, block_m, vocab_parallel_axis)` — [`L473`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L473) — Build TP-vocab sparse CE loss from local Pallas stats plus collectives.
- `_ce_tp_stats_kernel(logits_ref, targets_ref, weights_ref, max_ref, sum_exp_ref, target_logit_ref, sum_logits_ref, logits_tile_ref, target_ref, weight_ref, dma_sem_ref, *, ignore_index: int, block_v: int, block_m: int)` — [`L286`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L286) — Compute local-vocab CE statistics for one row block on a TP shard.
- `_ce_tp_stats_pallas(logits_2d, targets_1d, weights_1d, *, ignore_index, block_v, block_m)` — [`L369`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L369) — Launch the local-vocab CE stats kernel used by the TP path.
- `_compute_active_block()` — [`L188`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L188) — Single streaming pass over vocab tiles (online softmax).
- `_compute_active_block()` — [`L329`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L329) — Collect local TP statistics for rows that are not masked out.
- `_copy_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, col_start: int, block_m: int, size: int)` — [`L123`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L123) — DMA-copy one ``(rows, vocab)`` tile from HBM into VMEM scratch.
- `_copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int)` — [`L134`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L134) — DMA-copy one row-vector block from HBM into VMEM scratch.
- `_default_block_m()` — [`L64`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L64) — Return the row tile size used by the sparse-row TPU CE kernels.
- `_default_block_v(vocab_size: int)` — [`L44`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L44) — Choose the default TPU vocab tile width for CE forward kernels.
- `_flatten_logits(logits: jax.Array)` — [`L109`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L109) — Flatten leading dimensions into rows while preserving output shape.
- `_fused_ce_loss_pallas(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m)` — [`L422`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L422) — Differentiable replicated-vocab CE loss wrapper.
- `_fused_ce_loss_pallas_tp(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m, vocab_parallel_axis)` — [`L526`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L526) — Differentiable TP-vocab CE loss wrapper backed by Pallas kernels.
- `_pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L102`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L102) — Pad a rank-1 row tensor with the value expected by the caller.
- `_pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0)` — [`L95`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L95) — Pad a rank-2 tensor along rows so Pallas grid blocks are rectangular.
- `_pallas_out_shape(shape: tuple[int, ...], dtype: jnp.dtype)` — [`L69`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L69) — Build a Pallas ``out_shape`` entry that varies over the active manual mesh axes.
- `_per_token_weights(targets: jax.Array, weights: jax.Array | None, ignore_index: int)` — [`L116`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L116) — Return float32 row weights, deriving them from ``ignore_index`` if absent.
- `fused_cross_entropy_pallas(logits, targets=None, weights=None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, soft_targets=None, reduction: str = "mean", vocab_parallel_axis: str | None = None, block_v: int = 0, block_m: int = 0)` — [`L610`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L610) — Run TPU Pallas fused sparse cross-entropy.

