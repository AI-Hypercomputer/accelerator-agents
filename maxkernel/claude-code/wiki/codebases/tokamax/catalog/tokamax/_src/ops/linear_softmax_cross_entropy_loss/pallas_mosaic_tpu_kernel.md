---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel`/
symbols:
  linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu: linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu().
  linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu: linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu().
  linear_softmax_cross_entropy_loss_forward_pallas_kernel: linear_softmax_cross_entropy_loss_forward_pallas_kernel().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_xw: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_xw().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.init_w_grad: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().init_w_grad().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.init_x_grad: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().init_x_grad().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.accumulate_w_grad: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().accumulate_w_grad().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.accumulate_x_grad: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().accumulate_x_grad().
  validate_inputs: validate_inputs().
  calculate_xw_tiled: calculate_xw_tiled().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.perform_x_grad_tile_scaling: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().perform_x_grad_tile_scaling().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.perform_w_grad_tile_scaling: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().perform_w_grad_tile_scaling().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel: linear_softmax_cross_entropy_loss_backward_pallas_kernel().
  calculate_xw_tiled.pad_non_aligned_v_block: calculate_xw_tiled().pad_non_aligned_v_block().
  calculate_xw_tiled.pad_non_aligned_h_block: calculate_xw_tiled().pad_non_aligned_h_block().
  calculate_xw_tiled.init_xw: calculate_xw_tiled().init_xw().
  calculate_xw_tiled.accumulate_xw: calculate_xw_tiled().accumulate_xw().
  linear_softmax_cross_entropy_loss_forward_pallas_kernel.init_loss: linear_softmax_cross_entropy_loss_forward_pallas_kernel().init_loss().
  linear_softmax_cross_entropy_loss_forward_pallas_kernel.init_lse: linear_softmax_cross_entropy_loss_forward_pallas_kernel().init_lse().
  linear_softmax_cross_entropy_loss_forward_pallas_kernel.accumulate_loss: linear_softmax_cross_entropy_loss_forward_pallas_kernel().accumulate_loss().
  linear_softmax_cross_entropy_loss_forward_pallas_kernel.perform_loss_reduction: linear_softmax_cross_entropy_loss_forward_pallas_kernel().perform_loss_reduction().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.w_read: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().w_read().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.x_read: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().x_read().
  linear_softmax_cross_entropy_loss_backward_pallas_kernel.calculate_grads.compute_s: linear_softmax_cross_entropy_loss_backward_pallas_kernel().calculate_grads().compute_s().
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py)

## Functions
- `accumulate_loss()` — [`L200`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L200)
- `accumulate_w_grad()` — [`L539`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L539)
- `accumulate_x_grad()` — [`L550`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L550)
- `accumulate_xw()` — [`L120`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L120)
- `calculate_grads()` — [`L413`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L413)
- `calculate_xw()` — [`L396`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L396)
- `calculate_xw_tiled(x_ref, w_ref, xw_tiled, h_index, v_index, num_h_blocks, num_v_blocks, h_dim, v_dim, preferred_element_type: jnp.dtype)` — [`L66`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L66) — A kernel block for common logic between forward and backward kernel
- `compute_s()` — [`L510`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L510)
- `init_loss()` — [`L191`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L191)
- `init_lse()` — [`L195`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L195)
- `init_w_grad()` — [`L521`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L521)
- `init_x_grad()` — [`L530`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L530)
- `init_xw()` — [`L111`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L111)
- `linear_softmax_cross_entropy_loss_backward_pallas_kernel(dout, x_ref, labels_ref, w_ref, lse_ref, x_grad_hbm_ref, w_grad_hbm_ref, xw_scratch_ref, x_grad_tile_ref, w_grad_tile_ref, x_read_sem, w_read_sem, x_write_sem, w_write_sem, reduction: Literal["sum", "mean"], preferred_element_type: jnp.dtype)` — [`L348`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L348) — Pallas kernel for the backward pass of Linear Softmax Cross-Entropy Loss.
- `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(dout: Real[Array, ""], lse: Real[Array, B], x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, b_block_size: int = 1024, h_block_size: int = 512, v_block_size: int = 2048, reduction: Literal["sum", "mean"] = "sum", preferred_element_type: jnp.dtype = jnp.float32)` — [`L574`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L574) — The pallas kernel implementation of the Linear Softmax Cross-Entropy Loss backward kernel.
- `linear_softmax_cross_entropy_loss_forward_pallas_kernel(x_ref, labels_ref, w_ref, loss_ref, lse_ref, xw_tiled, b_block_loss_ref, reduction: Literal["sum", "mean"], h_dim: int, v_dim: int, preferred_element_type: jnp.dtype)` — [`L129`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L129) — Pallas kernel for the forward pass of Linear Softmax Cross-Entropy Loss.
- `linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, b_block_size: int = 1024, h_block_size: int = 512, v_block_size: int = 2048, reduction: Literal["sum", "mean"] = "sum", preferred_element_type: jnp.dtype = jnp.float32)` — [`L238`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L238) — The pallas kernel implementation of linear softmax cross-entropy loss.
- `pad_non_aligned_h_block()` — [`L98`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L98)
- `pad_non_aligned_v_block()` — [`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L87)
- `perform_loss_reduction()` — [`L215`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L215)
- `perform_w_grad_tile_scaling()` — [`L467`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L467)
- `perform_x_grad_tile_scaling()` — [`L452`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L452)
- `validate_inputs(x: Real[Array, "B H"], labels: Real[Array, "B V"], w: Real[Array, "H V"], b_block_size: int, h_block_size: int, v_block_size: int)` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L28) — Validates the inputs to the kernels.
- `w_read()` — [`L501`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L501)
- `x_read()` — [`L505`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py#L505)

