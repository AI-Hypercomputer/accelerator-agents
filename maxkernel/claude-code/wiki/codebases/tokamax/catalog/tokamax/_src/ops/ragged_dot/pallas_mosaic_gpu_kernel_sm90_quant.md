---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90_quant`/
symbols:
  ragged_dot_quantized_kernel: ragged_dot_quantized_kernel().
  _COMPUTE_WGS: _COMPUTE_WGS.
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.compute_wg.compute_acc.k_loop_body: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().compute_wg().compute_acc().k_loop_body().
  ragged_dot_quantized_kernel.kernel: ragged_dot_quantized_kernel().kernel().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.compute_wg: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().compute_wg().
  ragged_dot_quantized_kernel.kernel.mn_loop_body: ragged_dot_quantized_kernel().kernel().mn_loop_body().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.compute_wg.compute_acc: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().compute_wg().compute_acc().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.x_tma_warp.k_loop.cp_rec: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().x_tma_warp().k_loop().cp_rec().
  ragged_dot_quantized_kernel.kernel.run_mn_loop: ragged_dot_quantized_kernel().kernel().run_mn_loop().
  _TMA_WG: _TMA_WG.
  _WGMMA_UPCAST_4X: _WGMMA_UPCAST_4X.
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.compute_wg.compute_acc.k_loop_body.load_scales: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().compute_wg().compute_acc().k_loop_body().load_scales().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.w_tma_warp: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().w_tma_warp().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.x_tma_warp.k_loop: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().x_tma_warp().k_loop().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.o_tma_warp: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().o_tma_warp().
  _WGMMA: _WGMMA.
  _WGMMA_ROW: _WGMMA_ROW.
  ragged_dot_quantized_kernel.tiled_smem: ragged_dot_quantized_kernel().tiled_smem().
  _WGMMA_TRANSPOSED: _WGMMA_TRANSPOSED.
  _WGMMA_UPCAST_2X: _WGMMA_UPCAST_2X.
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.w_tma_warp.k_loop: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().w_tma_warp().k_loop().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.w_tma_warp.k_loop.wait_w_consumed: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().w_tma_warp().k_loop().wait_w_consumed().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.x_tma_warp: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().x_tma_warp().
  ragged_dot_quantized_kernel.kernel.mn_loop_body.body.tma_wg.tma_warps.x_tma_warp.k_loop.wait_x_consumed: ragged_dot_quantized_kernel().kernel().mn_loop_body().body().tma_wg().tma_warps().x_tma_warp().k_loop().wait_x_consumed().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py)

## Functions
- `body()` — [`L148`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L148)
- `compute_acc(acc)` — [`L155`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L155)
- `compute_wg()` — [`L151`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L151)
- `cp_rec(size)` — [`L252`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L252)
- `k_loop(ki)` — [`L217`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L217)
- `k_loop(ki)` — [`L240`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L240)
- `k_loop_body(ki, w_scales)` — [`L159`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L159)
- `kernel(x_gmem, w_gmem, w_scales_gmem, group_id_gmem, start_within_block_gmem, actual_size_gmem, block_start_gmem, out_gmem, *, x_smem, w_smem, w_scales_smem, o_smem, x_barrier, w_barrier, o_barrier, x_consumed_barrier, w_consumed_barrier, o_consumed_barrier)` — [`L108`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L108)
- `load_scales()` — [`L179`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L179)
- `mn_loop_body(m_offset, m_iters, loop_info: plgpu.NDLoopInfo, carry)` — [`L130`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L130)
- `o_tma_warp()` — [`L265`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L265)
- `ragged_dot_quantized_kernel(lhs: Float[Array, "M K"], rhs: Float[qwix.QArray, "G K N"], group_sizes: Integer[Array, G], out_dtype: jnp.dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L46) — Returns the Pallas kernel for quantized ragged dot.
- `run_mn_loop(m_offset, m_iters, carry=0)` — [`L299`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L299)
- `tiled_smem(*args)` — [`L318`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L318)
- `tma_warps(warp_id)` — [`L210`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L210)
- `tma_wg()` — [`L206`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L206)
- `w_tma_warp()` — [`L213`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L213)
- `wait_w_consumed()` — [`L222`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L222)
- `wait_x_consumed()` — [`L245`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L245)
- `x_tma_warp()` — [`L237`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L237)

## Module values
- `_COMPUTE_WGS` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L35)
- `_TMA_WG` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L36)
- `_WGMMA` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L38)
- `_WGMMA_ROW` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L39)
- `_WGMMA_TRANSPOSED` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L40)
- `_WGMMA_UPCAST_2X` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L41)
- `_WGMMA_UPCAST_4X` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90_quant.py#L42)

