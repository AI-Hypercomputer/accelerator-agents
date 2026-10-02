---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100`/
symbols:
  ragged_dot_gpu_non_quant_blackwell_kernel: ragged_dot_gpu_non_quant_blackwell_kernel().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.epilogue_wg: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().epilogue_wg().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().
  _TCGEN05_TRANSPOSED: _TCGEN05_TRANSPOSED.
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.compute_wg: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().compute_wg().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.compute_wg.compute_warps.tma_warp: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().compute_wg().compute_warps().tma_warp().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.compute_wg.compute_warps.mma_warp: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().compute_wg().compute_warps().mma_warp().
  _COMPUTE_WG: _COMPUTE_WG.
  _MMA_WARP: _MMA_WARP.
  _TMA_WARP: _TMA_WARP.
  _EPILOGUE_WG: _EPILOGUE_WG.
  _TCGEN05: _TCGEN05.
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.compute_wg.compute_warps: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().compute_wg().compute_warps().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.compute_wg.compute_warps.tma_warp.tma_loop: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().compute_wg().compute_warps().tma_warp().tma_loop().
  ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop.compute.compute_wg.compute_warps.mma_warp.mma_loop_body: ragged_dot_gpu_non_quant_blackwell_kernel().kernel().mn_loop().compute().compute_wg().compute_warps().mma_warp().mma_loop_body().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py)

## Functions
- `compute()` — [`L100`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L100)
- `compute_warps()` — [`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L104)
- `compute_wg()` — [`L102`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L102)
- `epilogue_wg()` — [`L170`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L170)
- `kernel(x_gmem, w_gmem, group_id_gmem, start_within_block_gmem, actual_size_gmem, block_start_gmem, out_gmem, x_smem, w_smem, acc_smem, acc_tmem, xw_barrier, xw_consumed_barrier, acc_barrier, acc_consumed_barrier)` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L64)
- `mma_loop_body(ki)` — [`L151`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L151)
- `mma_warp()` — [`L141`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L141)
- `mn_loop(loop_info: plgpu.NDLoopInfo, carry)` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L84)
- `ragged_dot_gpu_non_quant_blackwell_kernel(lhs: Float[Array, "M K"], rhs: Float[Array, "G K N"], group_sizes: Integer[Array, G], out_dtype: jnp.dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L38) — Pallas kernel for ragged dot with GPU quantization.
- `tma_loop(ki)` — [`L112`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L112)
- `tma_warp()` — [`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L109)

## Module values
- `_COMPUTE_WG` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L28)
- `_EPILOGUE_WG` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L31)
- `_MMA_WARP` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L29)
- `_TCGEN05` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L33)
- `_TCGEN05_TRANSPOSED` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L34)
- `_TMA_WARP` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py#L30)

