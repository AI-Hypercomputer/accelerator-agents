---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_i8_quant`/
symbols:
  ragged_dot_gpu_i8_quant_blackwell_kernel: ragged_dot_gpu_i8_quant_blackwell_kernel().
  _DEQ_WG: _DEQ_WG.
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._loop_body: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._loop_body().
  rescale_i32_tcgen05_acc.rescale: rescale_i32_tcgen05_acc().rescale().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop().
  _TCGEN05: _TCGEN05.
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp.loop_body: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().loop_body().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel_entry: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel_entry().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.mma_warp: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().mma_warp().
  _MAIN_WG: _MAIN_WG.
  _TCGEN05_COL: _TCGEN05_COL.
  _TCGEN05_ROW: _TCGEN05_ROW.
  _STORE_WG: _STORE_WG.
  rescale_i32_tcgen05_acc: rescale_i32_tcgen05_acc().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp.loop_body: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().loop_body().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.mma_warp.do_mma: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().mma_warp().do_mma().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._deq: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._deq().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._loop_body.load_ws_acc: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._loop_body().load_ws_acc().
  _TMEM: _TMEM.
  _MMA_WARP: _MMA_WARP.
  _W_TMA_WARP: _W_TMA_WARP.
  _X_TMA_WARP: _X_TMA_WARP.
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp.do_tma_w: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().do_tma_w().
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp.do_tma_x: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().do_tma_x().
  _SCALE_TMA_WARP: _SCALE_TMA_WARP.
  ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp: ragged_dot_gpu_i8_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py)

## Functions
- `_body()` — [`L294`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L294)
- `_deq(ki, _)` — [`L410`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L410)
- `_loop_body(ki, acc_carry)` — [`L468`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L468)
- `_per_warp(warp_id)` — [`L300`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L300)
- `do_mma(ki, _)` — [`L362`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L362)
- `do_tma_w(ki, ni, slot)` — [`L304`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L304)
- `do_tma_x(ki, slot)` — [`L333`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L333)
- `kernel(*refs, scoped)` — [`L234`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L234)
- `kernel_entry(*refs)` — [`L566`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L566)
- `load_ws_acc(ni)` — [`L482`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L482)
- `loop_body(ki, _)` — [`L315`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L315)
- `loop_body(ki, _)` — [`L347`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L347)
- `mma_warp()` — [`L361`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L361)
- `mn_loop(loop_info: plgpu.NDLoopInfo, carry)` — [`L272`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L272)
- `ragged_dot_gpu_i8_quant_blackwell_kernel(lhs: Float[qwix.QArray, "M K"], rhs: Float[qwix.QArray, "G K N"], group_sizes: Integer[Array, G], out_dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L106`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L106) — Returns the Pallas kernel for quantized ragged dot by using native int8 MMA.
- `rescale(_, running_acc: mgpu.FragmentedArray, a: mgpu.FragmentedArray, rs: mgpu.FragmentedArray, cs: mgpu.FragmentedArray)` — [`L63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L63)
- `rescale_i32_tcgen05_acc(running_acc, acc, row_scale, col_scale)` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L51) — Dequantizes S32 TCGEN05 accumulator and adds it to a running f32 accumulator.
- `w_tma_warp()` — [`L303`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L303)
- `x_tma_warp()` — [`L332`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L332)

## Module values
- `_DEQ_WG` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L35)
- `_MAIN_WG` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L37)
- `_MMA_WARP` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L41)
- `_SCALE_TMA_WARP` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L44)
- `_STORE_WG` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L39)
- `_TCGEN05` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L46)
- `_TCGEN05_COL` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L47)
- `_TCGEN05_ROW` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L48)
- `_TMEM` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L45)
- `_W_TMA_WARP` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L42)
- `_X_TMA_WARP` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant.py#L43)

