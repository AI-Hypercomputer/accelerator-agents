---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_quant`/
symbols:
  ragged_dot_gpu_quant_blackwell_kernel: ragged_dot_gpu_quant_blackwell_kernel().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body().
  dequant.scaled_w: dequant().scaled_w().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._deq: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._deq().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.mma_warp: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().mma_warp().
  _TMEM: _TMEM.
  dequant: dequant().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp._iter_w: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp()._iter_w().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp._iter_x: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp()._iter_x().
  ragged_dot_gpu_quant_blackwell_kernel.kernel_entry: ragged_dot_gpu_quant_blackwell_kernel().kernel_entry().
  _MAIN_WG: _MAIN_WG.
  _DEQ_WG: _DEQ_WG.
  _STORE_WG: _STORE_WG.
  _MMA_WARP: _MMA_WARP.
  _W_TMA_WARP: _W_TMA_WARP.
  _X_TMA_WARP: _X_TMA_WARP.
  dequant.scaled_w.scale: dequant().scaled_w().scale().
  ragged_dot_gpu_quant_blackwell_kernel.kernel: ragged_dot_gpu_quant_blackwell_kernel().kernel().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp.do_tma_w: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().do_tma_w().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp.do_tma_x: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().do_tma_x().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp.mma_warp.do_mma: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().mma_warp().do_mma().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._.tmem_to_smem_loop_body: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._().tmem_to_smem_loop_body().
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._.store_loop_body: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._().store_loop_body().
  _STORE_WARP: _STORE_WARP.
  ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop._body._._per_warp: ragged_dot_gpu_quant_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py)

## Functions
- `_body()` — [`L211`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L211)
- `_deq(ki, _)` — [`L320`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L320)
- `_iter_w(ki, _)` — [`L240`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L240)
- `_iter_x(ki, _)` — [`L267`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L267)
- `_per_warp()` — [`L215`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L215)
- `dequant(s_ref, w)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L47) — Dequantize the array `w` using a 1D ref `s_ref`.
- `do_mma(ki, _)` — [`L281`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L281)
- `do_tma_w(ki, slot)` — [`L220`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L220)
- `do_tma_x(ki, slot)` — [`L253`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L253)
- `kernel(*refs, scoped)` — [`L157`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L157)
- `kernel_entry(*refs)` — [`L433`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L433)
- `mma_warp()` — [`L280`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L280)
- `mn_loop(loop_info: plgpu.NDLoopInfo, carry)` — [`L194`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L194)
- `ragged_dot_gpu_quant_blackwell_kernel(lhs: Float[Array, "M K"], rhs: Float[qwix.QArray, "G K N"], group_sizes: Integer[Array, G], out_dtype: jnp.dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L65) — Pallas kernel for ragged dot with GPU quantization.
- `scale(w_val, idx)` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L55)
- `scaled_w(_, s_smem, w)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L54)
- `store_loop_body(index, _)` — [`L392`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L392)
- `tmem_to_smem_loop_body(mi, _)` — [`L365`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L365)
- `w_tma_warp()` — [`L219`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L219)
- `x_tma_warp()` — [`L252`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L252)

## Module values
- `_DEQ_WG` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L37)
- `_MAIN_WG` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L35)
- `_MMA_WARP` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L40)
- `_STORE_WARP` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L43)
- `_STORE_WG` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L38)
- `_TMEM` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L44)
- `_W_TMA_WARP` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L41)
- `_X_TMA_WARP` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant.py#L42)

