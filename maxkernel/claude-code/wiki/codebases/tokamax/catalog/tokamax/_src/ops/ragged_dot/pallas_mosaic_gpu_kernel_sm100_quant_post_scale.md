---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_quant_post_scale`/
symbols:
  ragged_dot_gpu_quant_post_scale_blackwell_kernel: ragged_dot_gpu_quant_post_scale_blackwell_kernel().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body().
  dequant.scaled_w: dequant().scaled_w().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._loop_body: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._loop_body().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.mma_warp: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().mma_warp().
  _TMEM: _TMEM.
  _TCGEN05_ROW: _TCGEN05_ROW.
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp._iter_w: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp()._iter_w().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp._iter_x: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp()._iter_x().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._deq: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._deq().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel_entry: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel_entry().
  _TCGEN05: _TCGEN05.
  dequant: dequant().
  _MAIN_WG: _MAIN_WG.
  _DEQ_WG: _DEQ_WG.
  _STORE_WG: _STORE_WG.
  _MMA_WARP: _MMA_WARP.
  _W_TMA_WARP: _W_TMA_WARP.
  _X_TMA_WARP: _X_TMA_WARP.
  dequant.scaled_w.scale: dequant().scaled_w().scale().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.w_tma_warp.do_tma_w: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().w_tma_warp().do_tma_w().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.x_tma_warp.do_tma_x: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().x_tma_warp().do_tma_x().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp.mma_warp.do_mma: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().mma_warp().do_mma().
  ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop._body._._per_warp: ragged_dot_gpu_quant_post_scale_blackwell_kernel().kernel().mn_loop()._body()._()._per_warp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py)

## Functions
- `_body()` — [`L192`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L192)
- `_deq(ki, _)` — [`L291`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L291)
- `_iter_w(ki, _)` — [`L212`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L212)
- `_iter_x(ki, _)` — [`L239`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L239)
- `_loop_body(ki, acc)` — [`L333`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L333)
- `_per_warp()` — [`L196`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L196)
- `dequant(s_ref, w)` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L46) — Dequantize the array `w` using a 1D ref `s_ref`.
- `do_mma(ki, _)` — [`L253`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L253)
- `do_tma_w(ki, slot)` — [`L201`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L201)
- `do_tma_x(ki, slot)` — [`L225`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L225)
- `kernel(*refs, scoped)` — [`L139`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L139)
- `kernel_entry(*refs)` — [`L398`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L398)
- `mma_warp()` — [`L252`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L252)
- `mn_loop(loop_info: plgpu.NDLoopInfo, carry)` — [`L175`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L175)
- `ragged_dot_gpu_quant_post_scale_blackwell_kernel(lhs: Float[Array, "M K"], rhs: Float[qwix.QArray, "G K N"], group_sizes: Integer[Array, G], out_dtype: jnp.dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L64) — Pallas kernel for ragged dot with GPU quantization.
- `scale(w_val, idx)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L54)
- `scaled_w(_, s_smem, w)` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L53)
- `w_tma_warp()` — [`L200`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L200)
- `x_tma_warp()` — [`L224`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L224)

## Module values
- `_DEQ_WG` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L35)
- `_MAIN_WG` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L33)
- `_MMA_WARP` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L38)
- `_STORE_WG` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L36)
- `_TCGEN05` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L42)
- `_TCGEN05_ROW` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L43)
- `_TMEM` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L41)
- `_W_TMA_WARP` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L39)
- `_X_TMA_WARP` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_post_scale.py#L40)

