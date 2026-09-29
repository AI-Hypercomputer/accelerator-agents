---
title: 'Module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.pallas_mosaic_gpu_kernel_sm100`/
symbols:
  gated_linear_unit: gated_linear_unit().
  get_heuristics_config: get_heuristics_config().
  gated_linear_unit.kernel.mn_loop: gated_linear_unit().kernel().mn_loop().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._tma_warp._loop_body: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._tma_warp()._loop_body().
  get_autotuning_configs: get_autotuning_configs().
  gated_linear_unit.kernel: gated_linear_unit().kernel().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._tma_warp: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._tma_warp().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._mma_warp: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._mma_warp().
  gated_linear_unit.kernel.mn_loop._store_wg: gated_linear_unit().kernel().mn_loop()._store_wg().
  ACC_NUM_SLOTS: ACC_NUM_SLOTS.
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._tma_warp._loop_body._wait_barrier: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._tma_warp()._loop_body()._wait_barrier().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._mma_warp._loop_body: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._mma_warp()._loop_body().
  gated_linear_unit.kernel.mn_loop._compute_wg: gated_linear_unit().kernel().mn_loop()._compute_wg().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._wait_store: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._wait_store().
  gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._mma_warp._loop_body._arrive: gated_linear_unit().kernel().mn_loop()._compute_wg()._per_warp()._mma_warp()._loop_body()._arrive().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py)

## Functions
- `_arrive()` — [`L211`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L211)
- `_compute_wg()` — [`L142`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L142)
- `_loop_body(ki, _)` — [`L151`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L151)
- `_loop_body(ki, _)` — [`L192`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L192)
- `_mma_warp()` — [`L191`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L191)
- `_per_warp()` — [`L144`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L144)
- `_store_wg()` — [`L220`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L220)
- `_tma_warp()` — [`L150`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L150)
- `_wait_barrier(barrier)` — [`L156`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L156)
- `_wait_store()` — [`L187`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L187)
- `gated_linear_unit(x: Float[Array, "*B M K"], weights: Float[Array, "K 2 N"], *, activation: Callable[[jax.Array], jax.Array], config: common.Config)` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L49) — Gated Linear Unit implementation for SM100.
- `get_autotuning_configs(ba: op.BoundArguments)` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L43)
- `get_heuristics_config(ba: op.BoundArguments)` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L33)
- `kernel(a_gmem, b_gmem, out_gmem, a_smem, b0_smem, b1_smem, acc0_tmem, acc1_tmem, acc_smem, ab_tma_barrier, store_done_barrier, mma_done_barrier, consumed_barrier_0, consumed_barrier_1)` — [`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L97)
- `mn_loop(loop_info)` — [`L118`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L118)

## Module values
- `ACC_NUM_SLOTS` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm100.py#L30)

