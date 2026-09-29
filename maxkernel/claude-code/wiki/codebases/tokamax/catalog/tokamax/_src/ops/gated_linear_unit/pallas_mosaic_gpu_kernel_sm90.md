---
title: 'Module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.pallas_mosaic_gpu_kernel_sm90`/g
symbols:
  get_autotuning_configs: et_autotuning_configs().
  gated_linear_unit: ated_linear_unit().
  gated_linear_unit.kernel._pipeline_scope._mn_loop: ated_linear_unit().kernel()._pipeline_scope()._mn_loop().
  get_heuristics_config: et_heuristics_config().
  gated_linear_unit.kernel.get_pipeline: ated_linear_unit().kernel().get_pipeline().
  gated_linear_unit.kernel._pipeline_scope: ated_linear_unit().kernel()._pipeline_scope().
  gated_linear_unit.kernel: ated_linear_unit().kernel().
  gated_linear_unit.kernel._pipeline_scope._mn_loop.compute_context: ated_linear_unit().kernel()._pipeline_scope()._mn_loop().compute_context().
  gated_linear_unit.kernel._pipeline_scope._mn_loop.mma_body: ated_linear_unit().kernel()._pipeline_scope()._mn_loop().mma_body().
  gated_linear_unit.kernel._pipeline_scope._mn_loop.compute_context._acc_scope: ated_linear_unit().kernel()._pipeline_scope()._mn_loop().compute_context()._acc_scope().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py)

## Functions
- `_acc_scope(acc_ref)` — [`L228`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L228)
- `_mn_loop(loop_info)` — [`L199`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L199)
- `_pipeline_scope(pipeline_allocs)` — [`L194`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L194)
- `compute_context(eval_pipeline)` — [`L222`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L222)
- `gated_linear_unit(x: Float[Array, "*B M K"], weights: Float[Array, "K 2 N"], *, activation: Callable[[jax.Array], jax.Array], config: common.Config)` — [`L80`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L80) — Gated Linear Unit implementation for SM90.
- `get_autotuning_configs(ba: op.BoundArguments)` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L42) — Returns the autotuning configs for the Pallas:MGPU GLU SM90 kernel. — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
- `get_heuristics_config(ba: op.BoundArguments)` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L32)
- `get_pipeline(pipeline_body, compute_context)` — [`L153`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L153)
- `kernel(a_gmem, b_gmem, out_gmem, out_smem)` — [`L151`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L151)
- `mma_body(_, a_smem, b_smem, acc_ref)` — [`L253`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.py#L253)

