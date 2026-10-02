---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90`/
symbols:
  ragged_dot_kernel: ragged_dot_kernel().
  _ragged_contracting_dim_dot_kernel_body: _ragged_contracting_dim_dot_kernel_body().
  ragged_contracting_dim_dot_kernel: ragged_contracting_dim_dot_kernel().
  ragged_dot_kernel.kernel: ragged_dot_kernel().kernel().
  ragged_dot_kernel.kernel.mn_loop_body.body.compute_acc: ragged_dot_kernel().kernel().mn_loop_body().body().compute_acc().
  _ragged_contracting_dim_dot_kernel_body.acc_scope: _ragged_contracting_dim_dot_kernel_body().acc_scope().
  ragged_dot_kernel.kernel.mn_loop_body: ragged_dot_kernel().kernel().mn_loop_body().
  ragged_dot_kernel.kernel.mn_loop_body.body: ragged_dot_kernel().kernel().mn_loop_body().body().
  _tiled_swizzled_smem: _tiled_swizzled_smem.
  ragged_dot_kernel.kernel.run_mn_loop: ragged_dot_kernel().kernel().run_mn_loop().
  _ragged_contracting_dim_dot_kernel_body.acc_scope.body: _ragged_contracting_dim_dot_kernel_body().acc_scope().body().
  _tiled_swizzled_block_spec: _tiled_swizzled_block_spec.
  _WGMMA: _WGMMA.
  ragged_dot_kernel.kernel.mn_loop_body.body.compute_acc.pipeline_body: ragged_dot_kernel().kernel().mn_loop_body().body().compute_acc().pipeline_body().
  ragged_dot_kernel.kernel.mn_loop_body.body.epilogue: ragged_dot_kernel().kernel().mn_loop_body().body().epilogue().
  _ragged_contracting_dim_dot_kernel_body.epilogue: _ragged_contracting_dim_dot_kernel_body().epilogue().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py)

## Functions
- `_ragged_contracting_dim_dot_kernel_body(group_sizes_gmem, group_sizes_starts_gmem, lhs_gmem, rhs_gmem, o_gmem, *, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L184`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L184) — Pallas kernel body for non-quantized ragged contracting dim dot.
- `acc_scope(acc)` — [`L206`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L206)
- `body()` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L84)
- `body(idxs, lhs_smem, rhs_smem)` — [`L207`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L207)
- `compute_acc(acc)` — [`L86`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L86)
- `epilogue(o_smem)` — [`L113`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L113)
- `epilogue(o_smem)` — [`L236`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L236)
- `kernel(lhs_gmem, rhs_gmem, group_id_gmem, start_within_block_gmem, actual_size_gmem, block_start_gmem, o_gmem)` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L59)
- `mn_loop_body(m_offset, m_iters, loop_info: plgpu.NDLoopInfo)` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L69)
- `pipeline_body(_, lhs_smem, rhs_smem)` — [`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L87)
- `ragged_contracting_dim_dot_kernel(lhs: Float[Array, "K M"], rhs: Float[Array, "K N"], group_sizes: Integer[Array, G], out_dtype: jnp.dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L247`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L247) — Pallas kernel for ragged contracting dim dot with non-quantized inputs.
- `ragged_dot_kernel(lhs: Float[Array, "M K"], rhs: Float[Array, "G K N"], group_sizes: Integer[Array, G], out_dtype: jnp.dtype, config: common.Config, activation: base.ActivationFunction | None = None)` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L37) — Pallas kernel for ragged dot with non-quantized inputs.
- `run_mn_loop(m_offset, m_iters)` — [`L141`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L141)

## Module values
- `_WGMMA` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L31)
- `_tiled_swizzled_block_spec` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L32)
- `_tiled_swizzled_smem` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L33)

