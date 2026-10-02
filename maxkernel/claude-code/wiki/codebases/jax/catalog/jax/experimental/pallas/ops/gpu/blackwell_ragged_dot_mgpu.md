---
title: 'Module: jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu`/
symbols:
  do_matmul: do_matmul().
  ragged_dot_kernel: ragged_dot_kernel().
  main: main().
  ragged_dot_kernel.kernel._scoped.mn_loop: ragged_dot_kernel().kernel()._scoped().mn_loop().
  do_matmul._._per_warp._memory._loop_body: do_matmul()._()._per_warp()._memory()._loop_body().
  ragged_dot_kernel.kernel._scoped: ragged_dot_kernel().kernel()._scoped().
  sample_group_sizes: sample_group_sizes().
  do_matmul._._per_warp._compute._loop_body: do_matmul()._()._per_warp()._compute()._loop_body().
  do_matmul._._per_warp: do_matmul()._()._per_warp().
  do_matmul._._per_warp._compute: do_matmul()._()._per_warp()._compute().
  do_matmul._._per_warp._memory: do_matmul()._()._per_warp()._memory().
  do_matmul._._per_warp._wait_store: do_matmul()._()._per_warp()._wait_store().
  do_matmul._._clamp: do_matmul()._()._clamp().
  ragged_dot_reference: ragged_dot_reference().
  TuningConfig.grid_minor_dim: TuningConfig#grid_minor_dim.
  ragged_dot_kernel.kernel: ragged_dot_kernel().kernel().
  TuningConfig: TuningConfig#
  TuningConfig.tile_m: TuningConfig#tile_m.
  TuningConfig.tile_n: TuningConfig#tile_n.
  TuningConfig.tile_k: TuningConfig#tile_k.
  TuningConfig.max_concurrent_steps: TuningConfig#max_concurrent_steps.
  TuningConfig.collective: TuningConfig#collective.
  TuningConfig.grid_tile_width: TuningConfig#grid_tile_width.
  TuningConfig.epilogue_tile_n: TuningConfig#epilogue_tile_n.
  TuningConfig.__str__: TuningConfig#__str__().
---
# Module: [`jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py)

## Classes
### `TuningConfig`
- def: [`jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py:33`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L33)
- signature: `class TuningConfig:`
- members:
  - `collective` — [`L38`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L38)
  - `epilogue_tile_n` — [`L41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L41)
  - `grid_minor_dim` — [`L40`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L40)
  - `grid_tile_width` — [`L39`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L39)
  - `max_concurrent_steps` — [`L37`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L37)
  - `tile_k` — [`L36`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L36)
  - `tile_m` — [`L34`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L34)
  - `tile_n` — [`L35`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L35)
- protocol/private: `__str__`[`L43`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L43)
- uses (calls/refs, reference-scoped): [`MatmulDimension`](blackwell_matmul_mgpu.md#MatmulDimension)
- used by: [`do_matmul`](blackwell_ragged_dot_mgpu.md#do_matmul), [`ragged_dot_kernel`](blackwell_ragged_dot_mgpu.md#ragged_dot_kernel), [`main`](blackwell_ragged_dot_mgpu.md#main), [`mn_loop`](blackwell_ragged_dot_mgpu.md#ragged_dot_kernel.kernel._scoped.mn_loop)

## Functions
- `_clamp(min, x, max)` — [`L183`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L183)
- `_compute()` — [`L136`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L136)
- `_loop_body(ki, _)` — [`L105`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L105)
- `_loop_body(ki, _)` — [`L137`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L137)
- `_memory()` — [`L104`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L104)
- `_per_warp()` — [`L101`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L101)
- `_scoped(**ref_kwargs)` — [`L302`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L302)
- `_wait_store()` — [`L133`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L133)
- `do_matmul(a_gmem, b_gmem, out_gmem, grid_indices: Sequence[jax.Array], wg_axis: str, collective_axes: tuple[str, ...], local_index: jax.Array | int, config: TuningConfig, group_info: ragged_dot_mgpu.GroupInfo, a_smem, b_smem, acc_tmem, acc_smem, a_tma_barrier, b_tma_barrier, store_done_barrier, mma_done_barrier, consumed_barrier)` — [`L48`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L48) — Compute a non-ragged matmul for a single output block.
- `kernel(a_gmem, b_gmem, group_sizes_gmem, out_gmem)` — [`L265`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L265)
- `main(_)` — [`L378`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L378)
- `mn_loop(loop_info: plgpu.NDLoopInfo)` — [`L305`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L305)
- `ragged_dot_kernel(a, b, group_sizes, config: TuningConfig)` — [`L220`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L220)
- `ragged_dot_reference(a, b, g)` — [`L350`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L350)
- `sample_group_sizes(key: jax.Array, num_groups: int, num_elements: int, alpha: float = 10)` — [`L354`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py#L354) — Sample group sizes.

