---
title: 'Module: jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.collective_matmul_mgpu`/
symbols:
  all_gather_lhs_matmul: all_gather_lhs_matmul().
  _run_example: _run_example().
  all_gather_lhs_matmul.kernel_body: all_gather_lhs_matmul().kernel_body().
  all_gather_lhs_matmul.kernel_body.device_step: all_gather_lhs_matmul().kernel_body().device_step().
  all_gather_lhs_matmul.kernel_body.send_lhs: all_gather_lhs_matmul().kernel_body().send_lhs().
  _run_example.build_kernel: _run_example().build_kernel().
  all_gather_lhs_matmul.kernel_body.device_step._signal: all_gather_lhs_matmul().kernel_body().device_step()._signal().
  _min_results_across_devices: _min_results_across_devices().
  MatmulDimension: MatmulDimension.
  all_gather_lhs_matmul.kernel_body.device_step._wait: all_gather_lhs_matmul().kernel_body().device_step()._wait().
  all_gather_lhs_matmul.kernel_body._device_loop: all_gather_lhs_matmul().kernel_body()._device_loop().
  TuningConfig: TuningConfig.
  is_nvshmem_used: is_nvshmem_used().
---
# Module: [`jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py)

## Functions
- `_device_loop(device_offset)` — [`L155`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L155)
- `_min_results_across_devices(kernels_ms: list[tuple[str, float]])` — [`L182`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L182)
- `_run_example()` — [`L200`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L200)
- `_signal()` — [`L141`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L141)
- `_wait()` — [`L149`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L149)
- `all_gather_lhs_matmul(lhs: jax.Array, rhs: jax.Array, axis_name, *, config: hopper_matmul_mgpu.TuningConfig, dtype: jnp.dtype = jnp.float16)` — [`L41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L41)
- `build_kernel(**kwargs)` — [`L243`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L243)
- `device_step(lhs_source_ref, device_offset)` — [`L109`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L109)
- `is_nvshmem_used()` — [`L35`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L35)
- `kernel_body(lhs_local_ref, rhs_ref, out_ref, scratch_ref)` — [`L89`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L89)
- `send_lhs(m_idx, n_idx, k_idx, a_smem, b_smem, send_ref, should_send)` — [`L96`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L96)

## Module values
- `MatmulDimension` — [`L31`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L31)
- `TuningConfig` — [`L32`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py#L32)

