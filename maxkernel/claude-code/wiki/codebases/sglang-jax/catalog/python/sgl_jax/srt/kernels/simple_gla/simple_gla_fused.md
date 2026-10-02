---
title: 'Module: python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.simple_gla.simple_gla_fused`/
symbols:
  _launch_decode_simple_gla: _launch_decode_simple_gla().
  decode_simple_gla_fused: decode_simple_gla_fused().
  _decode_simple_gla_kernel: _decode_simple_gla_kernel().
  _decode_simple_gla_kernel._buf_in_slice: _decode_simple_gla_kernel()._buf_in_slice().
  _decode_simple_gla_kernel._buf_out_slice: _decode_simple_gla_kernel()._buf_out_slice().
  _semaphore_kwargs: _semaphore_kwargs().
  _COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS: _COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS.
  _launch_decode_simple_gla.q_index_map: _launch_decode_simple_gla().q_index_map().
  _launch_decode_simple_gla.k_index_map: _launch_decode_simple_gla().k_index_map().
  _launch_decode_simple_gla.v_index_map: _launch_decode_simple_gla().v_index_map().
  _launch_decode_simple_gla.o_index_map: _launch_decode_simple_gla().o_index_map().
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py)

## Functions
- `_buf_in_slice(n_token: int)` — [`L120`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L120)
- `_buf_out_slice(n_token: int)` — [`L128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L128)
- `_decode_simple_gla_kernel(q_ref, k_ref, v_ref, g_gamma_ref, recurrent_indices_ref, has_initial_state_ref, recurrent_buffer_ref, o_ref, updated_recurrent_buffer_ref, h_in_buf, h_out_buf, sem_gather, sem_scatter, *, BK: int, BV: int, scale: float, N: int, H: int)` — [`L57`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L57) — Single Pallas program per (k_block, v_block); ALL tokens + ALL heads inside.
- `_launch_decode_simple_gla(q: jax.Array, k: jax.Array, v: jax.Array, g_gamma: jax.Array, recurrent_buffer: jax.Array, recurrent_indices: jax.Array, has_initial_state: jax.Array, *, scale: float)` — [`L216`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L216) — Launch the DECODE fused Pallas kernel.
- `_semaphore_kwargs(disable_semaphore_checks: bool)` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L50) — Forward `disable_semaphore_checks` only if the running jaxlib supports it.
- `decode_simple_gla_fused(q: jax.Array, k: jax.Array, v: jax.Array, recurrent_buffer: jax.Array, recurrent_indices: jax.Array, has_initial_state: jax.Array, *, g_gamma: jax.Array, scale: float | None = None)` — [`L342`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L342) — DECODE fused entry point. Returns (o, updated_recurrent_buffer).
- `k_index_map(k_i, _v_i)` — [`L268`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L268)
- `o_index_map(_k_i, v_i)` — [`L274`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L274)
- `q_index_map(k_i, _v_i)` — [`L265`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L265)
- `v_index_map(_k_i, v_i)` — [`L271`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L271)

## Module values
- `_COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS` — [`L45`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L45)
- `__all__` — [`L369`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L369)

