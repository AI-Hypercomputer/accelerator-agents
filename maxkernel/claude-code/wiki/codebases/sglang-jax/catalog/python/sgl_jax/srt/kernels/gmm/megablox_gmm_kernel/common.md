---
title: 'Module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.common`/
symbols:
  tpu_generation: tpu_generation().
  supports_bfloat16_matmul: supports_bfloat16_matmul().
  select_input_dtype: select_input_dtype().
  assert_is_supported_dtype: assert_is_supported_dtype().
  is_tpu: is_tpu().
  tpu_kind: tpu_kind().
  _TPU_KIND_PATTERN: _TPU_KIND_PATTERN.
---
# Module: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py)

## Functions
- `assert_is_supported_dtype(dtype: jnp.dtype)` — [`L38`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L38)
- `is_tpu()` — [`L10`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L10)
- `select_input_dtype(lhs: jax.Array, rhs: jax.Array)` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L49) — A type to which both input should be adapted to before dot product.
- `supports_bfloat16_matmul()` — [`L33`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L33) — Does the currently attached CPU support bfloat16 inputs?
- `tpu_generation()` — [`L23`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L23) — Generation number of the currently attached TPU.
- `tpu_kind()` — [`L14`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L14) — Query identification string for the currently attached TPU.

## Module values
- `_TPU_KIND_PATTERN` — [`L20`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py#L20)

