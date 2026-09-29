---
title: 'Module: src/maxtext/kernels/megablox/common.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/megablox/common.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.megablox.common`/
symbols:
  tpu_generation: tpu_generation().
  supports_bfloat16_matmul: supports_bfloat16_matmul().
  select_input_dtype: select_input_dtype().
  is_tpu: is_tpu().
  tpu_kind: tpu_kind().
  _TPU_KIND_PATTERN: _TPU_KIND_PATTERN.
  assert_is_supported_dtype: assert_is_supported_dtype().
---
# Module: [`src/maxtext/kernels/megablox/common.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py)

## Functions
- `assert_is_supported_dtype(dtype: jnp.dtype)` — [`L49`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L49)
- `is_tpu()` — [`L23`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L23)
- `select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray)` — [`L54`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L54) — A type to which both input should be adapted to before dot product.
- `supports_bfloat16_matmul()` — [`L44`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L44) — Does the currently attached CPU support bfloat16 inputs?
- `tpu_generation()` — [`L36`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L36) — Generation number of the currently attached TPU.
- `tpu_kind()` — [`L27`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L27) — Query identification string for the currently attached TPU.

## Module values
- `_TPU_KIND_PATTERN` — [`L33`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/common.py#L33)

