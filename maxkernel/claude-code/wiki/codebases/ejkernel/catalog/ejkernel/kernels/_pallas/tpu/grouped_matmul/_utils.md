---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmul._utils`/
symbols:
  tpu_generation: tpu_generation().
  supports_bfloat16_matmul: supports_bfloat16_matmul().
  select_input_dtype: select_input_dtype().
  assert_is_supported_dtype: assert_is_supported_dtype().
  is_tpu: is_tpu().
  tpu_kind: tpu_kind().
  _TPU_KIND_PATTERN: _TPU_KIND_PATTERN.
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py)

## Functions
- `assert_is_supported_dtype(dtype: DTypeLike)` — [`L124`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L124) — Validate that a dtype is supported for grouped matrix multiplication.
- `is_tpu()` — [`L30`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L30) — Check if the current JAX backend is running on TPU.
- `select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray)` — [`L151`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L151) — Select the optimal dtype for matrix multiplication inputs.
- `supports_bfloat16_matmul()` — [`L96`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L96) — Check if the current device supports bfloat16 matrix multiplication.
- `tpu_generation()` — [`L65`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L65) — Extract the generation number of the currently attached TPU.
- `tpu_kind()` — [`L45`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L45) — Query identification string for the currently attached TPU.

## Module values
- `_TPU_KIND_PATTERN` — [`L62`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_utils.py#L62)

