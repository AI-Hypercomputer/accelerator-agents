---
title: 'Module: jax/experimental/pallas/ops/tpu/matmul.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/matmul.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.matmul`/matmul
symbols:
  matmul: ().
  matmul_kernel.init: _kernel().init().
  matmul_kernel: _kernel().
---
# Module: [`jax/experimental/pallas/ops/tpu/matmul.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/matmul.py)

## Functions
- `init()` — [`L30`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/matmul.py#L30)
- `matmul(x: jax.Array, y: jax.Array, *, block_shape, block_k: int = 256, out_dtype: jnp.dtype | None = None, debug: bool = False)` — [`L46`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/matmul.py#L46)
- `matmul_kernel(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref)` — [`L28`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/matmul.py#L28)

