---
title: 'Module: jax/experimental/pallas/ops/tpu/megablox/ops.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/megablox/ops.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.megablox.ops`/
symbols:
  _gmm_fwd: _gmm_fwd().
  _gmm_bwd: _gmm_bwd().
  gmm: gmm.
---
# Module: [`jax/experimental/pallas/ops/tpu/megablox/ops.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/ops.py)

## Functions
- `_gmm_bwd(preferred_element_type: jnp.dtype, tiling: tuple[int, int, int], transpose_rhs: bool, interpret: bool, residual: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, int,], grad: jnp.ndarray)` — [`L63`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/ops.py#L63) — Backward function for throughput GMM VJP.
- `_gmm_fwd(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int] = (128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L28`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/ops.py#L28) — Forward function for GMM VJP.

## Module values
- `gmm` — [`L22`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/ops.py#L22)

