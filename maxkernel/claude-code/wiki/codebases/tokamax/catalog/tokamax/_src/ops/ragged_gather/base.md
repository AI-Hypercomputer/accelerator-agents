---
title: 'Module: tokamax/_src/ops/ragged_gather/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_gather/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_gather.base`/
symbols:
  RaggedGather: RaggedGather#
  RaggedGather._fwd: RaggedGather#_fwd().
  RaggedGather.bind: RaggedGather#bind().
  _Config: _Config.
  ragged_gather: ragged_gather().
---
# Module: [`tokamax/_src/ops/ragged_gather/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/base.py)

## Classes
### `RaggedGather`  ·  implements/extends Op
- def: [`tokamax/_src/ops/ragged_gather/base.py:34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/base.py#L34)
- doc: Tokamax operator for Ragged Gather.
- signature: `class RaggedGather(op.Op[Any, jax.Array, None, _Config, Any]):`
- members:
  - `bind(self, x: Shaped[Array, "in_size hidden_size"], indices: Int[Array, out_size], start: Int[Array, 1], end: Int[Array, 1], *, return_residuals: bool = False)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/base.py#L38)
- protocol/private: `_fwd`[`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/base.py#L58)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`_fwd`](pallas_mosaic_tpu.md#PallasTpuRaggedGather._fwd), [`PallasTpuRaggedGather`](pallas_mosaic_tpu.md#PallasTpuRaggedGather), [`_Config`](base.md#_Config), [`ragged_gather`](base.md#ragged_gather)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`_fwd`](../op.md#Op._fwd), [`PallasTpuRaggedGather`](pallas_mosaic_tpu.md#PallasTpuRaggedGather), [`test_sc_gather`](pallas_mosaic_tpu_test.md#PallasTpuRaggedGatherTest.test_sc_gather), [`test_base_running_correctly`](base_test.md#BaseRaggedGatherTest.test_base_running_correctly)

## Functions
- `ragged_gather(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array)` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/base.py#L27)

## Module values
- `_Config` — [`L24`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/base.py#L24)

