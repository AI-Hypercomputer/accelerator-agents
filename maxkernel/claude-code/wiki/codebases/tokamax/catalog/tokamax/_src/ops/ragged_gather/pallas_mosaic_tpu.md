---
title: 'Module: tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_gather.pallas_mosaic_tpu`/
symbols:
  PallasTpuRaggedGather._fwd: PallasTpuRaggedGather#_fwd().
  PallasTpuRaggedGather: PallasTpuRaggedGather#
  _Config: _Config.
  PallasTpuRaggedGather.supported_on: PallasTpuRaggedGather#supported_on().
---
# Module: [`tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py)

## Classes
### `PallasTpuRaggedGather`  ·  implements/extends RaggedGather
- def: [`tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py:29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py#L29)
- doc: Tokamax operator invoking the Pallas kernel for Ragged Gather.
- signature: `class PallasTpuRaggedGather(base.RaggedGather[_Config]):`
- members:
  - `supported_on(self, device)` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py#L50)
- protocol/private: `_fwd`[`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py#L34)
- uses (calls/refs, reference-scoped): [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`RaggedGather`](base.md#RaggedGather), [`ragged_gather_pallas`](pallas_mosaic_tpu_kernel.md#ragged_gather_pallas), [`_Config`](pallas_mosaic_tpu.md#_Config)
- used by: [`supported_on`](../op.md#Op.supported_on), [`RaggedGather`](base.md#RaggedGather), [`_fwd`](base.md#RaggedGather._fwd), [`test_sc_gather`](pallas_mosaic_tpu_test.md#PallasTpuRaggedGatherTest.test_sc_gather)

## Module values
- `_Config` — [`L26`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu.py#L26)

