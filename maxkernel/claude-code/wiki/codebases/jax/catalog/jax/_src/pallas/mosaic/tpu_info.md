---
title: 'Module: jax/_src/pallas/mosaic/tpu_info.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/tpu_info.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.tpu_info`/
symbols:
  _get_tpu_info_impl: _get_tpu_info_impl().
  ChipVersion: ChipVersion#
  TpuInfo.is_matmul_supported: TpuInfo#is_matmul_supported().
  ChipVersion.num_physical_tensor_cores_per_chip: ChipVersion#num_physical_tensor_cores_per_chip().
  chip_version_from_device_kind: chip_version_from_device_kind().
  get_tpu_info: get_tpu_info().
  infer_tiling: infer_tiling().
  get_tpu_info_for_chip: get_tpu_info_for_chip().
  TpuInfo.generation: TpuInfo#generation.
  TpuInfo.chip_version: TpuInfo#chip_version.
  SparseCoreInfo.num_lanes: SparseCoreInfo#num_lanes.
  TpuInfo: TpuInfo#
  TpuInfo.num_sublanes: TpuInfo#num_sublanes.
  Tiling.shape: Tiling#shape().
  ChipVersion.is_lite: ChipVersion#is_lite().
  TpuInfo.num_cores: TpuInfo#num_cores.
  SparseCoreInfo.num_cores: SparseCoreInfo#num_cores.
  TpuInfo.num_lanes: TpuInfo#num_lanes.
  TpuInfo.is_split_chip: TpuInfo#is_split_chip().
  TpuInfo.get_sublane_tiling: TpuInfo#get_sublane_tiling().
  TpuInfo.sparse_core: TpuInfo#sparse_core.
  TpuInfo.mxu_column_size: TpuInfo#mxu_column_size.
  TpuInfo.num_mxus: TpuInfo#num_mxus.
  TpuInfo.num_accumulators: TpuInfo#num_accumulators.
  TpuInfo.vmem_capacity_bytes: TpuInfo#vmem_capacity_bytes.
  TpuInfo.cmem_capacity_bytes: TpuInfo#cmem_capacity_bytes.
  TpuInfo.smem_capacity_bytes: TpuInfo#smem_capacity_bytes.
  TpuInfo.hbm_capacity_bytes: TpuInfo#hbm_capacity_bytes.
  TpuInfo.mem_bw_bytes_per_second: TpuInfo#mem_bw_bytes_per_second.
  TpuInfo.bf16_ops_per_second: TpuInfo#bf16_ops_per_second.
  TpuInfo.int8_ops_per_second: TpuInfo#int8_ops_per_second.
  TpuInfo.fp8_ops_per_second: TpuInfo#fp8_ops_per_second.
  TpuInfo.int4_ops_per_second: TpuInfo#int4_ops_per_second.
  Tiling: Tiling#
  get_device_kind: get_device_kind().
  ChipVersion.supports_megacore: ChipVersion#supports_megacore().
  TpuInfo.is_lite: TpuInfo#is_lite().
  SparseCoreInfo: SparseCoreInfo#
  SparseCoreInfo.num_subcores: SparseCoreInfo#num_subcores.
  is_tpu_device: is_tpu_device().
  _get_tiling_factor: _get_tiling_factor().
  get_num_device_cores: get_num_device_cores().
  ChipVersionBase: ChipVersionBase#
  registry.registry: registry.registry.
  ChipVersion.TPU_V2: ChipVersion#TPU_V2.
  ChipVersion.TPU_V3: ChipVersion#TPU_V3.
  ChipVersion.TPU_V4I: ChipVersion#TPU_V4I.
  ChipVersion.TPU_V4: ChipVersion#TPU_V4.
  ChipVersion.TPU_V5E: ChipVersion#TPU_V5E.
  ChipVersion.TPU_V5P: ChipVersion#TPU_V5P.
  ChipVersion.TPU_V6E: ChipVersion#TPU_V6E.
  ChipVersion.TPU_7: ChipVersion#TPU_7.
  ChipVersion.TPU_7X: ChipVersion#TPU_7X.
  ChipVersion.TPU_8I: ChipVersion#TPU_8I.
  SparseCoreInfo.vmem_capacity_bytes: SparseCoreInfo#vmem_capacity_bytes.
  SparseCoreInfo.dma_granule_size_bytes: SparseCoreInfo#dma_granule_size_bytes.
  Tiling.COMPACT: Tiling#COMPACT.
  Tiling.SPARSE_CORE: Tiling#SPARSE_CORE.
  TpuInfo.is_megacore: TpuInfo#is_megacore().
  ChipVersion.__str__: ChipVersion#__str__().
---
# Module: [`jax/_src/pallas/mosaic/tpu_info.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py)

## Classes
### `ChipVersion`  ·  implements/extends ChipVersionBase, Enum
- def: [`jax/_src/pallas/mosaic/tpu_info.py:34`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L34)
- doc: TPU chip version.
- signature: `class ChipVersion(ChipVersionBase, enum.Enum):`
- members:
  - `is_lite(self)` — [`L103`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L103)
  - `num_physical_tensor_cores_per_chip(self)` — [`L79`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L79)
  - `supports_megacore(self)` — [`L95`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L95)
  - `TPU_7` — [`L71`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L71)
  - `TPU_7X` — [`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L72)
  - `TPU_8I` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L73)
  - `TPU_V2` — [`L64`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L64)
  - `TPU_V3` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L65)
  - `TPU_V4` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L67)
  - `TPU_V4I` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L66)
  - `TPU_V5E` — [`L68`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L68)
  - `TPU_V5P` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L69)
  - `TPU_V6E` — [`L70`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L70)
- protocol/private: `__str__`[`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L75)
- uses (calls/refs, reference-scoped): [`ChipVersionBase`](tpu_info.md#ChipVersionBase)
- used by: [`_get_tpu_info_impl`](tpu_info.md#_get_tpu_info_impl), [`chip_version_from_device_kind`](tpu_info.md#chip_version_from_device_kind), [`get_tpu_info_for_chip`](tpu_info.md#get_tpu_info_for_chip), [`is_split_chip`](tpu_info.md#TpuInfo.is_split_chip), [`is_lite`](tpu_info.md#TpuInfo.is_lite), [`ChipVersionBase`](tpu_info.md#ChipVersionBase)

### `ChipVersionBase`
- def: [`jax/_src/pallas/mosaic/tpu_info.py:30`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L30)
- signature: `class ChipVersionBase:`
- uses (calls/refs, reference-scoped): [`ChipVersion`](tpu_info.md#ChipVersion)
- used by: [`ChipVersion`](tpu_info.md#ChipVersion), [`chip_version`](tpu_info.md#TpuInfo.chip_version)

### `SparseCoreInfo`
- def: [`jax/_src/pallas/mosaic/tpu_info.py:138`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L138)
- doc: SparseCore-specific information.
- signature: `class SparseCoreInfo:`
- members:
  - `dma_granule_size_bytes` — [`L145`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L145)
  - `num_cores` — [`L141`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L141)
  - `num_lanes` — [`L143`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L143)
  - `num_subcores` — [`L142`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L142)
  - `vmem_capacity_bytes` — [`L144`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L144)
- used by: [`_get_tpu_info_impl`](tpu_info.md#_get_tpu_info_impl), [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`new_body`](pallas_call_registration.md#_rewrite_jaxpr_for_lowering.new_body), [`_iota_lowering_rule_sc`](sc_lowering.md#_iota_lowering_rule_sc), [`_masked_sort_lowering_rule`](sc_primitives.md#_masked_sort_lowering_rule), [`_sort_lowering_rule`](sc_lowering.md#_sort_lowering_rule), [`get_sparse_core_info`](sc_core.md#get_sparse_core_info), [`_masked_sort_abstract_eval`](sc_primitives.md#_masked_sort_abstract_eval), [`_rev_lowering_rule`](sc_lowering.md#_rev_lowering_rule), [`__post_init__`](sc_core.md#VectorSubcoreMesh.__post_init__), [`num_cores`](sc_core.md#ScalarSubcoreMesh.num_cores), [`num_cores`](sc_core.md#VectorSubcoreMesh.num_cores), [`supported_shapes`](sc_core.md#supported_shapes), [`sparse_core`](tpu_info.md#TpuInfo.sparse_core), [`num_subcores`](sc_core.md#VectorSubcoreMesh.num_subcores), [`__post_init__`](sc_core.md#ScalarSubcoreMesh.__post_init__)

### `Tiling`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/tpu_info.py:565`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L565)
- signature: `class Tiling(enum.Enum):`
- members:
  - `shape(self)` — [`L570`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L570)
  - `COMPACT` — [`L566`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L566)
  - `SPARSE_CORE` — [`L567`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L567)
- used by: [`create`](pipeline.md#BufferedRef.create), [`infer_tiling`](tpu_info.md#infer_tiling), [`Tiling`](pipeline.md#Tiling)

### `TpuInfo`
- def: [`jax/_src/pallas/mosaic/tpu_info.py:149`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L149)
- doc: TPU hardware information.
- signature: `class TpuInfo:`
- members:
  - `get_sublane_tiling(self, dtype: jnp.dtype)` — [`L251`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L251) — Returns the sublane tiling for the given itemsize.
  - `is_lite(self)` — [`L180`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L180)
  - `is_matmul_supported(self, lhs_dtype: dtypes.DTypeLike, rhs_dtype: dtypes.DTypeLike)` — [`L210`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L210) — Returns whether the chip natively supports matmul on the given input dtypes (no casting needed).
  - `is_megacore(self)` — [`L202`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L202) — Returns True if the chip is configured in Megacore mode.
  - `is_split_chip(self)` — [`L184`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L184) — Returns True if the chip is a multi-core chip being used in single-core mode.
  - `bf16_ops_per_second` — [`L172`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L172)
  - `chip_version` — [`L156`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L156)
  - `cmem_capacity_bytes` — [`L168`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L168)
  - `fp8_ops_per_second` — [`L174`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L174)
  - `generation` — [`L157`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L157)
  - `hbm_capacity_bytes` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L170)
  - `int4_ops_per_second` — [`L175`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L175)
  - `int8_ops_per_second` — [`L173`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L173)
  - `mem_bw_bytes_per_second` — [`L171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L171)
  - `mxu_column_size` — [`L161`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L161)
  - `num_accumulators` — [`L166`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L166)
  - `num_cores` — [`L158`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L158)
  - `num_lanes` — [`L159`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L159)
  - `num_mxus` — [`L163`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L163)
  - `num_sublanes` — [`L160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L160)
  - `smem_capacity_bytes` — [`L169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L169)
  - `sparse_core` — [`L177`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L177)
  - `vmem_capacity_bytes` — [`L167`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L167)
- uses (calls/refs, reference-scoped): [`float32`](../../../numpy/__init__.pyi.md#float32.float32), [`DTypeLike`](../../typing.md#DTypeLike), [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`itemsize_bits`](../../dtypes.md#itemsize_bits), [`dtype`](../../dtypes.md#dtype), [`ChipVersion`](tpu_info.md#ChipVersion), [`bfloat16`](../../../numpy/__init__.pyi.md#bfloat16.bfloat16), [`num_physical_tensor_cores_per_chip`](tpu_info.md#ChipVersion.num_physical_tensor_cores_per_chip), [`is_lite`](tpu_info.md#ChipVersion.is_lite), [`int8`](../../../numpy/__init__.pyi.md#int8.int8), [`SparseCoreInfo`](tpu_info.md#SparseCoreInfo), [`ChipVersionBase`](tpu_info.md#ChipVersionBase), [`int4`](../../../numpy/__init__.pyi.md#int4.int4), [`uint4`](../../../numpy/__init__.pyi.md#uint4.uint4), [`uint8`](../../../numpy/__init__.pyi.md#uint8.uint8), [`float8_e4m3b11fnuz`](../../../numpy/__init__.pyi.md#float8_e4m3b11fnuz.float8_e4m3b11fnuz), [`float8_e4m3fn`](../../../numpy/__init__.pyi.md#float8_e4m3fn.float8_e4m3fn), [`float8_e5m2`](../../../numpy/__init__.pyi.md#float8_e5m2.float8_e5m2)
- used by: [`_get_tpu_info_impl`](tpu_info.md#_get_tpu_info_impl), [`_check_block_mappings`](lowering.md#_check_block_mappings), [`mpmd_map_tpu_lowering_rule`](pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`get_tpu_info`](tpu_info.md#get_tpu_info), [`infer_tiling`](tpu_info.md#infer_tiling), [`get_tpu_info_for_chip`](tpu_info.md#get_tpu_info_for_chip), [`_sign_lowering_rule`](lowering.md#_sign_lowering_rule), [`get_sparse_core_info`](sc_core.md#get_sparse_core_info), [`_get_tiling_factor`](tpu_info.md#_get_tiling_factor), [`registry`](tpu_info.md#registry.registry)

## Functions
- `_get_tiling_factor(src: int, max_tiling: int, packing: int)` — [`L579`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L579)
- `_get_tpu_info_impl(chip_version: ChipVersion, num_cores: int)` — [`L279`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L279) — Returns the TPU hardware info for the given chip version and core count.
- `chip_version_from_device_kind(device_kind: str)` — [`L111`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L111)
- `get_device_kind()` — [`L633`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L633)
- `get_num_device_cores()` — [`L639`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L639)
- `get_tpu_info()` — [`L510`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L510) — Returns the TPU hardware info for the current device.
- `get_tpu_info_for_chip(chip_version: ChipVersion, num_tensor_cores_per_logical_device: int)` — [`L526`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L526) — Returns the TPU hardware info for the given TPU chip version.
- `infer_tiling(ty: jax_core.AbstractValue, tiling: Tiling | None = None)` — [`L588`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L588) — Compute a tiling for the given shape and type.
- `is_tpu_device()` — [`L272`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L272)

## Module values
- `registry` — [`L276`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/tpu_info.py#L276)

