---
title: 'Module: tokamax/_src/ops/ragged_dot/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.base`/
symbols:
  RaggedDot: RaggedDot#
  RaggedDot._fwd: RaggedDot#_fwd().
  GroupSizes: GroupSizes#
  RaggedDot.bind: RaggedDot#bind().
  ActivationFunction: ActivationFunction.
  vjp: vjp().
  DEFAULT_RAGGED_DOT_DIM_NUMS: DEFAULT_RAGGED_DOT_DIM_NUMS.
  GroupSizes.serialize: GroupSizes#serialize().
  AsQArray: AsQArray.
  GroupSizes.repr_value_or_total_size: GroupSizes#repr_value_or_total_size.
  Residuals: Residuals.
  GroupSizes.value: GroupSizes#value.
  QArray: QArray.
  CanonicalPrecision: CanonicalPrecision.
  GroupSizes.__eq__: GroupSizes#__eq__().
  GroupSizes.representative_value: GroupSizes#representative_value().
  generate_group_sizes: generate_group_sizes().
  GroupSizes.__hash__: GroupSizes#__hash__().
  GroupSizes.__get_pydantic_core_schema__: GroupSizes#__get_pydantic_core_schema__().
  RAGGED_CONTRACTING_DOT_DIM_NUMS: RAGGED_CONTRACTING_DOT_DIM_NUMS.
  GroupSizes.validate_dict: GroupSizes#validate_dict().
  TRANS_RHS_RAGGED_DOT_DIM_NUMS: TRANS_RHS_RAGGED_DOT_DIM_NUMS.
  GroupSizes.__init__: GroupSizes#__init__().
  GroupSizes.__jax_array__: GroupSizes#__jax_array__().
  _Config: _Config.
  _Key: _Key.
  _DotAlgorithmLike: _DotAlgorithmLike.
  _STATIC: _STATIC.
  RaggedDot.checkify_group_sizes: RaggedDot#checkify_group_sizes.
---
# Module: [`tokamax/_src/ops/ragged_dot/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py)

## Classes
### `GroupSizes`
- def: [`tokamax/_src/ops/ragged_dot/base.py:69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L69) — documented in [tokamax-_src-ops-ragged_dot-base](../../../../../concepts/tokamax-_src-ops-ragged_dot-base.md)
- doc: A group sizes array with representative values.
- signature: `class GroupSizes:`
- members:
  - `representative_value(self)` — [`L121`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L121) — The representative value.
  - `serialize(x: GroupSizes)` — [`L146`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L146)
  - `validate_dict(x: dict[str, int])` — [`L158`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L158)
  - `repr_value_or_total_size` — [`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L87)
  - `value` — [`L86`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L86)
- protocol/private: `__eq__`[`L131`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L131), `__get_pydantic_core_schema__`[`L142`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L142), `__hash__`[`L138`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L138), `__init__`[`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L89), `__jax_array__`[`L128`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L128)
- uses (calls/refs, reference-scoped): [`generate_group_sizes`](base.md#generate_group_sizes), [`_STATIC`](base.md#_STATIC)
- used by: [`_fwd`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot._fwd), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._fwd), [`_fwd`](base.md#RaggedDot._fwd), [`bind`](base.md#RaggedDot.bind), [`ragged_dot`](api.md#ragged_dot), [`ragged_dot_general`](api.md#ragged_dot_general), [`test_group_sizes`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_group_sizes), [`test_group_sizes`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_group_sizes), [`GroupSizes`](pallas_mosaic_gpu.md#GroupSizes), [`_make_spec`](arg_specs.md#_make_spec), [`GroupSizes`](pallas_triton.md#GroupSizes)  (1 test-only)

### `RaggedDot`  ·  implements/extends Op
- def: [`tokamax/_src/ops/ragged_dot/base.py:207`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L207) — documented in [tokamax-_src-ops-ragged_dot-base](../../../../../concepts/tokamax-_src-ops-ragged_dot-base.md)
- doc: Ragged dot base class.
- signature: `class RaggedDot(op.Op[Any, jax.Array, Residuals, _Config, _Key]):`
- members:
  - `bind(self, lhs: jax.Array | QArray | AsQArray, rhs: jax.Array | QArray | AsQArray, *, group_sizes: jax.Array | GroupSizes | Sequence[int], ragged_dot_dimension_numbers: (jax.lax.RaggedDotDimensionNumbers | None) = None, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, return_residuals: bool = False, group_offset: jax.Array | None = None, activation: ActivationFunction | None = None, manual_axis_type: ManualAxisType | None = None, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, maybe_quantize_lhs: bool = False, zero_initialize: bool = True, fuse_gateup_activation: str | None = None, lhs_quantization_dtype: jax.typing.DTypeLike | None = None, rhs_quantization_dtype: jax.typing.DTypeLike | None = None)` — [`L233`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L233) — documented in [tokamax-_src-ops-ragged_dot-base](../../../../../concepts/tokamax-_src-ops-ragged_dot-base.md)
  - `checkify_group_sizes` — [`L230`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L230)
- protocol/private: `_fwd`[`L298`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L298)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`_fwd`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot._fwd), [`bind`](../op.md#Op.bind), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._fwd), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._fwd), [`_fwd`](pallas_triton.md#PallasTritonRaggedDot._fwd), [`GroupSizes`](base.md#GroupSizes), [`ActivationFunction`](base.md#ActivationFunction), [`PallasMosaicTpuRaggedDot`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot), [`PallasMosaicTpuV2RaggedDot`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot), [`PallasTritonRaggedDot`](pallas_triton.md#PallasTritonRaggedDot), [`PallasMosaicGpuRaggedDot`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot), [`DEFAULT_RAGGED_DOT_DIM_NUMS`](base.md#DEFAULT_RAGGED_DOT_DIM_NUMS), [`AsQArray`](base.md#AsQArray), [`Residuals`](base.md#Residuals), [`value`](base.md#GroupSizes.value), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`QArray`](base.md#QArray), [`_Config`](base.md#_Config), [`_DotAlgorithmLike`](base.md#_DotAlgorithmLike), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`test_roundtrip`](../op_test.md#BoundArgumentsTest.test_roundtrip), [`_fwd`](../op.md#Op._fwd), [`_API_IMPLEMENTATIONS`](../../autotuning/api.md#_API_IMPLEMENTATIONS._API_IMPLEMENTATIONS), [`vjp`](base.md#vjp), [`PallasMosaicTpuRaggedDot`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot), [`PallasMosaicTpuV2RaggedDot`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot), [`_IMPLS`](bench.md#_IMPLS), [`PallasTritonRaggedDot`](pallas_triton.md#PallasTritonRaggedDot), [`_get_autotuning_cache_key`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_autotuning_cache_key), [`PallasMosaicGpuRaggedDot`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot), [`__init__`](base_test.md#RaggedDotWithExplicitVjpTest.__init__), [`_IMPLEMENTATIONS`](api.md#_IMPLEMENTATIONS), [`__init__`](base_test.md#RaggedDotTest.__init__)  (1 test-only)

## Functions
- `generate_group_sizes(*, m: int, num_groups: int, p: Sequence[float] | None = None, seed: int = 0)` — [`L178`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L178) — Returns random group sizes.
- `vjp(residuals: Residuals, out: jax.Array, dout: jax.Array, lhs: jax.Array | AsQArray, rhs: jax.Array | AsQArray, *, group_sizes: jax.Array, ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: CanonicalPrecision, preferred_element_type: jnp.dtype | None, activation: ActivationFunction | None = None, dlhs_ragged_dot: Callable[..., jax.Array] = RaggedDot(), drhs_ragged_dot: Callable[..., jax.Array] = RaggedDot(), manual_axis_type: ManualAxisType | None = None, group_offset: jax.Array | None = None, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, maybe_quantize_lhs: bool = False, zero_initialize: bool = True, fuse_gateup_activation: str | None = None, lhs_quantization_dtype: jax.typing.DTypeLike | None = None, rhs_quantization_dtype: jax.typing.DTypeLike | None = None)` — [`L370`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L370) — Ragged dot VJP.

## Module values
- `ActivationFunction` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L45)
- `AsQArray` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L42)
- `CanonicalPrecision` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L43)
- `DEFAULT_RAGGED_DOT_DIM_NUMS` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L47)
- `QArray` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L41)
- `RAGGED_CONTRACTING_DOT_DIM_NUMS` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L57)
- `Residuals` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L40)
- `TRANS_RHS_RAGGED_DOT_DIM_NUMS` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L52)
- `_Config` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L38)
- `_DotAlgorithmLike` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L44)
- `_Key` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L39)
- `_STATIC` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base.py#L64)

