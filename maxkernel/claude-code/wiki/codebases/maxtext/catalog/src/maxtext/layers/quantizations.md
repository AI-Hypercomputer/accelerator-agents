---
title: 'Module: src/maxtext/layers/quantizations.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/quantizations.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.quantizations`/
symbols:
  DEFAULT: DEFAULT.
  _W_BITS: _W_BITS.
  _A_BITS: _A_BITS.
  _W_SCALE: _W_SCALE.
  _A_SCALE: _A_SCALE.
  _TILE_SIZE: _TILE_SIZE.
  Quantization: Quantization#
  Quantization.dot_general_cls: Quantization#dot_general_cls().
  Quantization.einsum: Quantization#einsum().
  _tiling_fn: _tiling_fn().
  _rhs_axis_metadata_wrapper: _rhs_axis_metadata_wrapper().
  AqtQuantization: AqtQuantization#
  AqtQuantization.quant_dg: AqtQuantization#quant_dg.
  AqtQuantization.quant_mode: AqtQuantization#quant_mode.
  AqtQuantization.replicate_scale: AqtQuantization#replicate_scale.
  AqtQuantization._get_mixed_precision_cfg: AqtQuantization#_get_mixed_precision_cfg().
  AqtQuantization._get_rhs_axis_metadata_wrapper: AqtQuantization#_get_rhs_axis_metadata_wrapper().
  AqtQuantization.dot_general_cls: AqtQuantization#dot_general_cls().
  AqtQuantization.einsum: AqtQuantization#einsum().
  QwixQuantization: QwixQuantization#
  QwixQuantization.quant_mode: QwixQuantization#quant_mode.
  QwixQuantization.act_calibration_method: QwixQuantization#act_calibration_method.
  QwixQuantization.weight_calibration_method: QwixQuantization#weight_calibration_method.
  QwixQuantization.bwd_calibration_method: QwixQuantization#bwd_calibration_method.
  QwixQuantization._get_fp8_full_qwix_config: QwixQuantization#_get_fp8_full_qwix_config().
  QwixQuantization.dot_general_cls: QwixQuantization#dot_general_cls().
  QwixQuantization.einsum: QwixQuantization#einsum().
  QwixDotGeneral: QwixDotGeneral#
  QwixDotGeneral.config: QwixDotGeneral#config.
  QwixDotGeneral.__call__: QwixDotGeneral#__call__().
  QwixEinsum: QwixEinsum#
  QwixEinsum.config: QwixEinsum#config.
  QwixEinsum.__call__: QwixEinsum#__call__().
  Fp8Quantization: Fp8Quantization#
  Fp8Quantization.quant_mode: Fp8Quantization#quant_mode.
  Fp8Quantization.dot_general_cls: Fp8Quantization#dot_general_cls().
  Fp8Quantization.einsum: Fp8Quantization#einsum().
  _Fp8EinsumWrapper: _Fp8EinsumWrapper#
  _Fp8EinsumWrapper.dtype: _Fp8EinsumWrapper#dtype.
  _Fp8EinsumWrapper.__call__: _Fp8EinsumWrapper#__call__().
  Fp8Einsum: Fp8Einsum#
  Fp8Einsum.amax_history_length: Fp8Einsum#amax_history_length.
  Fp8Einsum.e4m3_dtype: Fp8Einsum#e4m3_dtype.
  Fp8Einsum.e5m2_dtype: Fp8Einsum#e5m2_dtype.
  Fp8Einsum.dtype: Fp8Einsum#dtype.
  Fp8Einsum.setup: Fp8Einsum#setup().
  Fp8Einsum.__call__: Fp8Einsum#__call__().
  NANOOFp8Quantization: NANOOFp8Quantization#
  NANOOFp8Quantization.quant_mode: NANOOFp8Quantization#quant_mode.
  NANOOFp8Quantization.dot_general_cls: NANOOFp8Quantization#dot_general_cls().
  _get_int8_quant_config: _get_int8_quant_config().
  ConstantBoundConfig: ConstantBoundConfig#
  ConstantBoundConfig.fwd_lhs_bound: ConstantBoundConfig#fwd_lhs_bound.
  ConstantBoundConfig.fwd_rhs_bound: ConstantBoundConfig#fwd_rhs_bound.
  ConstantBoundConfig.dlhs_lhs_bound: ConstantBoundConfig#dlhs_lhs_bound.
  ConstantBoundConfig.dlhs_rhs_bound: ConstantBoundConfig#dlhs_rhs_bound.
  ConstantBoundConfig.drhs_lhs_bound: ConstantBoundConfig#drhs_lhs_bound.
  ConstantBoundConfig.drhs_rhs_bound: ConstantBoundConfig#drhs_rhs_bound.
  _build_const_scale_config: _build_const_scale_config().
  PerTensorScales: PerTensorScales#
  PerTensorScales.fwd_lhs: PerTensorScales#fwd_lhs.
  PerTensorScales.fwd_rhs: PerTensorScales#fwd_rhs.
  PerTensorScales.dlhs_lhs: PerTensorScales#dlhs_lhs.
  PerTensorScales.dlhs_rhs: PerTensorScales#dlhs_rhs.
  PerTensorScales.drhs_lhs: PerTensorScales#drhs_lhs.
  PerTensorScales.drhs_rhs: PerTensorScales#drhs_rhs.
  _build_per_tensor_config: _build_per_tensor_config().
  _get_aqt_fp8_default_config: _get_aqt_fp8_default_config().
  _get_aqt_fp8_quant_config: _get_aqt_fp8_quant_config().
  _dot_general_make: _dot_general_make().
  _get_default_mp_config: _get_default_mp_config().
  _get_mixed_precision_quant_config: _get_mixed_precision_quant_config().
  _get_quant_config: _get_quant_config().
  in_convert_mode: in_convert_mode().
  in_serve_mode: in_serve_mode().
  get_quant_mode: get_quant_mode().
  configure_quantization: configure_quantization().
  match_aqt_and_unquantized_param: match_aqt_and_unquantized_param().
  _get_aqt_key_paths: _get_aqt_key_paths().
  remove_quantized_params: remove_quantized_params().
  configure_kv_quant: configure_kv_quant().
  _apply_linen_module_in_nnx: _apply_linen_module_in_nnx().
  NvidaFp8Provider: NvidaFp8Provider#
  NvidaFp8Provider.dot_general: NvidaFp8Provider#dot_general().
  NvidaFp8Provider.einsum: NvidaFp8Provider#einsum().
  NANOOFp8Provider: NANOOFp8Provider#
  NANOOFp8Provider.dot_general: NANOOFp8Provider#dot_general().
  get_fp8_full_qwix_rule_w_sparsity: get_fp8_full_qwix_rule_w_sparsity().
  get_quantization_rule: get_quantization_rule().
  get_qt_provider: get_qt_provider().
  maybe_quantize_model: maybe_quantize_model().
  _cast_reduced_from: _cast_reduced_from().
  _make_scale_tensor: _make_scale_tensor().
  _get_max_min: _get_max_min().
  manual_quantize: manual_quantize().
  TransformerEngineQuantization: TransformerEngineQuantization#
  TransformerEngineQuantization.__init__: TransformerEngineQuantization#__init__().
  TransformerEngineQuantization.__hash__: TransformerEngineQuantization#__hash__().
  TransformerEngineQuantization.__eq__: TransformerEngineQuantization#__eq__().
  TransformerEngineQuantization._get_recipe: TransformerEngineQuantization#_get_recipe().
  TransformerEngineQuantization.get_block_size: TransformerEngineQuantization#get_block_size().
  TransformerEngineQuantization._wrap: TransformerEngineQuantization#_wrap().
  TransformerEngineQuantization.dot_general_cls: TransformerEngineQuantization#dot_general_cls().
  TransformerEngineQuantization.einsum: TransformerEngineQuantization#einsum().
---
# Module: [`src/maxtext/layers/quantizations.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py)

## Classes
### `AqtQuantization`
- def: [`src/maxtext/layers/quantizations.py:127`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L127)
- doc: Configures AQT quantization github.com/google/aqt.
- signature: `class AqtQuantization:`
- members:
  - `_get_mixed_precision_cfg(self)` — [`L134`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L134) — get configuration for mixed precision
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L161`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L161) — Returns dot_general configured with aqt params.
  - `einsum(self, mesh_axes: Tuple[str, ...]=())` — [`L184`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L184) — Returns einsum configured with aqt params.
  - `quant_dg` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L130)
  - `quant_mode` — [`L131`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L131)
  - `replicate_scale` — [`L132`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L132)
- protocol/private: `_get_rhs_axis_metadata_wrapper`[`L152`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L152)

### `ConstantBoundConfig`
- def: [`src/maxtext/layers/quantizations.py:410`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L410)
- signature: `class ConstantBoundConfig:`
- members:
  - `dlhs_lhs_bound` — [`L413`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L413)
  - `dlhs_rhs_bound` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L414)
  - `drhs_lhs_bound` — [`L415`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L415)
  - `drhs_rhs_bound` — [`L416`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L416)
  - `fwd_lhs_bound` — [`L411`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L411)
  - `fwd_rhs_bound` — [`L412`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L412)

### `Fp8Einsum`
- def: [`src/maxtext/layers/quantizations.py:316`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L316)
- doc: An fp8 einsum op.
- signature: `class Fp8Einsum(nn.Module):`
- members:
  - `setup(self)` — [`L328`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L328) — init with input_amax_history, kernel_amax_history, output_grad_amax_history,
  - `amax_history_length` — [`L320`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L320)
  - `dtype` — [`L326`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L326)
  - `e4m3_dtype` — [`L322`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L322)
  - `e5m2_dtype` — [`L324`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L324)
- protocol/private: `__call__`[`L353`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L353)

### `Fp8Quantization`
- def: [`src/maxtext/layers/quantizations.py:289`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L289)
- doc: Configures Fp8 quantization for NVIDIA GPUs
- signature: `class Fp8Quantization(Quantization):`
- members:
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L294`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L294) — Returns dot_general configured with aqt params.
  - `einsum(self, dtype: DType=jnp.float32)` — [`L298`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L298)
  - `quant_mode` — [`L292`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L292)

### `NANOOFp8Provider`
- def: [`src/maxtext/layers/quantizations.py:764`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L764)
- signature: `class NANOOFp8Provider(qwix.QtProvider):`
- members:
  - `dot_general(self, *args, **kwargs)` — [`L766`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L766)

### `NANOOFp8Quantization`
- def: [`src/maxtext/layers/quantizations.py:378`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L378)
- doc: Configures NANOO Fp8 quantization for AMD MI300/MI325 GPUs
- signature: `class NANOOFp8Quantization(Quantization):`
- members:
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L383`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L383) — Returns dot_general configured with aqt params.
  - `quant_mode` — [`L381`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L381)

### `NvidaFp8Provider`
- def: [`src/maxtext/layers/quantizations.py:747`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L747)
- doc: Wraps nn.Fp8DirectDotGeneralOp with Qwix's provider interface.
- signature: `class NvidaFp8Provider(qwix.QtProvider):`
- members:
  - `dot_general(self, *args, **kwargs)` — [`L750`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L750)
  - `einsum(self, *args, **kwargs)` — [`L757`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L757)

### `PerTensorScales`
- def: [`src/maxtext/layers/quantizations.py:464`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L464)
- signature: `class PerTensorScales:`
- members:
  - `dlhs_lhs` — [`L467`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L467)
  - `dlhs_rhs` — [`L468`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L468)
  - `drhs_lhs` — [`L469`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L469)
  - `drhs_rhs` — [`L470`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L470)
  - `fwd_lhs` — [`L465`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L465)
  - `fwd_rhs` — [`L466`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L466)

### `Quantization`
- def: [`src/maxtext/layers/quantizations.py:63`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L63)
- doc: Base class for quantization configurations
- signature: `class Quantization:`
- members:
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L66) — Placeholder for dot_general implementation in subclasses.
  - `einsum(self, dtype: DType=jnp.float32)` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L69) — Placeholder for einsum implementation in subclasses.

### `QwixDotGeneral`
- def: [`src/maxtext/layers/quantizations.py:240`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L240)
- doc: A callable class for Qwix dot_general.
- signature: `class QwixDotGeneral(nn.Module):`
- members:
  - `config` — [`L243`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L243)
- protocol/private: `__call__`[`L246`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L246)

### `QwixEinsum`
- def: [`src/maxtext/layers/quantizations.py:259`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L259)
- doc: A callable class for Qwix einsum.
- signature: `class QwixEinsum(nn.Module):`
- members:
  - `config` — [`L262`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L262)
- protocol/private: `__call__`[`L265`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L265)

### `QwixQuantization`
- def: [`src/maxtext/layers/quantizations.py:209`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L209)
- doc: Configures Qwix quantization github.com/google/qwix, for training only.
- signature: `class QwixQuantization:`
- members:
  - `_get_fp8_full_qwix_config(self)` — [`L217`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L217) — Returns Qwix dot_general config for fp8_full quantization.
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L231`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L231) — Returns Qwix dot_general.
  - `einsum(self, mesh_axes: Tuple[str, ...]=())` — [`L235`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L235) — Returns Qwix einsum.
  - `act_calibration_method` — [`L213`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L213)
  - `bwd_calibration_method` — [`L215`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L215)
  - `quant_mode` — [`L212`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L212)
  - `weight_calibration_method` — [`L214`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L214)

### `TransformerEngineQuantization`
- def: [`src/maxtext/layers/quantizations.py:941`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L941)
- doc: Class for TransformerEngine quantization recipes.
- signature: `class TransformerEngineQuantization(Quantization):`
- members:
  - `__init__(self, config)` — [`L944`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L944) — Initialize TransformerEngine quantization.
  - `_get_recipe(recipe_name: str)` — [`L963`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L963) — Get the TransformerEngine recipe based on the name.
  - `_wrap(self, f, name=None)` — [`L991`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L991) — Wraps the given function `f` to support TransformerEngine quantization.
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L1052`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L1052) — Placeholder for dot_general implementation in subclasses.
  - `einsum(self, dtype: DType=jnp.float32)` — [`L1070`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L1070) — Placeholder for einsum implementation in subclasses.
  - `get_block_size(self)` — [`L978`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L978) — Get the block size for quantization for recipes that require blocks.
- protocol/private: `__eq__`[`L957`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L957), `__hash__`[`L954`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L954)

### `_Fp8EinsumWrapper`
- def: [`src/maxtext/layers/quantizations.py:302`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L302)
- doc: Wrapper for nn.Fp8Einsum to handle computation dtype.
- signature: `class _Fp8EinsumWrapper(nn.Module):`
- members:
  - `dtype` — [`L305`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L305)
- protocol/private: `__call__`[`L308`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L308)

## Functions
- `_apply_linen_module_in_nnx(linen_module_cls, op_id, *args, **kwargs)` — [`L721`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L721) — Applies a Linen module within an NNX context.
- `_build_const_scale_config(aqt_dg: aqt_config.DotGeneral, cst_bound_config: ConstantBoundConfig)` — [`L419`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L419) — Build a constant scale config for AQT dot general.
- `_build_per_tensor_config(aqt_dg: aqt_config.DotGeneral, per_tensor_scales: PerTensorScales)` — [`L473`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L473) — Build a per tensor config for AQT dot general.
- `_cast_reduced_from(arr, reduced_arr)` — [`L874`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L874)
- `_dot_general_make(quant_cfg)` — [`L560`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L560) — Create quantization configs for input matrices to a matmul
- `_get_aqt_fp8_default_config(config)` — [`L502`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L502) — Get aqt for 8-bit floating point quantization configuration.
- `_get_aqt_fp8_quant_config(config)` — [`L554`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L554) — get aqt for 8-bit floating point quantization configuration
- `_get_aqt_key_paths(aqt_vars, params)` — [`L699`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L699) — Generate a list of paths which have aqt state
- `_get_default_mp_config(default=None)` — [`L574`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L574)
- `_get_int8_quant_config(config)` — [`L388`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L388)
- `_get_max_min(target_dtype)` — [`L890`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L890)
- `_get_mixed_precision_quant_config(mixed_precision_config)` — [`L581`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L581) — Set quantization params based on user configuration.
- `_get_quant_config(config)` — [`L597`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L597) — Set quantization params based on user configuration.
- `_make_scale_tensor(scale, arr)` — [`L885`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L885)
- `_rhs_axis_metadata_wrapper(x: jnp.ndarray, tile_map, no_sharding_axis: Sequence[int], mesh_axes: Tuple[str, ...], is_tiled: bool, replicate_scale: bool=False)` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L90) — right-hand-side axis metadata wrapper
- `_tiling_fn(lhs, rhs, dimension_numbers, tile_size)` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L73) — apply tiling function
- `configure_kv_quant(config)` — [`L717`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L717)
- `configure_quantization(config: Config, quant_mode_str: str='train')` — [`L641`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L641) — Configure quantization based on user config and quant mode.
- `get_fp8_full_qwix_rule_w_sparsity(config: Config)` — [`L774`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L774)
- `get_qt_provider(config)` — [`L834`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L834) — Get quantization rules based on the config.
- `get_quant_mode(quant_mode_str: str='train')` — [`L630`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L630) — Set quant mode.
- `get_quantization_rule(config: Config)` — [`L798`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L798) — Returns a list of qwix.QtRule from `dtype`.
- `in_convert_mode(quant)` — [`L622`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L622)
- `in_serve_mode(quant)` — [`L626`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L626)
- `manual_quantize(tensor: jax.Array, dtype: jax.typing.DTypeLike, calibration_method: str)` — [`L897`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L897) — Manually quantizes a tensor based on a fixed calibration method.
- `match_aqt_and_unquantized_param(aqt_params, params)` — [`L670`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L670) — match aqt and unquantized params
- `maybe_quantize_model(model, config)` — [`L846`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L846) — Quantize the model if quantization is enabled.
- `remove_quantized_params(params, aqt_vars)` — [`L706`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L706) — Remove param values with aqt tensors to Null to optimize memory.

## Module values
- `DEFAULT` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L54)
- `_A_BITS` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L56)
- `_A_SCALE` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L58)
- `_TILE_SIZE` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L59)
- `_W_BITS` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L55)
- `_W_SCALE` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/layers/quantizations.py#L57)

