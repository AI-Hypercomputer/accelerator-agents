---
title: 'Module: src/maxdiffusion/models/quantizations.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/quantizations.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.quantizations`/
symbols:
  AqtQuantization: AqtQuantization#
  AqtQuantization.dot_general_cls: AqtQuantization#dot_general_cls().
  AqtQuantization.einsum: AqtQuantization#einsum().
  DEFAULT: DEFAULT.
  MAX_INT8: MAX_INT8.
  MAX_INT4: MAX_INT4.
  Array: Array.
  DType: DType.
  Quantization: Quantization#
  Quantization.dot_general_cls: Quantization#dot_general_cls().
  Quantization.einsum: Quantization#einsum().
  _rhs_axis_metadata_wrapper: _rhs_axis_metadata_wrapper().
  AqtQuantization.quant_dg: AqtQuantization#quant_dg.
  AqtQuantization.quant_mode: AqtQuantization#quant_mode.
  AqtQuantization.replicate_scale: AqtQuantization#replicate_scale.
  AqtQuantization._get_rhs_axis_metadata_wrapper: AqtQuantization#_get_rhs_axis_metadata_wrapper().
  _get_quant_config: _get_quant_config().
  _get_int8_quant_config: _get_int8_quant_config().
  in_convert_mode: in_convert_mode().
  in_serve_mode: in_serve_mode().
  get_quant_mode: get_quant_mode().
  configure_quantization: configure_quantization().
  match_aqt_and_unquantized_param: match_aqt_and_unquantized_param().
  _get_aqt_key_paths: _get_aqt_key_paths().
  remove_quantized_params: remove_quantized_params().
---
# Module: [`src/maxdiffusion/models/quantizations.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py)

## Classes
### `AqtQuantization`
- def: [`src/maxdiffusion/models/quantizations.py:87`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L87)
- doc: Configures AQT quantization github.com/google/aqt.
- signature: `class AqtQuantization:`
- members:
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L103`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L103) — Returns dot_general configured with aqt params.
  - `einsum(self)` — [`L117`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L117) — Returns einsum configured with aqt params
  - `quant_dg` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L90)
  - `quant_mode` — [`L91`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L91)
  - `replicate_scale` — [`L92`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L92)
- protocol/private: `_get_rhs_axis_metadata_wrapper`[`L94`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L94)
- used by: [`attention_op`](attention_flax.md#FlaxAttention.attention_op), [`Quant`](attention_flax.md#Quant), [`Quant`](unet_2d_blocks_flax.md#Quant), [`_maybe_aqt_einsum`](attention_flax.md#_maybe_aqt_einsum), [`Quant`](unet_2d_condition_flax.md#Quant)

### `Quantization`
- def: [`src/maxdiffusion/models/quantizations.py:39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L39)
- doc: Base class for quantization configurations
- signature: `class Quantization:`
- members:
  - `dot_general_cls(self, mesh_axes: Tuple[str, ...]=())` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L42) — Placeholder for dot_general implementation in subclasses.
  - `einsum(self, dtype: DType=jnp.float32)` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L46) — Placeholder for einsum implementation in subclasses.

## Functions
- `_get_aqt_key_paths(aqt_vars, params)` — [`L242`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L242) — Generate a list of paths which have aqt state
- `_get_int8_quant_config(config)` — [`L157`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L157)
- `_get_quant_config(config)` — [`L131`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L131)
- `_rhs_axis_metadata_wrapper(x: jnp.ndarray, tile_map, no_sharding_axis: Sequence[int], mesh_axes: Tuple[str, ...], is_tiled: bool, replicate_scale: bool=False)` — [`L51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L51)
- `configure_quantization(config, quant_mode_str: str='train')` — [`L208`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L208) — Configure quantization based on user config and quant mode.
- `get_quant_mode(quant_mode_str: str='train')` — [`L195`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L195) — Set quant mode.
- `in_convert_mode(quant)` — [`L187`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L187)
- `in_serve_mode(quant)` — [`L191`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L191)
- `match_aqt_and_unquantized_param(aqt_params, params)` — [`L218`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L218)
- `remove_quantized_params(params, aqt_vars)` — [`L249`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L249) — Remove param values with aqt tensors to Null to optimize memory.

## Module values
- `Array` — [`L34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L34)
- `DEFAULT` — [`L30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L30)
- `DType` — [`L35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L35)
- `MAX_INT4` — [`L33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L33)
- `MAX_INT8` — [`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/quantizations.py#L32)

