---
title: 'Module: aqt/jax/v2/flax/aqt_flax.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/aqt_flax.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.aqt_flax`/
symbols:
  NoShardingAxes: NoShardingAxes.
  AxisMetadataWrapper: AxisMetadataWrapper.
  DotGeneralTilingFn: DotGeneralTilingFn.
  EinsumTilingFn: EinsumTilingFn.
  _QUANT_TO_FREEZER_MODE: _QUANT_TO_FREEZER_MODE.
  _freezer_qtensor_init_wrapper: _freezer_qtensor_init_wrapper().
  aqt_promote_dtype: aqt_promote_dtype().
  FreezerMode: FreezerMode#
  FreezerMode.NONE: FreezerMode#NONE.
  FreezerMode.CALIBRATION: FreezerMode#CALIBRATION.
  FreezerMode.CALIBRATION_AND_VALUE: FreezerMode#CALIBRATION_AND_VALUE.
  Freezer: Freezer#
  Freezer.quant_collection: Freezer#quant_collection.
  Freezer.quant_mode: Freezer#quant_mode.
  Freezer.q_shape: Freezer#q_shape.
  Freezer.q_dtype: Freezer#q_dtype.
  Freezer.q_init: Freezer#q_init.
  Freezer.s_shape: Freezer#s_shape.
  Freezer.s_init: Freezer#s_init.
  Freezer.setup: Freezer#setup().
  Freezer.get: Freezer#get().
  Freezer.set: Freezer#set().
  _maybe_recover_scale_from_scale_t: _maybe_recover_scale_from_scale_t().
  _populate_scale_t: _populate_scale_t().
  AqtDotGeneral: AqtDotGeneral#
  AqtDotGeneral.cfg: AqtDotGeneral#cfg.
  AqtDotGeneral.prng_name: AqtDotGeneral#prng_name.
  AqtDotGeneral.lhs_quant_mode: AqtDotGeneral#lhs_quant_mode.
  AqtDotGeneral.lhs_apply_quant_mode: AqtDotGeneral#lhs_apply_quant_mode.
  AqtDotGeneral.lhs_var_name: AqtDotGeneral#lhs_var_name.
  AqtDotGeneral.lhs_qtensor: AqtDotGeneral#lhs_qtensor.
  AqtDotGeneral.rhs_quant_mode: AqtDotGeneral#rhs_quant_mode.
  AqtDotGeneral.rhs_apply_quant_mode: AqtDotGeneral#rhs_apply_quant_mode.
  AqtDotGeneral.rhs_var_name: AqtDotGeneral#rhs_var_name.
  AqtDotGeneral.rhs_qtensor: AqtDotGeneral#rhs_qtensor.
  AqtDotGeneral.lhs_init: AqtDotGeneral#lhs_init.
  AqtDotGeneral.lhs_scale_init: AqtDotGeneral#lhs_scale_init.
  AqtDotGeneral.rhs_init: AqtDotGeneral#rhs_init.
  AqtDotGeneral.rhs_scale_init: AqtDotGeneral#rhs_scale_init.
  AqtDotGeneral.lhs_axis_metadata_wrapper: AqtDotGeneral#lhs_axis_metadata_wrapper.
  AqtDotGeneral.rhs_axis_metadata_wrapper: AqtDotGeneral#rhs_axis_metadata_wrapper.
  AqtDotGeneral.lhs_freeze_mode: AqtDotGeneral#lhs_freeze_mode.
  AqtDotGeneral.rhs_freeze_mode: AqtDotGeneral#rhs_freeze_mode.
  AqtDotGeneral.quant_collection: AqtDotGeneral#quant_collection.
  AqtDotGeneral.tiling_cfg: AqtDotGeneral#tiling_cfg.
  AqtDotGeneral.tiling_fn: AqtDotGeneral#tiling_fn.
  AqtDotGeneral.use_legacy_freezer: AqtDotGeneral#use_legacy_freezer.
  AqtDotGeneral.make_aqt_dg: AqtDotGeneral#make_aqt_dg().
  AqtDotGeneral.__call__: AqtDotGeneral#__call__().
  AqtEinsum: AqtEinsum#
  AqtEinsum.cfg: AqtEinsum#cfg.
  AqtEinsum.prng_name: AqtEinsum#prng_name.
  AqtEinsum.lhs_quant_mode: AqtEinsum#lhs_quant_mode.
  AqtEinsum.lhs_var_name: AqtEinsum#lhs_var_name.
  AqtEinsum.rhs_quant_mode: AqtEinsum#rhs_quant_mode.
  AqtEinsum.rhs_var_name: AqtEinsum#rhs_var_name.
  AqtEinsum.lhs_init: AqtEinsum#lhs_init.
  AqtEinsum.lhs_scale_init: AqtEinsum#lhs_scale_init.
  AqtEinsum.rhs_init: AqtEinsum#rhs_init.
  AqtEinsum.rhs_scale_init: AqtEinsum#rhs_scale_init.
  AqtEinsum.lhs_axis_metadata_wrapper: AqtEinsum#lhs_axis_metadata_wrapper.
  AqtEinsum.rhs_axis_metadata_wrapper: AqtEinsum#rhs_axis_metadata_wrapper.
  AqtEinsum.lhs_freeze_mode: AqtEinsum#lhs_freeze_mode.
  AqtEinsum.rhs_freeze_mode: AqtEinsum#rhs_freeze_mode.
  AqtEinsum.quant_collection: AqtEinsum#quant_collection.
  AqtEinsum.assert_eqn: AqtEinsum#assert_eqn.
  AqtEinsum.assert_lhs_shape: AqtEinsum#assert_lhs_shape.
  AqtEinsum.assert_rhs_shape: AqtEinsum#assert_rhs_shape.
  AqtEinsum.tile_sizes: AqtEinsum#tile_sizes.
  AqtEinsum.tiling_fn: AqtEinsum#tiling_fn.
  AqtEinsum.use_legacy_freezer: AqtEinsum#use_legacy_freezer.
  AqtEinsum.__call__: AqtEinsum#__call__().
  AqtConvGeneralDilated: AqtConvGeneralDilated#
  AqtConvGeneralDilated.cfg: AqtConvGeneralDilated#cfg.
  AqtConvGeneralDilated.prng_name: AqtConvGeneralDilated#prng_name.
  AqtConvGeneralDilated.lhs_quant_mode: AqtConvGeneralDilated#lhs_quant_mode.
  AqtConvGeneralDilated.lhs_apply_quant_mode: AqtConvGeneralDilated#lhs_apply_quant_mode.
  AqtConvGeneralDilated.lhs_var_name: AqtConvGeneralDilated#lhs_var_name.
  AqtConvGeneralDilated.lhs_qtensor: AqtConvGeneralDilated#lhs_qtensor.
  AqtConvGeneralDilated.rhs_quant_mode: AqtConvGeneralDilated#rhs_quant_mode.
  AqtConvGeneralDilated.rhs_apply_quant_mode: AqtConvGeneralDilated#rhs_apply_quant_mode.
  AqtConvGeneralDilated.rhs_var_name: AqtConvGeneralDilated#rhs_var_name.
  AqtConvGeneralDilated.rhs_qtensor: AqtConvGeneralDilated#rhs_qtensor.
  AqtConvGeneralDilated.lhs_axis_metadata_wrapper: AqtConvGeneralDilated#lhs_axis_metadata_wrapper.
  AqtConvGeneralDilated.rhs_axis_metadata_wrapper: AqtConvGeneralDilated#rhs_axis_metadata_wrapper.
  AqtConvGeneralDilated.lhs_freeze_mode: AqtConvGeneralDilated#lhs_freeze_mode.
  AqtConvGeneralDilated.rhs_freeze_mode: AqtConvGeneralDilated#rhs_freeze_mode.
  AqtConvGeneralDilated.quant_collection: AqtConvGeneralDilated#quant_collection.
  AqtConvGeneralDilated.make_aqt_cg: AqtConvGeneralDilated#make_aqt_cg().
  AqtConvGeneralDilated.__call__: AqtConvGeneralDilated#__call__().
---
# Module: [`aqt/jax/v2/flax/aqt_flax.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py)

## Classes
### `AqtConvGeneralDilated`
- def: [`aqt/jax/v2/flax/aqt_flax.py:735`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L735)
- doc: A layer that can be injected into flax.nn.Dense, etc.
- signature: `class AqtConvGeneralDilated(nn.Module):`
- members:
  - `make_aqt_cg(self)` — [`L768`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L768)
  - `cfg` — [`L738`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L738)
  - `lhs_apply_quant_mode` — [`L746`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L746)
  - `lhs_axis_metadata_wrapper` — [`L756`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L756)
  - `lhs_freeze_mode` — [`L761`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L761)
  - `lhs_qtensor` — [`L748`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L748)
  - `lhs_quant_mode` — [`L744`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L744)
  - `lhs_var_name` — [`L747`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L747)
  - `prng_name` — [`L739`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L739)
  - `quant_collection` — [`L766`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L766)
  - `rhs_apply_quant_mode` — [`L751`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L751)
  - `rhs_axis_metadata_wrapper` — [`L757`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L757)
  - `rhs_freeze_mode` — [`L762`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L762)
  - `rhs_qtensor` — [`L753`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L753)
  - `rhs_quant_mode` — [`L750`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L750)
  - `rhs_var_name` — [`L752`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L752)
- protocol/private: `__call__`[`L902`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L902)

### `AqtDotGeneral`
- def: [`aqt/jax/v2/flax/aqt_flax.py:278`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L278)
- doc: A layer that can be injected into flax.nn.Dense, etc.
- signature: `class AqtDotGeneral(nn.Module):`
- members:
  - `make_aqt_dg(self, lhs_shape, rhs_shape, dimension_numbers: tuple[Iterable[int], Iterable[int]], lhs_tile_map: None | tiled_dot_general.AqtTileMap=None, rhs_tile_map: None | tiled_dot_general.AqtTileMap=None)` — [`L326`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L326)
  - `cfg` — [`L281`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L281)
  - `lhs_apply_quant_mode` — [`L289`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L289)
  - `lhs_axis_metadata_wrapper` — [`L306`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L306)
  - `lhs_freeze_mode` — [`L311`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L311)
  - `lhs_init` — [`L299`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L299)
  - `lhs_qtensor` — [`L291`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L291)
  - `lhs_quant_mode` — [`L287`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L287)
  - `lhs_scale_init` — [`L300`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L300)
  - `lhs_var_name` — [`L290`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L290)
  - `prng_name` — [`L282`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L282)
  - `quant_collection` — [`L316`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L316)
  - `rhs_apply_quant_mode` — [`L294`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L294)
  - `rhs_axis_metadata_wrapper` — [`L307`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L307)
  - `rhs_freeze_mode` — [`L312`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L312)
  - `rhs_init` — [`L302`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L302)
  - `rhs_qtensor` — [`L296`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L296)
  - `rhs_quant_mode` — [`L293`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L293)
  - `rhs_scale_init` — [`L303`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L303)
  - `rhs_var_name` — [`L295`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L295)
  - `tiling_cfg` — [`L319`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L319)
  - `tiling_fn` — [`L320`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L320)
  - `use_legacy_freezer` — [`L324`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L324)
- protocol/private: `__call__`[`L517`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L517)

### `AqtEinsum`
- def: [`aqt/jax/v2/flax/aqt_flax.py:573`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L573)
- doc: Quantized Einsum class for model injection.
- signature: `class AqtEinsum(nn.Module):`
- members:
  - `assert_eqn` — [`L606`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L606)
  - `assert_lhs_shape` — [`L607`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L607)
  - `assert_rhs_shape` — [`L608`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L608)
  - `cfg` — [`L576`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L576)
  - `lhs_axis_metadata_wrapper` — [`L594`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L594)
  - `lhs_freeze_mode` — [`L599`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L599)
  - `lhs_init` — [`L587`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L587)
  - `lhs_quant_mode` — [`L580`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L580)
  - `lhs_scale_init` — [`L588`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L588)
  - `lhs_var_name` — [`L581`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L581)
  - `prng_name` — [`L577`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L577)
  - `quant_collection` — [`L604`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L604)
  - `rhs_axis_metadata_wrapper` — [`L595`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L595)
  - `rhs_freeze_mode` — [`L600`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L600)
  - `rhs_init` — [`L590`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L590)
  - `rhs_quant_mode` — [`L583`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L583)
  - `rhs_scale_init` — [`L591`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L591)
  - `rhs_var_name` — [`L584`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L584)
  - `tile_sizes` — [`L609`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L609)
  - `tiling_fn` — [`L610`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L610)
  - `use_legacy_freezer` — [`L614`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L614)
- protocol/private: `__call__`[`L617`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L617)

### `Freezer`
- def: [`aqt/jax/v2/flax/aqt_flax.py:149`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L149)
- doc: Identity function that can freeze its input.
- signature: `class Freezer(nn.Module):`
- members:
  - `get(self)` — [`L188`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L188)
  - `set(self, inputs: aqt_tensor.QTensor)` — [`L209`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L209)
  - `setup(self)` — [`L171`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L171)
  - `q_dtype` — [`L166`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L166)
  - `q_init` — [`L167`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L167)
  - `q_shape` — [`L165`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L165)
  - `quant_collection` — [`L163`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L163)
  - `quant_mode` — [`L164`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L164)
  - `s_init` — [`L169`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L169)
  - `s_shape` — [`L168`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L168)

### `FreezerMode`
- def: [`aqt/jax/v2/flax/aqt_flax.py:143`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L143)
- signature: `class FreezerMode(enum.Enum):`
- members:
  - `CALIBRATION` — [`L145`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L145)
  - `CALIBRATION_AND_VALUE` — [`L146`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L146)
  - `NONE` — [`L144`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L144)

## Functions
- `_freezer_qtensor_init_wrapper(qt: aqt_tensor.QTensor, contracting_axis: Sequence[utils.AxisIdx], axis_metadata_wrapper: None | AxisMetadataWrapper, tile_map: tiled_dot_general.AqtTileMap)` — [`L61`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L61) — QTensor initialization wrapper function for the new freezer.
- `_maybe_recover_scale_from_scale_t(qt: None | aqt_tensor.QTensor, dimension_numbers: jax.lax.DotDimensionNumbers, is_rhs: bool, lhs_shape: Sequence[int], rhs_shape: Sequence[int])` — [`L235`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L235) — Recovers scale from scale_t if necessary.
- `_populate_scale_t(qt: aqt_tensor.QTensor, dimension_numbers: jax.lax.DotDimensionNumbers, is_rhs: bool, lhs_shape: Sequence[int], rhs_shape: Sequence[int])` — [`L257`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L257) — Populates scale_t from scale.
- `aqt_promote_dtype(lhs_in: jnp.ndarray, rhs_in: jnp.ndarray)` — [`L102`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L102) — Promotes the dtype of lhs_in and rhs_in.

## Module values
- `AxisMetadataWrapper` — [`L39`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L39)
- `DotGeneralTilingFn` — [`L43`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L43)
- `EinsumTilingFn` — [`L47`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L47)
- `NoShardingAxes` — [`L38`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L38)
- `_QUANT_TO_FREEZER_MODE` — [`L53`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax.py#L53)

