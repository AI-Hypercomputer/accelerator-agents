---
title: 'Module: qwix/_src/core/qarray.py'
type: catalog
provenance: extracted
module: qwix/_src/core/qarray.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.qarray`/
symbols:
  QArray: QArray#
  HowToQuantize: HowToQuantize#
  HowToQuantize.qtype: HowToQuantize#qtype.
  quantize: quantize().
  QArray.qvalue: QArray#qvalue.
  QArray.scale: QArray#scale.
  dequantize: dequantize().
  HowToQuantize.tiled_axes: HowToQuantize#tiled_axes.
  HowToQuantize.channelwise_axes: HowToQuantize#channelwise_axes.
  MaybeQArray.MaybeQArray: MaybeQArray.MaybeQArray.
  HowToQuantize.calibration_method: HowToQuantize#calibration_method.
  QArray.zero_point: QArray#zero_point.
  calibrate: calibrate().
  quantize_with_scale_zero_point: quantize_with_scale_zero_point().
  compute_scale_zero_point: compute_scale_zero_point().
  quantize_api: quantize_api().
  sparsify: sparsify().
  QArray.shape: QArray#shape.
  call_with_generic_broadcast: call_with_generic_broadcast().
  validate_qarray: validate_qarray().
  get_tiled_axes: get_tiled_axes().
  QArray.qtype: QArray#qtype.
  split_axis: split_axis().
  get_accumulator_and_result_type: get_accumulator_and_result_type().
  get_scale_shape: get_scale_shape().
  transpose_array: transpose_array().
  QArray.reshape: QArray#reshape().
  reshape: reshape().
  rewriting_take: rewriting_take().
  clip_gradient_to_calibration: clip_gradient_to_calibration().
  QArray.ndim: QArray#ndim.
  broadcast_to: broadcast_to().
  QArray.astype: QArray#astype().
  QArray.scale_tile_shape: QArray#scale_tile_shape().
  QArray.zero_point_tile_shape: QArray#zero_point_tile_shape().
  USE_RECIPROCAL_FOR_QUANTIZATION: USE_RECIPROCAL_FOR_QUANTIZATION.
  clip_to_calibration: clip_to_calibration().
  QArray.__getitem__: QArray#__getitem__().
  QArray.__post_init__: QArray#__post_init__().
  HowToQuantize.__post_init__: HowToQuantize#__post_init__().
  QArray.transpose: QArray#transpose().
  QArray.swapaxes: QArray#swapaxes().
  rewriting_take.take: rewriting_take().take().
  HowToQuantize.noise_fn: HowToQuantize#noise_fn.
  broadcast_to._broadcast_component: broadcast_to()._broadcast_component().
  QArray.T: QArray#T.
  ShapeT.ShapeT: ShapeT.ShapeT.
  QArray.dtype: QArray#dtype.
  QArray.mT: QArray#mT.
  reshape.reshape_by_groups: reshape().reshape_by_groups().
---
# Module: [`qwix/_src/core/qarray.py`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py)

## Classes
### `HowToQuantize`
- def: [`qwix/_src/core/qarray.py:281`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L281) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- doc: Determines how to quantize an array.
- signature: `class HowToQuantize:`
- members:
  - `calibration_method` — [`L302`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L302) — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
  - `channelwise_axes` — [`L290`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L290) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
  - `noise_fn` — [`L304`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L304)
  - `qtype` — [`L287`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L287) — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
  - `tiled_axes` — [`L297`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L297) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- protocol/private: `__post_init__`[`L306`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L306)
- uses (calls/refs, reference-scoped): [`NoiseFn`](numerics.md#NoiseFn)
- used by: [`quantize`](qarray.md#quantize), [`_maybe_fake_quant`](../providers/odml_ops.md#QuantizedOp._maybe_fake_quant), [`quantize_params`](../providers/ptq.md#quantize_params), [`compute_stats`](../../contrib/smooth_quant.md#SqCalibrationProvider.compute_stats), [`extract_calibrated_quant_context`](../../contrib/calibration.md#extract_calibrated_quant_context), [`_fake_quant`](../providers/odml.md#OdmlQatProvider._fake_quant), [`ragged_dot_qt_bwd`](ragged_dot_qt.md#ragged_dot_qt_bwd), [`quantize_act`](../providers/ptq.md#quantize_act), [`_fake_quant`](../providers/odml.md#OdmlConversionProvider._fake_quant), [`quantize_params`](../../contrib/smooth_quant.md#quantize_params), [`calibrate`](qarray.md#calibrate), [`get_how_to_quantize`](einsum.md#get_how_to_quantize), [`get_how_to_quantize`](dot_general.md#get_how_to_quantize), [`quantize_weight`](../../contrib/gptq_core.md#quantize_weight), [`quantize_api`](qarray.md#quantize_api), [`quantize`](../../contrib/padded_ptq.md#quantize), [`ragged_dot_qt_fwd`](ragged_dot_qt.md#ragged_dot_qt_fwd), [`create_quantized_param`](../providers/ptq.md#create_quantized_param), [`_compute_static_scale_zero_point`](../providers/odml.md#OdmlConversionProvider._compute_static_scale_zero_point), [`init`](../providers/ptq.md#quantize_act.init), [`get_how_to_quantize`](conv_general.md#get_how_to_quantize), [`_get_how_to_quantize`](../providers/odml_ops.md#DotEinsumConv._get_how_to_quantize), [`quantize_weight`](../../contrib/awq_core.md#quantize_weight), [`_fake_quant_op`](../providers/odml.md#OdmlConversionProvider._fake_quant_op), [`find_params`](../../contrib/gptq_core.md#find_params), [`get_scale_shape`](qarray.md#get_scale_shape), [`how`](../providers/ptq.md#WithAux.how), [`search_optimal_scales`](../../contrib/awq_core.md#search_optimal_scales), [`HowToQuantize`](../../contrib/padded_ptq.md#HowToQuantize), [`FakeQuantFn`](../providers/odml_ops.md#FakeQuantFn)  (26 test-only)

### `QArray`
- def: [`qwix/_src/core/qarray.py:32`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L32) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- doc: A quantized array implementation with subchannel support.
- signature: `class QArray:`
- members:
  - `astype(self, dtype: jax.typing.DTypeLike)` — [`L94`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L94) — Cast the dequant type to the given dtype.
  - `reshape(self, *new_shape)` — [`L81`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L81)
  - `scale_tile_shape(self)` — [`L70`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L70) — Returns the tile shape for the scale values.
  - `swapaxes(self, axis1: int, axis2: int)` — [`L98`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L98)
  - `transpose(self, *args)` — [`L84`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L84)
  - `zero_point_tile_shape(self)` — [`L75`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L75) — Returns the tile shape for the zero point values.
  - `T` — [`L66`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L66)
  - `dtype` — [`L65`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L65)
  - `mT` — [`L67`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L67)
  - `ndim` — [`L64`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L64)
  - `qtype` — [`L58`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L58) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
  - `qvalue` — [`L55`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L55)
  - `scale` — [`L56`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L56)
  - `shape` — [`L63`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L63)
  - `zero_point` — [`L57`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L57)
- protocol/private: `__getitem__`[`L87`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L87), `__post_init__`[`L90`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L90)
- uses (calls/refs, reference-scoped): [`PaddedQArray`](../../contrib/padded_ptq.md#PaddedQArray), [`reshape`](qarray.md#reshape), [`rewriting_take`](qarray.md#rewriting_take)
- used by: [`quantize`](qarray.md#quantize), [`_compute_gradient_for_operand`](dot_general_qt.md#dot_general_qt_bwd._compute_gradient_for_operand), [`dequantize`](qarray.md#dequantize), [`conv_general_qt_bwd`](conv_general_qt.md#conv_general_qt_bwd), [`dot_general`](../providers/ptq.md#PtqProvider.dot_general), [`MaybeQArray`](qarray.md#MaybeQArray.MaybeQArray), [`dot_general`](dot_general.md#dot_general), [`process_prequantized_params`](../utils/checkpoint_util.md#process_prequantized_params), [`conv_general_dilated`](../providers/ptq.md#PtqProvider.conv_general_dilated), [`_fast_dot_general`](dot_general.md#_fast_dot_general), [`ragged_dot_qt_bwd`](ragged_dot_qt.md#ragged_dot_qt_bwd), [`loop_dot_general`](dot_general.md#loop_dot_general), [`quantize_act`](../providers/ptq.md#quantize_act), [`dot_general`](../providers/lora.md#LoraProvider.dot_general), [`einsum`](../providers/lora.md#LoraProvider.einsum), [`_fast_ragged_dot_general`](ragged_dot.md#_fast_ragged_dot_general), [`einsum`](einsum.md#einsum), [`einsum`](../../contrib/padded_ptq.md#einsum), [`conv_general_dilated`](../providers/lora.md#LoraProvider.conv_general_dilated), [`_fast_conv_general_dilated`](conv_general.md#_fast_conv_general_dilated), [`quantize_with_scale_zero_point`](qarray.md#quantize_with_scale_zero_point), [`quantize_weight`](../../contrib/gptq_core.md#quantize_weight), [`quantize_api`](qarray.md#quantize_api), [`can_use_qmm_in_dot_general`](../../contrib/kernels/quantized_matmul.md#can_use_qmm_in_dot_general), [`_dequantize_quantized_param`](../utils/checkpoint_util.md#_dequantize_quantized_param), [`_process_quantized_param`](../utils/checkpoint_util.md#_process_quantized_param), [`asarray`](../providers/ptq.md#PtqProvider.asarray), [`create_quantized_param`](../providers/ptq.md#create_quantized_param), [`ragged_dot_general`](ragged_dot.md#ragged_dot_general), [`dot_general`](../../contrib/smooth_quant.md#SqInferenceProvider.dot_general), [`dequantize`](../../contrib/padded_ptq.md#dequantize), [`get_tiled_axes`](qarray.md#get_tiled_axes), [`validate_qarray`](qarray.md#validate_qarray), [`_get_residual_for_backward`](dot_general_qt.md#_get_residual_for_backward), [`_flatten_to_3d`](mxfp_dot.md#_flatten_to_3d), [`dot_general`](../../contrib/padded_ptq.md#dot_general), [`get_accumulator_and_result_type`](qarray.md#get_accumulator_and_result_type), [`_pad_operand_if_qarray`](../../contrib/padded_ptq.md#_pad_operand_if_qarray), [`dot_general`](../../contrib/awq.md#AwqInferenceProvider.dot_general), [`WithAwqScale`](../../contrib/awq.md#WithAwqScale)  (+19 more; 34 test-only)

## Functions
- `_broadcast_component(x: jax.Array | None)` — [`L453`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L453) — Broadcasts a single array component to the target shape.
- `broadcast_to(operand: MaybeQArray, target_shape: tuple[int, ...])` — [`L446`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L446) — Broadcasts a QArray or Array to the target shape recursively. — documented in [qwix-_src-utils-checkpoint_util](../../../../concepts/qwix-_src-utils-checkpoint_util.md)
- `calibrate(array: jax.Array, how: HowToQuantize)` — [`L469`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L469) — Calibrates the array. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `call_with_generic_broadcast(op: Callable[[jax.Array, jax.Array], jax.Array], x: jax.Array, y: jax.Array)` — [`L424`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L424) — Call an element-wise binary op with generic broadcast. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `clip_gradient_to_calibration(g: jax.Array, array: jax.Array, calibration: dict[str, jax.Array], calibration_method: str)` — [`L717`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L717) — Clips the gradient if data falls outside calibration bounds. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `clip_to_calibration(array: jax.Array, calibration: Mapping[str, jax.Array], tiled_axes: Mapping[int, int | float])` — [`L697`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L697) — Clips an array to the calibration range.
- `compute_scale_zero_point(calibration: Mapping[str, jax.Array], qtype: jax.typing.DTypeLike)` — [`L536`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L536) — Computes the scale and zero_point from the calibration result. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `dequantize(array: QArray)` — [`L678`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L678) — Dequantizes an array. The reverse of |quantize|. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `get_accumulator_and_result_type(*args: MaybeQArray, preferred_element_type: jax.typing.DTypeLike | None)` — [`L767`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L767) — jnp.result_type for QArray. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `get_scale_shape(array_shape: ShapeT, how: HowToQuantize)` — [`L331`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L331) — Returns the scale shape.
- `get_tiled_axes(array: QArray)` — [`L408`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L408) — Infers the tiled axes from a QArray.
- `quantize(array: jax.Array, how: HowToQuantize)` — [`L624`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L624) — Quantizes an array using a dynamic range. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `quantize_api(array: jax.Array, qtype: jax.typing.DTypeLike, *, channelwise_axes: Collection[int] = (), tiled_axes: Mapping[int, int | float] | None = None, calibration_method: str = 'absmax', scale_dtype: jax.typing.DTypeLike | None = None)` — [`L633`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L633) — Quantize a Jax Array into QArray using a dynamic range. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `quantize_with_scale_zero_point(array: jax.Array, qtype: jax.typing.DTypeLike, scale: jax.Array, zero_point: jax.Array | None, noise_fn: numerics.NoiseFn | None = None)` — [`L580`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L580) — Quantizes an array with the given scale and zero_point. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `reshape(array: QArray, *new_shape)` — [`L106`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L106) — Reshapes the array, which is not always feasible.
- `reshape_by_groups(x: jax.Array)` — [`L147`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L147)
- `rewriting_take(array: QArray, idx)` — [`L173`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L173) — Returns array[*idx].
- `sparsify(array: jax.Array, how: sparsity.SparsityRule)` — [`L255`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L255) — Applies N:M sparsity to a dense array. — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
- `split_axis(array: jax.Array, tiled_axes: Mapping[int, int | float])` — [`L389`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L389) — Reshape the array where the axis is split into (tile_count, tile_size). — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `take(x: jax.Array)` — [`L185`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L185)
- `transpose_array(array: jax.Array, transpose: Sequence[int | None])` — [`L351`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L351) — Enhanced version of jnp.transpose.
- `validate_qarray(array: QArray)` — [`L212`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L212) — Validates the internal consistency of a QArray.

## Module values
- `MaybeQArray` — [`L328`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L328) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `ShapeT` — [`L327`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L327)
- `USE_RECIPROCAL_FOR_QUANTIZATION` — [`L577`](../../../../../../../raw/code/qwix/qwix/_src/core/qarray.py#L577)

