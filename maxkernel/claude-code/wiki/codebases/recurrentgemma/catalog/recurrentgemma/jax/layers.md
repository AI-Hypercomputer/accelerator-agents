---
title: 'Module: recurrentgemma/jax/layers.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/layers.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.layers`/
symbols:
  RGLRU: RGLRU#
  Conv1D: Conv1D#
  RGLRU.input_gate: RGLRU#input_gate.
  RGLRU.a_gate: RGLRU#a_gate.
  RGLRU.real_imag_complex: RGLRU#real_imag_complex().
  Einsum.__call__: Einsum#__call__().
  RGLRU.complex_to_merged: RGLRU#complex_to_merged().
  BlockDiagonalLinear.num_blocks: BlockDiagonalLinear#num_blocks.
  Conv1D.w: Conv1D#w.
  RMSNorm.__call__: RMSNorm#__call__().
  BlockDiagonalLinear.__call__: BlockDiagonalLinear#__call__().
  RGLRU.merged_to_complex: RGLRU#merged_to_complex().
  RGLRU.__call__: RGLRU#__call__().
  Conv1D.__call__: Conv1D#__call__().
  RGLRU.scan_type: RGLRU#scan_type.
  RGLRU.only_real: RGLRU#only_real.
  Conv1D.temporal_width: Conv1D#temporal_width.
  BlockDiagonalLinear.w: BlockDiagonalLinear#w.
  BlockDiagonalLinear.b: BlockDiagonalLinear#b.
  RGLRU.dtype: RGLRU#dtype.
  RGLRU.param_dtype: RGLRU#param_dtype.
  RGLRU.use_custom_complex: RGLRU#use_custom_complex().
  RGLRU.a_real_param: RGLRU#a_real_param.
  Conv1D._concatenate_with_state: Conv1D#_concatenate_with_state().
  Einsum.w: Einsum#w.
  RGLRU.width: RGLRU#width.
  RMSNorm.param_dtype: RMSNorm#param_dtype.
  BlockDiagonalLinear.param_dtype: BlockDiagonalLinear#param_dtype.
  BlockDiagonalLinear.setup: BlockDiagonalLinear#setup().
  RGLRU.setup: RGLRU#setup().
  sqrt_bound_derivative: sqrt_bound_derivative().
  RMSNorm.dtype: RMSNorm#dtype.
  Conv1D.param_dtype: Conv1D#param_dtype.
  Einsum.param_dtype: Einsum#param_dtype.
  RMSNorm.width: RMSNorm#width.
  BlockDiagonalLinear.width_input: BlockDiagonalLinear#width_input.
  rnn_real_param_init.init: rnn_real_param_init().init().
  rnn_imag_param_init.init: rnn_imag_param_init().init().
  RGLRU.a_real_param_init: RGLRU#a_real_param_init().
  RGLRU.a_imag_param: RGLRU#a_imag_param.
  Conv1D.b: Conv1D#b.
  Conv1D._pad_cache: Conv1D#_pad_cache().
  Conv1D.init_cache: Conv1D#init_cache().
  Einsum.b: Einsum#b.
  BlockDiagonalLinear.dtype: BlockDiagonalLinear#dtype.
  RGLRU.min_rad: RGLRU#min_rad.
  RMSNorm: RMSNorm#
  RMSNorm.setup: RMSNorm#setup().
  RGLRU.num_heads: RGLRU#num_heads.
  RGLRU.scan_sharding_spec: RGLRU#scan_sharding_spec.
  Conv1D.dtype: Conv1D#dtype.
  Einsum.dtype: Einsum#dtype.
  BlockDiagonalLinear: BlockDiagonalLinear#
  BlockDiagonalLinear.width_output: BlockDiagonalLinear#width_output.
  Conv1D.width: Conv1D#width.
  Einsum.w_shape: Einsum#w_shape.
  Einsum.b_shape: Einsum#b_shape.
  Einsum.eqn: Einsum#eqn.
  RMSNorm.scale: RMSNorm#scale.
  BlockDiagonalLinear.kernel_init: BlockDiagonalLinear#kernel_init().
  rnn_real_param_init: rnn_real_param_init().
  rnn_imag_param_init: rnn_imag_param_init().
  stable_sqrt_fwd: stable_sqrt_fwd().
  stable_sqrt_bwd: stable_sqrt_bwd().
  RGLRU.a_imag_param_init: RGLRU#a_imag_param_init().
  RGLRU.init_cache: RGLRU#init_cache().
  Conv1D.kernel_init: Conv1D#kernel_init().
  Conv1D._compute_document_mask: Conv1D#_compute_document_mask().
  Conv1D._pad_window: Conv1D#_pad_window().
  Einsum.kernel_init: Einsum#kernel_init().
  BlockDiagonalLinear.w_init_variance_scale: BlockDiagonalLinear#w_init_variance_scale.
  Einsum: Einsum#
  RMSNorm.eps: RMSNorm#eps.
  RGLRU.w_init_variance_scale: RGLRU#w_init_variance_scale.
  Conv1D.w_init_variance_scale: Conv1D#w_init_variance_scale.
  Conv1D._convolution_window_indices: Conv1D#_convolution_window_indices().
  Einsum.w_init_variance_scale: Einsum#w_init_variance_scale.
  Conv1D.setup: Conv1D#setup().
  Einsum.setup: Einsum#setup().
---
# Module: [`recurrentgemma/jax/layers.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py)

## Classes
### `BlockDiagonalLinear`  ·  implements/extends Module
- def: [`recurrentgemma/jax/layers.py:69`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L69)
- doc: Block-diagonal linear layer.
- signature: `class BlockDiagonalLinear(nn.Module):`
- members:
  - `__call__(self, x: jt.Float[jt.Array, "*b t e"])` — [`L126`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L126) — Calls the BlockDiagonalLinear.
  - `kernel_init(self)` — [`L90`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L90) — Initializer for the weight `w` of the layer.
  - `setup(self)` — [`L98`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L98)
  - `b` — [`L118`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L118)
  - `dtype` — [`L86`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L86)
  - `num_blocks` — [`L83`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L83) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `param_dtype` — [`L87`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L87) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `w` — [`L112`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L112)
  - `w_init_variance_scale` — [`L85`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L85)
  - `width_input` — [`L82`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L82) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `width_output` — [`L84`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L84)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype)
- used by: [`a_gate`](layers.md#RGLRU.a_gate), [`input_gate`](layers.md#RGLRU.input_gate), [`test_numerically_to_jax`](../torch/layers_test.md#BlockDiagonalLinearTest.test_numerically_to_jax), [`BlockDiagonalLinear`](__init__.md#BlockDiagonalLinear)

### `Conv1D`  ·  implements/extends Module
- def: [`recurrentgemma/jax/layers.py:481`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L481) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
- doc: A 1D temporal convolution layer.
- signature: `class Conv1D(nn.Module):`
- members:
  - `_compute_document_mask(self, *, segment_pos: at.SegmentPos, start_idx: int, end_idx: int, max_look_ahead: int)` — [`L673`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L673) — Creates a mask to prevent mixing of information between documents.
  - `_concatenate_with_state(self, x: at.ExpandedActivations, cache: at.Conv1DState)` — [`L629`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L629) — Concatenates the current input `x` with the previous cache for decoding. — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `_convolution_window_indices(self, *, prompt_len: int, shift_back: int, output_len: int)` — [`L650`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L650) — Calculates the start and end indices for the convolution window.
  - `_pad_cache(self, state: at.Conv1DState)` — [`L715`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L715) — Left-pads the state if it is shorter than the temporal width. — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `_pad_window(self, window: at.ExpandedActivations, output_len: int)` — [`L704`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L704) — Left-pads the window if it is shorter than the output sequence length.
  - `init_cache(cls, batch_size: int, width: int, dtype: at.dtype, conv1d_temporal_width: int = 4)` — [`L726`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L726) — Returns an empty initialized cache for the Conv1D.
  - `kernel_init(self)` — [`L500`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L500) — Initializer for the kernel of the Conv1D.
  - `setup(self)` — [`L508`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L508)
  - `b` — [`L516`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L516) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `dtype` — [`L496`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L496)
  - `param_dtype` — [`L497`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L497) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `temporal_width` — [`L494`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L494) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `w` — [`L510`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L510) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `w_init_variance_scale` — [`L495`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L495)
  - `width` — [`L493`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L493)
- protocol/private: `__call__`[`L524`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L524)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype), [`SegmentPos`](array_typing.md#SegmentPos), [`ExpandedActivations`](array_typing.md#ExpandedActivations), [`RNNState`](array_typing.md#RNNState), [`Conv1DState`](array_typing.md#Conv1DState)
- used by: [`conv_1d`](modules.md#RecurrentBlock.conv_1d), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`test_numerically_to_jax`](../torch/layers_test.md#Conv1DTest.test_numerically_to_jax), [`Conv1D`](__init__.md#Conv1D)

### `Einsum`  ·  implements/extends Module
- def: [`recurrentgemma/jax/layers.py:738`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L738)
- doc: Einsum is a convenience module for parameterized tensor multiplication.
- signature: `class Einsum(nn.Module):`
- members:
  - `__call__(self, x: jax.Array)` — [`L784`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L784) — Calls the Einsum. — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `kernel_init(self)` — [`L759`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L759) — Initializer for the kernel of the Einsum.
  - `setup(self)` — [`L768`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L768)
  - `b` — [`L776`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L776)
  - `b_shape` — [`L752`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L752)
  - `dtype` — [`L755`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L755)
  - `eqn` — [`L753`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L753)
  - `param_dtype` — [`L756`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L756) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `w` — [`L770`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L770)
  - `w_init_variance_scale` — [`L754`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L754)
  - `w_shape` — [`L751`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L751)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype)
- used by: [`ffw_up`](modules.md#MLPBlock.ffw_up), [`test_numerically_to_jax`](../torch/layers_test.md#EinsumTest.test_numerically_to_jax), [`test_einsum`](layers_test.md#LayersTest.test_einsum)

### `RGLRU`  ·  implements/extends Module
- def: [`recurrentgemma/jax/layers.py:217`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L217) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
- doc: A Real-Gated Linear Recurrent Unit (RG-LRU) layer.
- signature: `class RGLRU(nn.Module):`
- members:
  - `a_imag_param_init(self)` — [`L330`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L330) — Initializer for the imag `A` parameter of the RG-LRU.
  - `a_real_param_init(self)` — [`L325`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L325) — Initializer for the real `A` parameter of the RG-LRU.
  - `complex_to_merged(self, x: complex_lib.RealOrComplex)` — [`L301`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L301) — Returns a merged array from a (complex) array. — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `init_cache(cls, batch_size: int, width: int)` — [`L475`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L475) — Returns an empty initialized cache for the RG-LRU.
  - `merged_to_complex(self, x: jt.Float[jt.ArrayLike, "*b"])` — [`L245`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L245) — Returns a (complex) array from a merged array.
  - `real_imag_complex(self, real: jt.Float[jt.Array, "*b"], imag: jt.Float[jt.Array, "*b"])` — [`L268`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L268) — Based on the settings, creates a (complex) number in the correct format. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
  - `setup(self)` — [`L334`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L334)
  - `use_custom_complex(self, real_dtype: jnp.dtype)` — [`L294`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L294) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `a_gate` — [`L366`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L366) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `a_imag_param` — [`L350`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L350) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `a_real_param` — [`L342`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L342) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `dtype` — [`L239`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L239) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `input_gate` — [`L357`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L357) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `min_rad` — [`L242`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L242)
  - `num_heads` — [`L235`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L235)
  - `only_real` — [`L241`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L241) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `param_dtype` — [`L240`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L240) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `scan_sharding_spec` — [`L238`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L238)
  - `scan_type` — [`L236`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L236) — documented in [recurrentgemma-common](../../../concepts/recurrentgemma-common.md)
  - `w_init_variance_scale` — [`L237`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L237)
  - `width` — [`L234`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L234) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
- protocol/private: `__call__`[`L377`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L377)
- uses (calls/refs, reference-scoped): [`RealOrComplex`](complex_lib.md#RealOrComplex), [`Complex`](complex_lib.md#Complex), [`ScanType`](../common.md#ScanType), [`typed`](array_typing.md#typed), [`linear_scan`](scan.md#linear_scan), [`dtype`](array_typing.md#dtype), [`SegmentPos`](array_typing.md#SegmentPos), [`ExpandedActivations`](array_typing.md#ExpandedActivations), [`exp`](complex_lib.md#exp), [`AUTO`](../common.md#ScanType.AUTO), [`num_blocks`](layers.md#BlockDiagonalLinear.num_blocks), [`softplus`](complex_lib.md#softplus), [`RNNState`](array_typing.md#RNNState), [`sigmoid`](complex_lib.md#sigmoid), [`ShardingSpec`](scan.md#ShardingSpec), [`param_dtype`](layers.md#BlockDiagonalLinear.param_dtype), [`sqrt_bound_derivative`](layers.md#sqrt_bound_derivative), [`width_input`](layers.md#BlockDiagonalLinear.width_input), [`dtype`](layers.md#BlockDiagonalLinear.dtype), [`BlockDiagonalLinear`](layers.md#BlockDiagonalLinear), [`width_output`](layers.md#BlockDiagonalLinear.width_output), [`rnn_imag_param_init`](layers.md#rnn_imag_param_init), [`rnn_real_param_init`](layers.md#rnn_real_param_init), [`LINEAR_PALLAS`](../common.md#ScanType.LINEAR_PALLAS), [`w_init_variance_scale`](layers.md#BlockDiagonalLinear.w_init_variance_scale)
- used by: [`lru`](modules.md#RecurrentBlock.lru), [`test_numerically_to_jax`](../torch/layers_test.md#RGLRUTest.test_numerically_to_jax), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`test_rglu`](layers_test.md#LayersTest.test_rglu), [`RGLRU`](__init__.md#RGLRU)

### `RMSNorm`  ·  implements/extends Module
- def: [`recurrentgemma/jax/layers.py:32`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L32)
- doc: RMSNorm layer.
- signature: `class RMSNorm(nn.Module):`
- members:
  - `__call__(self, x: at.ExpandedActivations)` — [`L57`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L57) — Calls the RMSNorm.
  - `setup(self)` — [`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L47)
  - `dtype` — [`L44`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L44)
  - `eps` — [`L43`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L43)
  - `param_dtype` — [`L45`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L45) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `scale` — [`L49`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L49)
  - `width` — [`L42`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L42)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype), [`ExpandedActivations`](array_typing.md#ExpandedActivations)
- used by: [`temporal_pre_norm`](modules.md#ResidualBlock.temporal_pre_norm), [`final_norm`](griffin.md#Griffin.final_norm), [`channel_pre_norm`](modules.md#ResidualBlock.channel_pre_norm), [`test_numerically_to_jax`](../torch/layers_test.md#RMSNormTest.test_numerically_to_jax), [`test_rmsnorm`](layers_test.md#LayersTest.test_rmsnorm)

## Functions
- `init(key: jax.Array, shape: Sequence[int], dtype: at.dtype = jnp.float32)` — [`L150`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L150)
- `init(key: jax.Array, shape: Sequence[int], dtype: at.dtype = jnp.float32)` — [`L173`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L173)
- `rnn_imag_param_init(max_rad: float)` — [`L168`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L168) — Initializes the `A` imag parameter of the RG-LRU uniformly on a ring.
- `rnn_real_param_init(min_rad: float, max_rad: float, transform: str = "softplus", eps: float = 1e-8)` — [`L142`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L142) — Initializes the `A` real parameter of the RG-LRU uniformly on a ring.
- `sqrt_bound_derivative(x: complex_lib.RealOrComplex, max_gradient: float | jax.Array)` — [`L186`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L186) — Computes a square root with a gradient clipped at `max_gradient`. — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
- `stable_sqrt_bwd(max_gradient: float | jax.Array, res: tuple[jax.Array], g: jax.Array)` — [`L204`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L204)
- `stable_sqrt_fwd(x: jax.Array, _: float | jax.Array)` — [`L196`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers.py#L196)

