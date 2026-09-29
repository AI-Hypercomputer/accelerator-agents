---
title: 'Module: recurrentgemma/torch/layers.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/layers.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.layers`/
symbols:
  Conv1D: Conv1D#
  RGLRU: RGLRU#
  RGLRU.a_gate: RGLRU#a_gate.
  RGLRU.reset_parameters: RGLRU#reset_parameters().
  RMSNorm.forward: RMSNorm#forward().
  BlockDiagonalLinear.forward: BlockDiagonalLinear#forward().
  rnn_scan: rnn_scan().
  BlockDiagonalLinear.reset_parameters: BlockDiagonalLinear#reset_parameters().
  Einsum.forward: Einsum#forward().
  BlockDiagonalLinear.b: BlockDiagonalLinear#b.
  RGLRU.input_gate: RGLRU#input_gate.
  RGLRU.forward: RGLRU#forward().
  Conv1D.reset_parameters: Conv1D#reset_parameters().
  Conv1D.forward: Conv1D#forward().
  Einsum.reset_parameters: Einsum#reset_parameters().
  BlockDiagonalLinear.block_width: BlockDiagonalLinear#block_width.
  Conv1D._concatenate_with_cache: Conv1D#_concatenate_with_cache().
  Conv1D.temporal_width: Conv1D#temporal_width.
  RMSNorm.scale: RMSNorm#scale.
  BlockDiagonalLinear.w: BlockDiagonalLinear#w.
  Conv1D.w: Conv1D#w.
  Einsum.b: Einsum#b.
  RMSNorm.reset_parameters: RMSNorm#reset_parameters().
  BlockDiagonalLinear.w_init_: BlockDiagonalLinear#w_init_().
  Conv1D.w_init_: Conv1D#w_init_().
  Conv1D._pad_cache: Conv1D#_pad_cache().
  Einsum.w_init_: Einsum#w_init_().
  BlockDiagonalLinear.num_blocks: BlockDiagonalLinear#num_blocks.
  RGLRU.a_param: RGLRU#a_param.
  Conv1D.b: Conv1D#b.
  Einsum.w: Einsum#w.
  RMSNorm: RMSNorm#
  BlockDiagonalLinear: BlockDiagonalLinear#
  RGLRU.a_param_init: RGLRU#a_param_init().
  RGLRU.init_cache: RGLRU#init_cache().
  Conv1D._compute_document_mask: Conv1D#_compute_document_mask().
  Conv1D._pad_window: Conv1D#_pad_window().
  Conv1D.init_cache: Conv1D#init_cache().
  RGLRU.width: RGLRU#width.
  SqrtBoundDerivative.backward: SqrtBoundDerivative#backward().
  RGLRU.num_heads: RGLRU#num_heads.
  Einsum: Einsum#
  Einsum.w_shape: Einsum#w_shape.
  _MAX_SQRT_GRADIENT: _MAX_SQRT_GRADIENT.
  RMSNorm.width: RMSNorm#width.
  RMSNorm.eps: RMSNorm#eps.
  BlockDiagonalLinear.width: BlockDiagonalLinear#width.
  BlockDiagonalLinear.w_init_variance_scale: BlockDiagonalLinear#w_init_variance_scale.
  rnn_param_init: rnn_param_init().
  SqrtBoundDerivative: SqrtBoundDerivative#
  RGLRU.w_init_variance_scale: RGLRU#w_init_variance_scale.
  Conv1D.width: Conv1D#width.
  Conv1D.w_init_variance_scale: Conv1D#w_init_variance_scale.
  Conv1D._convolution_window_indices: Conv1D#_convolution_window_indices().
  Einsum.b_shape: Einsum#b_shape.
  Einsum.eqn: Einsum#eqn.
  Einsum.w_init_variance_scale: Einsum#w_init_variance_scale.
  RMSNorm.__init__: RMSNorm#__init__().
  BlockDiagonalLinear.__init__: BlockDiagonalLinear#__init__().
  SqrtBoundDerivative.forward: SqrtBoundDerivative#forward().
  RGLRU.__init__: RGLRU#__init__().
  Conv1D.__init__: Conv1D#__init__().
  Einsum.__init__: Einsum#__init__().
---
# Module: [`recurrentgemma/torch/layers.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py)

## Classes
### `BlockDiagonalLinear`  ·  implements/extends Module
- def: [`recurrentgemma/torch/layers.py:76`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L76) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
- doc: Block-diagonal linear layer.
- signature: `class BlockDiagonalLinear(nn.Module):`
- members:
  - `__init__(self, width: int, num_blocks: int, w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L79`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L79) — Initializes the BlockDiagonalLinear.
  - `forward(self, x: at.ExpandedActivations)` — [`L128`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L128) — Calls the BlockDiagonalLinear.
  - `reset_parameters(self)` — [`L117`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L117) — Resets the parameters of the module. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `w_init_(self, w: torch.Tensor)` — [`L122`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L122) — Initializes the weight `w` of the layer.
  - `b` — [`L110`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L110)
  - `block_width` — [`L102`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L102) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `num_blocks` — [`L100`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L100)
  - `w` — [`L105`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L105) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `w_init_variance_scale` — [`L101`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L101)
  - `width` — [`L99`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L99)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`ExpandedActivations`](array_typing.md#ExpandedActivations)
- used by: [`test_numerically_to_jax`](layers_test.md#BlockDiagonalLinearTest.test_numerically_to_jax), [`a_gate`](layers.md#RGLRU.a_gate), [`reset_parameters`](layers.md#RGLRU.reset_parameters), [`input_gate`](layers.md#RGLRU.input_gate), [`BlockDiagonalLinear`](__init__.md#BlockDiagonalLinear)

### `Conv1D`  ·  implements/extends Module
- def: [`recurrentgemma/torch/layers.py:379`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L379) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
- doc: A 1D temporal convolution layer.
- signature: `class Conv1D(nn.Module):`
- members:
  - `__init__(self, width: int, temporal_width: int, w_init_variance_scale: float = 0.01, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L382`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L382) — Initializes the Conv1D.
  - `_compute_document_mask(self, *, segment_pos: at.SegmentPos, start_idx: int, end_idx: int, max_look_ahead: int)` — [`L575`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L575) — Creates a mask to prevent mixing of information between documents. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `_concatenate_with_cache(self, x: at.ExpandedActivations, cache: at.Conv1DState)` — [`L531`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L531) — Concatenates the current input `x` with the previous cache for decoding. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `_convolution_window_indices(self, *, prompt_len: int, shift_back: int, output_len: int)` — [`L552`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L552) — Calculates the start and end indices for the convolution window.
  - `_pad_cache(self, state: at.Conv1DState)` — [`L624`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L624) — Left-pads the state if it is shorter than the temporal width.
  - `_pad_window(self, window: at.ExpandedActivations, output_len: int)` — [`L609`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L609) — Left-pads the window if it is shorter than the output sequence length. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `forward(self, x: at.ExpandedActivations, segment_pos: at.SegmentPos, cache: at.Conv1DState | None = None, return_cache: Literal[True] = True)` — [`L428`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L428) — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)
  - `init_cache(cls, *, batch_size: int, width: int, dtype: torch.dtype, conv1d_temporal_width: int = 4, device: str | torch.device | None = None)` — [`L639`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L639) — Returns an empty initialized cache for the Conv1D.
  - `reset_parameters(self)` — [`L417`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L417) — Resets the parameters of the module.
  - `w_init_(self, w: torch.Tensor)` — [`L422`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L422) — Initializes the weight matrix `w` of the Conv1D.
  - `b` — [`L412`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L412) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `temporal_width` — [`L405`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L405) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `w` — [`L409`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L409) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `w_init_variance_scale` — [`L406`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L406)
  - `width` — [`L404`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L404)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`SegmentPos`](array_typing.md#SegmentPos), [`ExpandedActivations`](array_typing.md#ExpandedActivations), [`Conv1DState`](array_typing.md#Conv1DState)
- used by: [`reset_parameters`](modules.md#RecurrentBlock.reset_parameters), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`test_numerically_to_jax`](layers_test.md#Conv1DTest.test_numerically_to_jax), [`conv_1d`](modules.md#RecurrentBlock.conv_1d), [`Conv1D`](__init__.md#Conv1D)

### `Einsum`  ·  implements/extends Module
- def: [`recurrentgemma/torch/layers.py:653`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L653)
- doc: Einsum is a convenience module for parameterized tensor multiplication.
- signature: `class Einsum(nn.Module):`
- members:
  - `__init__(self, w_shape: Sequence[int], b_shape: Sequence[int], eqn: str, w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L656`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L656) — Initializes the Einsum.
  - `forward(self, x: torch.Tensor)` — [`L701`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L701) — Calls the Einsum.
  - `reset_parameters(self)` — [`L690`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L690) — Resets the parameters of the module.
  - `w_init_(self, w: torch.Tensor)` — [`L695`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L695) — Initializes the weight matrix `w` of the Einsum.
  - `b` — [`L685`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L685)
  - `b_shape` — [`L679`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L679)
  - `eqn` — [`L680`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L680)
  - `w` — [`L684`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L684)
  - `w_init_variance_scale` — [`L681`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L681)
  - `w_shape` — [`L678`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L678)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed)
- used by: [`test_numerically_to_jax`](layers_test.md#EinsumTest.test_numerically_to_jax), [`reset_parameters`](modules.md#MLPBlock.reset_parameters), [`ffw_up`](modules.md#MLPBlock.ffw_up)

### `RGLRU`  ·  implements/extends Module
- def: [`recurrentgemma/torch/layers.py:233`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L233) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
- doc: A Real-Gated Linear Recurrent Unit (RG-LRU) layer.
- signature: `class RGLRU(nn.Module):`
- members:
  - `__init__(self, width: int, num_heads: int, w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L236`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L236) — Initializes the RG-LRU.
  - `a_param_init(self, w: torch.Tensor)` — [`L289`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L289) — Initializes the `A` parameter of the RG-LRU. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `forward(self, x: at.ExpandedActivations, segment_pos: at.SegmentPos, cache: at.RNNState | None = None, return_cache: Literal[True] = True)` — [`L294`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L294) — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)
  - `init_cache(cls, batch_size: int, width: int, device: str | torch.device | None = None)` — [`L368`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L368) — Returns an empty initialized cache for the RG-LRU.
  - `reset_parameters(self)` — [`L283`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L283) — Resets the parameters of the module. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `a_gate` — [`L272`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L272) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `a_param` — [`L262`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L262) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `input_gate` — [`L265`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L265) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `num_heads` — [`L258`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L258) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `w_init_variance_scale` — [`L259`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L259)
  - `width` — [`L257`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L257) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`SegmentPos`](array_typing.md#SegmentPos), [`ExpandedActivations`](array_typing.md#ExpandedActivations), [`rnn_scan`](layers.md#rnn_scan), [`reset_parameters`](layers.md#BlockDiagonalLinear.reset_parameters), [`RNNState`](array_typing.md#RNNState), [`BlockDiagonalLinear`](layers.md#BlockDiagonalLinear), [`SqrtBoundDerivative`](layers.md#SqrtBoundDerivative), [`rnn_param_init`](layers.md#rnn_param_init)
- used by: [`reset_parameters`](modules.md#RecurrentBlock.reset_parameters), [`test_numerically_to_jax`](layers_test.md#RGLRUTest.test_numerically_to_jax), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`rg_lru`](modules.md#RecurrentBlock.rg_lru), [`RGLRU`](__init__.md#RGLRU)

### `RMSNorm`  ·  implements/extends Module
- def: [`recurrentgemma/torch/layers.py:30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L30)
- doc: RMS Norm.
- signature: `class RMSNorm(nn.Module):`
- members:
  - `__init__(self, width: int, eps: float = 0.000001, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L33`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L33) — Initializes the RMSNorm.
  - `forward(self, x: at.Activations)` — [`L66`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L66) — Calls the RMSNorm.
  - `reset_parameters(self)` — [`L61`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L61) — Resets the parameters of the module.
  - `eps` — [`L51`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L51)
  - `scale` — [`L54`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L54)
  - `width` — [`L50`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L50)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`Activations`](array_typing.md#Activations), [`ExpandedActivations`](array_typing.md#ExpandedActivations)
- used by: [`reset_parameters`](modules.md#ResidualBlock.reset_parameters), [`test_numerically_to_jax`](layers_test.md#RMSNormTest.test_numerically_to_jax), [`temporal_pre_norm`](modules.md#ResidualBlock.temporal_pre_norm), [`reset_parameters`](griffin.md#Griffin.reset_parameters), [`final_norm`](griffin.md#Griffin.final_norm), [`channel_pre_norm`](modules.md#ResidualBlock.channel_pre_norm)

### `SqrtBoundDerivative`  ·  implements/extends Function
- def: [`recurrentgemma/torch/layers.py:216`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L216) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
- doc: Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`.
- signature: `class SqrtBoundDerivative(torch.autograd.Function):`
- members:
  - `backward(ctx, grad_output: torch.Tensor)` — [`L226`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L226) — The backward pass, which clips the `sqrt` gradient.
  - `forward(ctx, x: torch.Tensor)` — [`L220`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L220) — The forward pass, which is a normal `sqrt`.
- uses (calls/refs, reference-scoped): [`_MAX_SQRT_GRADIENT`](layers.md#_MAX_SQRT_GRADIENT)
- used by: [`RGLRU`](layers.md#RGLRU)

## Functions
- `rnn_param_init(tensor: torch.Tensor, min_rad: float, max_rad: float, transform: str = "softplus", eps: float = 1e-8)` — [`L194`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L194) — Initializes the `A` parameter of the RG-LRU uniformly on a ring. — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
- `rnn_scan(x: at.ExpandedActivations, a: at.ExpandedActivations, reset: at.Reset, h0: at.RNNState | None, acc_dtype: torch.dtype = torch.float32)` — [`L141`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L141) — Runs the recurrence of a linear RNN. — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)

## Module values
- `_MAX_SQRT_GRADIENT` — [`L27`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers.py#L27)

