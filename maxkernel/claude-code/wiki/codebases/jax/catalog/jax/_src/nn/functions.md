---
title: 'Module: jax/_src/nn/functions.py'
type: catalog
provenance: extracted
module: jax/_src/nn/functions.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.nn.functions`/
symbols:
  _one_hot: _one_hot().
  _dot_product_attention_core: _dot_product_attention_core().
  get_scaled_dot_general_config: get_scaled_dot_general_config().
  log_softmax: log_softmax().
  one_hot: one_hot().
  _softmax: _softmax().
  standardize: standardize().
  _softmax_deprecated: _softmax_deprecated().
  dot_product_attention._check_shape_and_dtype: dot_product_attention()._check_shape_and_dtype().
  log1mexp: log1mexp().
  softmax: softmax().
  glu: glu().
  _dot_product_attention_xla: _dot_product_attention_xla().
  _apply_masks: _apply_masks().
  gelu: gelu().
  logmeanexp: logmeanexp().
  scaled_matmul: scaled_matmul().
  silu: silu().
  elu: elu().
  hard_silu: hard_silu().
  squareplus: squareplus().
  sparse_plus: sparse_plus().
  soft_sign: soft_sign().
  log_sigmoid: log_sigmoid().
  leaky_relu: leaky_relu().
  hard_tanh: hard_tanh().
  dot_product_attention._ensure_4d: dot_product_attention()._ensure_4d().
  sigmoid: sigmoid().
  relu6: relu6().
  relu: relu().
  hard_sigmoid: hard_sigmoid().
  identity: identity().
  mish: mish().
  selu: selu().
  softplus: softplus().
  sparse_sigmoid: sparse_sigmoid().
  celu: celu().
  scaled_dot_general: scaled_dot_general().
  _dot_product_attention_xla._reshape_to_grouped: _dot_product_attention_xla()._reshape_to_grouped().
  dot_product_attention: dot_product_attention().
  logsumexp: logsumexp.
  _get_large_negative: _get_large_negative().
  swish: swish.
  _softmax_jvp: _softmax_jvp().
  _get_causal_mask: _get_causal_mask().
  _get_window_mask: _get_window_mask().
  _get_padding_mask_logits: _get_padding_mask_logits().
  _get_padding_mask_encoded: _get_padding_mask_encoded().
  hard_swish: hard_swish.
---
# Module: [`jax/_src/nn/functions.py`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py)

## Functions
- `_apply_masks(logits, mask, is_causal, q_seqlen, kv_seqlen, local_window_size)` — [`L897`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L897)
- `_check_shape_and_dtype(t: Array | None, shape: Sequence[int], dtype: DType | None, name: str)` — [`L1174`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1174)
- `_dot_product_attention_core(query, key, value, bias, mask, is_causal, scale, q_seqlen, kv_seqlen, local_window_size, return_residual)` — [`L925`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L925)
- `_dot_product_attention_xla(query: Array, key: Array, value: Array, bias: Array | None, mask: Array | None, is_causal: bool, scale: float, q_seqlen: Array | None, kv_seqlen: Array | None, local_window_size: tuple[int, int] | None, return_residual: bool = False)` — [`L985`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L985)
- `_ensure_4d(t)` — [`L1155`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1155)
- `_get_causal_mask(T, S)` — [`L868`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L868)
- `_get_large_negative(dtype)` — [`L859`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L859)
- `_get_padding_mask_encoded(T, q_seqlen)` — [`L892`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L892)
- `_get_padding_mask_logits(T, S, q_seqlen, kv_seqlen)` — [`L880`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L880)
- `_get_window_mask(T: int, S: int, local_window_size: tuple[int, int])` — [`L872`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L872)
- `_one_hot(x: Array, num_classes: int, *, dtype: DTypeLike, axis: int | AxisName)` — [`L715`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L715)
- `_reshape_to_grouped(t)` — [`L1003`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1003)
- `_softmax(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None, initial: ArrayLike = -np.inf)` — [`L615`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L615)
- `_softmax_deprecated(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None, initial: ArrayLike = -np.inf)` — [`L634`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L634)
- `_softmax_jvp(axis, primals, tangents)` — [`L629`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L629)
- `celu(x: ArrayLike, alpha: ArrayLike = 1)` — [`L379`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L379) — Continuously-differentiable exponential linear unit activation.
- `dot_product_attention(query: ArrayLike, key: ArrayLike, value: ArrayLike, bias: ArrayLike | None = None, mask: ArrayLike | None = None, *, scale: float | None = None, is_causal: bool = False, query_seq_lengths: ArrayLike | None = None, key_value_seq_lengths: ArrayLike | None = None, local_window_size: int | tuple[int, int] | None = None, implementation: Literal['xla', 'cudnn'] | None = None, return_residual: Literal[False] = ...)` — [`L1033`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1033)
- `elu(x: ArrayLike, alpha: ArrayLike = 1)` — [`L303`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L303) — Exponential linear unit activation function.
- `gelu(x: ArrayLike, approximate: bool = True)` — [`L437`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L437) — Gaussian error linear unit activation function.
- `get_scaled_dot_general_config(mode: Literal['nvfp4', 'mxfp8'], global_scale: Array | None = None)` — [`L1376`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1376) — Get quantization configs for scaled_dot_general.
- `glu(x: ArrayLike, axis: int = -1)` — [`L472`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L472) — Gated linear unit activation function.
- `hard_sigmoid(x: ArrayLike)` — [`L814`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L814) — Hard Sigmoid activation function.
- `hard_silu(x: ArrayLike)` — [`L834`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L834) — Hard SiLU (swish) activation function
- `hard_tanh(x: ArrayLike)` — [`L357`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L357) — Hard :math:`\mathrm{tanh}` activation function.
- `identity(x: ArrayLike)` — [`L53`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L53) — Identity activation function.
- `leaky_relu(x: ArrayLike, negative_slope: ArrayLike = 0.01)` — [`L330`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L330) — Leaky rectified linear unit activation function.
- `log1mexp(x: ArrayLike)` — [`L1494`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1494) — Numerically stable calculation of :math:`\log(1 - \exp(-x))`.
- `log_sigmoid(x: ArrayLike)` — [`L282`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L282) — Log-sigmoid activation function.
- `log_softmax(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None)` — [`L536`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L536) — Log-Softmax function.
- `logmeanexp(x: ArrayLike, axis: Axis = None, where: ArrayLike | None = None, keepdims: bool = False)` — [`L507`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L507) — Log mean exp.
- `mish(x: ArrayLike)` — [`L260`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L260) — Mish activation function.
- `one_hot(x: Any, num_classes: int, *, dtype: Any | None = None, axis: int | AxisName = -1)` — [`L741`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L741) — One-hot encodes the given indices.
- `relu(x: ArrayLike)` — [`L73`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L73) — Rectified linear unit activation function.
- `relu6(x: ArrayLike)` — [`L782`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L782) — Rectified Linear Unit 6 activation function.
- `scaled_dot_general(lhs, rhs, dimension_numbers, preferred_element_type=np.float32, configs: list[BlockScaleConfig] | None = None, implementation: Literal['cudnn'] | None = None)` — [`L1408`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1408) — Scaled dot general operation.
- `scaled_matmul(lhs: Array, rhs: Array, lhs_scales: Array, rhs_scales: Array, preferred_element_type: DTypeLike = np.float32)` — [`L1268`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L1268) — Scaled matrix multiplication function.
- `selu(x: ArrayLike)` — [`L404`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L404) — Scaled exponential linear unit activation.
- `sigmoid(x: ArrayLike)` — [`L182`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L182) — Sigmoid activation function.
- `silu(x: ArrayLike)` — [`L235`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L235) — SiLU (aka swish) activation function.
- `soft_sign(x: ArrayLike)` — [`L167`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L167) — Soft-sign activation function.
- `softmax(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None)` — [`L579`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L579) — Softmax function.
- `softplus(x: ArrayLike)` — [`L128`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L128) — Softplus activation function.
- `sparse_plus(x: ArrayLike)` — [`L142`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L142) — Sparse plus function.
- `sparse_sigmoid(x: ArrayLike)` — [`L203`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L203) — Sparse sigmoid activation function.
- `squareplus(x: ArrayLike, b: ArrayLike = 4)` — [`L109`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L109) — Squareplus activation function.
- `standardize(x: ArrayLike, axis: Axis = -1, mean: ArrayLike | None = None, variance: ArrayLike | None = None, epsilon: ArrayLike = 0.00001, where: ArrayLike | None = None, *, algorithm: str = "fast")` — [`L649`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L649) — Standardizes input to zero mean and unit variance.

## Module values
- `hard_swish` — [`L857`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L857)
- `logsumexp` — [`L503`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L503)
- `swish` — [`L257`](../../../../../../../raw/code/jax/jax/_src/nn/functions.py#L257)

