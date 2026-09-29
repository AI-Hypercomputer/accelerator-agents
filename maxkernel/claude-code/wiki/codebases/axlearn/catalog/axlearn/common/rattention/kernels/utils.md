---
title: 'Module: axlearn/common/rattention/kernels/utils.py'
type: catalog
provenance: extracted
module: axlearn/common/rattention/kernels/utils.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.rattention.kernels.utils`/
symbols:
  get_feature_map: get_feature_map().
  FeatureMap: FeatureMap#
  inner_float32: inner_float32().
  FeatureMapFn.fwd: FeatureMapFn#fwd.
  sm2_fwd: sm2_fwd().
  sm2_bwd: sm2_bwd().
  relu2_fwd: relu2_fwd().
  relu2_bwd: relu2_bwd().
  sm_fwd: sm_fwd().
  sm_bwd: sm_bwd().
  relu_fwd: relu_fwd().
  relu_bwd: relu_bwd().
  FeatureMapFn: FeatureMapFn#
  FeatureMap.SOFTMAX: FeatureMap#SOFTMAX.
  FeatureMap.RELU: FeatureMap#RELU.
  FeatureMapFn.bwd: FeatureMapFn#bwd.
  inner_float32.wrapper: inner_float32().wrapper().
---
# Module: [`axlearn/common/rattention/kernels/utils.py`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py)

## Classes
### `FeatureMap`  ·  implements/extends Enum
- def: [`axlearn/common/rattention/kernels/utils.py:19`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L19)
- signature: `class FeatureMap(enum.Enum):`
- members:
  - `RELU` — [`L21`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L21)
  - `SOFTMAX` — [`L20`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L20)
- used by: [`get_feature_map`](utils.md#get_feature_map), [`__init__`](../rattention.md#ResidualLinearAttention.__init__), [`residual_linear_attention_linear_scan`](linear_attention_kernels.md#residual_linear_attention_linear_scan), [`residual_linear_attention_w_timestep`](linear_attention_kernels.md#residual_linear_attention_w_timestep), [`_linear_attention_backward_kernel`](linear_attention_kernels.md#_linear_attention_backward_kernel), [`residual_linear_attention`](linear_attention_kernels.md#residual_linear_attention), [`_linear_attention_forward_kernel`](linear_attention_kernels.md#_linear_attention_forward_kernel), [`_linear_attention`](linear_attention_kernels.md#_linear_attention), [`_linear_attention_forward`](linear_attention_kernels.md#_linear_attention_forward), [`_la_backward`](linear_attention_kernels.md#_la_backward), [`feat_fn`](../rattention.md#ResidualLinearAttention.Config.feat_fn)

### `FeatureMapFn`  ·  implements/extends NamedTuple
- def: [`axlearn/common/rattention/kernels/utils.py:24`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L24)
- doc: A named tuple to hold the forward and backward functions of a feature map.
- signature: `class FeatureMapFn(NamedTuple):`
- members:
  - `bwd` — [`L28`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L28)
  - `fwd` — [`L27`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L27)
- used by: [`get_feature_map`](utils.md#get_feature_map), [`_compute_init_state`](../rattention.md#ResidualLinearAttention._compute_init_state), [`residual_linear_attention_linear_scan`](linear_attention_kernels.md#residual_linear_attention_linear_scan), [`residual_linear_attention_w_timestep`](linear_attention_kernels.md#residual_linear_attention_w_timestep), [`_la_backward_dkv_chunk_loop_body`](linear_attention_kernels.md#_linear_attention_backward_kernel._la_backward_dkv_chunk_loop_body), [`_la_backward_dq_chunk_loop_body`](linear_attention_kernels.md#_linear_attention_backward_kernel._la_backward_dq_chunk_loop_body), [`_prepare_linear_attention_inputs`](../rattention.md#ResidualLinearAttention._prepare_linear_attention_inputs), [`_la_forward_chunk_loop_body`](linear_attention_kernels.md#_linear_attention_forward_kernel._la_forward_chunk_loop_body)

## Functions
- `get_feature_map(feat_map: FeatureMap)` — [`L31`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L31) — Get the feature map function and its backward function.
- `inner_float32(func)` — [`L41`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L41) — Decorator to convert inputs to float32 before calling the function,
- `relu2_bwd(y: Tensor, dy: Tensor)` — [`L113`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L113) — ReLU feature map, backward pass [relu(x), relu(-x)]
- `relu2_fwd(x: Tensor)` — [`L105`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L105) — ReLU feature map, forward pass [relu(x), relu(-x)]
- `relu_bwd(y: Tensor, dy: Tensor)` — [`L99`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L99) — ReLU feature map, backward pass
- `relu_fwd(x: Tensor)` — [`L92`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L92) — ReLU feature map, forward pass
- `sm2_bwd(y: Tensor, dy: Tensor)` — [`L82`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L82) — Softmax feature map, backward pass [softmax(x), softmax(-x)]
- `sm2_fwd(x: Tensor)` — [`L74`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L74) — Softmax feature map, forward pass [softmax(x), softmax(-x)]
- `sm_bwd(y: Tensor, dy: Tensor)` — [`L68`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L68) — Softmax feature map, backward pass
- `sm_fwd(x: Tensor)` — [`L58`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L58) — Softmax feature map, forward pass
- `wrapper(*args)` — [`L48`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/utils.py#L48)

