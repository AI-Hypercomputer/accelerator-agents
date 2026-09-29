---
title: 'Module: axlearn/common/rattention/kernels/linear_attention_kernels.py'
type: catalog
provenance: extracted
module: axlearn/common/rattention/kernels/linear_attention_kernels.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.rattention.kernels.linear_attention_kernels`/
symbols:
  _matmul_fp32: _matmul_fp32().
  residual_linear_attention_linear_scan: residual_linear_attention_linear_scan().
  residual_linear_attention_w_timestep: residual_linear_attention_w_timestep().
  _linear_attention_backward_kernel: _linear_attention_backward_kernel().
  residual_linear_attention: residual_linear_attention().
  right_shift_and_zero_pad: right_shift_and_zero_pad().
  _linear_attention_forward_kernel: _linear_attention_forward_kernel().
  _linear_attention_backward_kernel._la_backward_dq_chunk_loop_body: _linear_attention_backward_kernel()._la_backward_dq_chunk_loop_body().
  _linear_attention_backward_kernel._la_backward_dkv_chunk_loop_body: _linear_attention_backward_kernel()._la_backward_dkv_chunk_loop_body().
  _linear_attention: _linear_attention().
  _linear_attention_forward: _linear_attention_forward().
  _linear_attention_forward_kernel._la_forward_chunk_loop_body: _linear_attention_forward_kernel()._la_forward_chunk_loop_body().
  _la_backward: _la_backward().
  _inner_linear_scan: _inner_linear_scan().
  _inner_linear_scan.single_head_scan: _inner_linear_scan().single_head_scan().
  _inner_linear_scan.single_head_scan.scan_body_fn: _inner_linear_scan().single_head_scan().scan_body_fn().
  _linear_attention_forward_kernel.init_carry: _linear_attention_forward_kernel().init_carry().
  _linear_attention_backward_kernel.init_carry: _linear_attention_backward_kernel().init_carry().
---
# Module: [`axlearn/common/rattention/kernels/linear_attention_kernels.py`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py)

## Functions
- `_inner_linear_scan(q: Tensor, k: Tensor, v: Tensor, h0: Tensor)` — [`L556`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L556) — Inner function for linear scan attention.
- `_la_backward(feat_map: FeatureMap, chunk_size: int, residuals: tuple, do: Tensor)` — [`L386`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L386) — Backward pass for LA.
- `_la_backward_dkv_chunk_loop_body(t: int, carry_ref: Tensor)` — [`L342`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L342)
- `_la_backward_dq_chunk_loop_body(t: int, h_carry_ref: Tensor)` — [`L321`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L321)
- `_la_forward_chunk_loop_body(t: int, h_carry_ref: Tensor)` — [`L145`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L145)
- `_linear_attention(q: Tensor, k: Tensor, v: Tensor, h0: Tensor, feat_map: FeatureMap, chunk_size: int)` — [`L83`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L83) — A differentiable function that computes the output of linear attention.
- `_linear_attention_backward_kernel(q_ref: Tensor, k_ref: Tensor, v_ref: Tensor, ch_ref: Tensor, mutable_do_ref: Tensor, mutable_dq_ref: Tensor, mutable_dk_ref: Tensor, mutable_dv_ref: Tensor, mutable_dh_carry_ref: Tensor, *, feat_map: FeatureMap)` — [`L281`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L281) — Backward kernel for linear attention.
- `_linear_attention_forward(q: Tensor, k: Tensor, v: Tensor, initial_state: Tensor, feat_map: FeatureMap, chunk_size: int)` — [`L187`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L187) — Forward pass for linear attention.
- `_linear_attention_forward_kernel(q_ref: Tensor, k_ref: Tensor, v_ref: Tensor, initial_state_ref: Tensor, mutable_ch_ref: Tensor, mutable_final_state_ref: Tensor, mutable_o_ref: Tensor, *, feat_map: FeatureMap)` — [`L106`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L106) — Forward kernel for LA.
- `_matmul_fp32(lhs: Tensor, rhs: Tensor)` — [`L36`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L36) — A wrapper around jax.lax.dot to conduct float32 matmul
- `init_carry()` — [`L142`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L142)
- `init_carry()` — [`L318`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L318)
- `residual_linear_attention(q: Tensor, k: Tensor, v: Tensor, h0: Tensor, *, window_size: int, feat_map: FeatureMap, chunk_size: int)` — [`L514`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L514) — Differentiable function that computes the output of linear attention.
- `residual_linear_attention_linear_scan(q: Tensor, k: Tensor, v: Tensor, h0: Tensor, *, window_size: int, feat_map: FeatureMap, chunk_size: int)` — [`L611`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L611) — LinearScan based reference implementations.
- `residual_linear_attention_w_timestep(q: Tensor, k: Tensor, v: Tensor, h0: Tensor, timestep: Tensor, *, window_size: int, feat_map: FeatureMap, chunk_size: int)` — [`L649`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L649) — LinearScan that takes timestep as input and masks useless k/v based on timestep.
- `right_shift_and_zero_pad(x: Tensor, shift_size: int, axis: int = 1)` — [`L41`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L41) — Right shift the tensor and pad zeros on the left side, e.g., [2, 5, 3] with shift_size=2
- `scan_body_fn(h_prev, current_inputs)` — [`L581`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L581)
- `single_head_scan(q_head, k_head, v_head, h0_head)` — [`L580`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L580)

