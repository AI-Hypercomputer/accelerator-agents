---
title: 'Module: custom_kernel_spmd.py'
type: catalog
provenance: extracted
module: custom_kernel_spmd.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 custom_kernel_spmd/
symbols:
  q_segment_indexes: q_segment_indexes.
  o: o.
  mask: mask.
  q: q.
  device: device.
  k: k.
  v: v.
  forward_only: forward_only.
  batch_size: batch_size.
  SPMDFlashAttention: SPMDFlashAttention#
  minibatch: minibatch.
  device_ids: device_ids.
  i: i.
  attn_spec: attn_spec.
  args: args.
  fsdp_axis: fsdp_axis.
  mesh_shape: mesh_shape.
  parser: parser.
  sm_scale: sm_scale.
  SPMDFlashAttention.forward: SPMDFlashAttention#forward().
  end: end.
  sharding_spec: sharding_spec.
  SPMDFlashAttention.backward: SPMDFlashAttention#backward().
  ddp_axis: ddp_axis.
  mesh: mesh.
  new_tensor: new_tensor.
  fa_custom_backward: fa_custom_backward().
  loss: loss.
  mask_partition_spec: mask_partition_spec.
  mask_sharding_spec: mask_sharding_spec.
  requires_jax: requires_jax().
  _manual_mode: _manual_mode().
  model_axis: model_axis.
  sequence_axis: sequence_axis.
  q_seq: q_seq.
  k_seq: k_seq.
  depth: depth.
  num_heads: num_heads.
  requires_jax.wrapper: requires_jax().wrapper().
  _manual_mode.wrapped: _manual_mode().wrapped().
  fa_custom_forward: fa_custom_forward().
  no_rand: no_rand.
  flash_attention: flash_attention().
  batch_spec: batch_spec.
  fa_q_grad: fa_q_grad.
  fa_k_grad: fa_k_grad.
  fa_v_grad: fa_v_grad.
  fa_o: fa_o.
  _manual_mode._full_shape: _manual_mode()._full_shape().
  num_devices: num_devices.
  _jax_env_context: _jax_env_context().
  generate_ctx_need_grad: generate_ctx_need_grad().
  get_segment_partition_spec: get_segment_partition_spec().
  per_axis_batch_size: per_axis_batch_size.
  start: start.
---
# Module: [`custom_kernel_spmd.py`](../../../../raw/code/learning-machine/custom_kernel_spmd.py)

## Classes
### `SPMDFlashAttention`
- def: [`custom_kernel_spmd.py:514`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L514) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- members:
  - `backward(ctx, grad_output)` — [`L634`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L634)
  - `forward(ctx, q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale, ab, partition_spec, mesh)` — [`L522`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L522)
- uses (calls/refs, reference-scoped): [`fa_custom_backward`](custom_kernel_spmd.md#fa_custom_backward), [`_manual_mode`](custom_kernel_spmd.md#_manual_mode), [`requires_jax`](custom_kernel_spmd.md#requires_jax), [`fa_custom_forward`](custom_kernel_spmd.md#fa_custom_forward), [`generate_ctx_need_grad`](custom_kernel_spmd.md#generate_ctx_need_grad)
- used by: [`o`](custom_kernel_spmd.md#o), [`fa_custom_backward`](custom_kernel_spmd.md#fa_custom_backward), [`fa_custom_forward`](custom_kernel_spmd.md#fa_custom_forward), [`flash_attention`](custom_kernel_spmd.md#flash_attention)

## Functions
- `_full_shape(a, spec)` — [`L76`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L76)
- `_jax_env_context()` — [`L21`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L21)
- `_manual_mode(func, mesh, input_specs, output_specs)` — [`L69`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L69)
- `fa_custom_backward(grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, l: torch.Tensor, m: torch.Tensor, q_segment_ids: Optional[torch.Tensor], kv_segment_ids: Optional[torch.Tensor], ab: Optional[torch.Tensor], causal: bool, sm_scale: float, partition_spec: str, mesh: str, q_full_shape: List[int], kv_full_shape: List[int], ab_full_shape: Optional[List[int]], ctx_grad: List[bool], q_mid_shape, k_mid_shape, v_mid_shape, ab_mid_shape, q_segment_ids_mid_shape, kv_segment_ids_mid_shape, lm_mid_shape)` — [`L294`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L294)
- `fa_custom_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, q_segment_ids: torch.Tensor, kv_segment_ids: torch.Tensor, sm_scale: float, ab: Optional[torch.Tensor], ctx_grad: List[bool])` — [`L123`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L123)
- `flash_attention(q, k, v, causal=False, q_segment_ids=None, kv_segment_ids=None, sm_scale=1, *, ab=None, partition_spec=None, mesh=None)` — [`L486`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L486)
- `generate_ctx_need_grad(*args)` — [`L45`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L45)
- `get_segment_partition_spec(partition_sepc: Optional[Tuple[str, ...]] = None, ndim: int = 3)` — [`L53`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L53)
- `requires_jax(func: Callable)` — [`L29`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L29) — Decorator that ensures JAX is safely imported before function execution
- `wrapped(*args)` — [`L100`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L100)
- `wrapper(*args, **kwargs)` — [`L33`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L33)

## Module values
- `args` — [`L733`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L733) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `attn_spec` — [`L738`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L738)
- `batch_size` — [`L764`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L764) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `batch_spec` — [`L802`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L802)
- `ddp_axis` — [`L752`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L752) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `depth` — [`L768`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L768) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `device` — [`L815`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L815) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `device_ids` — [`L756`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L756)
- `end` — [`L861`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L861)
- `fa_k_grad` — [`L866`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L866)
- `fa_o` — [`L869`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L869) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `fa_q_grad` — [`L865`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L865) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `fa_v_grad` — [`L867`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L867)
- `forward_only` — [`L791`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L791) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `fsdp_axis` — [`L754`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L754) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `i` — [`L779`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L779)
- `k` — [`L785`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L785) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `k_seq` — [`L767`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L767) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `loss` — [`L857`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L857) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `mask` — [`L787`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L787) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `mask_partition_spec` — [`L804`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L804) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `mask_sharding_spec` — [`L806`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L806)
- `mesh` — [`L738`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L738)
- `mesh_shape` — [`L755`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L755)
- `minibatch` — [`L770`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L770) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `model_axis` — [`L751`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L751) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `new_tensor` — [`L783`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L783) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `no_rand` — [`L735`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L735) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `num_devices` — [`L750`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L750) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `num_heads` — [`L769`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L769) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `o` — [`L849`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L849) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `parser` — [`L721`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L721) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `per_axis_batch_size` — [`L763`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L763) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `q` — [`L784`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L784) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `q_segment_indexes` — [`L829`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L829) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `q_seq` — [`L766`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L766) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `sequence_axis` — [`L753`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L753) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `sharding_spec` — [`L809`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L809) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `sm_scale` — [`L823`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L823) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)
- `start` — [`L846`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L846)
- `v` — [`L786`](../../../../raw/code/learning-machine/custom_kernel_spmd.py#L786) — documented in [custom_kernel_spmd](../concepts/custom_kernel_spmd.md)

