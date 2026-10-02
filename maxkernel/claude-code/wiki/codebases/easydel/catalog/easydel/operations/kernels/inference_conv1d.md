---
title: 'Module: easydel/operations/kernels/inference_conv1d.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/inference_conv1d.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.inference_conv1d`/
symbols:
  RaggedCausalConv1D.get_requirements: RaggedCausalConv1D#get_requirements().
  RaggedCausalConv1D.forward_native: RaggedCausalConv1D#forward_native().
  ragged_causal_conv1d: ragged_causal_conv1d().
  RaggedCausalConv1D: RaggedCausalConv1D#
  RaggedCausalConv1D.forward_tpu: RaggedCausalConv1D#forward_tpu().
  RaggedCausalConv1D.forward_gpu: RaggedCausalConv1D#forward_gpu().
  RaggedCausalConv1D.forward_cpu: RaggedCausalConv1D#forward_cpu().
  RaggedCausalConv1D.forward_cuda: RaggedCausalConv1D#forward_cuda().
  RaggedCausalConv1D.forward_rocm: RaggedCausalConv1D#forward_rocm().
  RaggedCausalConv1D.get_impl_name: RaggedCausalConv1D#get_impl_name().
---
# Module: [`easydel/operations/kernels/inference_conv1d.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py)

## Classes
### `RaggedCausalConv1D`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/inference_conv1d.py:226`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L226)
- doc: Ragged causal depthwise conv1d operation for packed inference batches.
- signature: `class RaggedCausalConv1D(OperationImpl):`
- members:
  - `forward_cpu(self, *args, **kwargs)` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L360) — CPU forward — delegates to :meth:`forward_native`.
  - `forward_cuda(self, *args, **kwargs)` — [`L368`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L368) — CUDA forward — alias of :meth:`forward_gpu` for NVIDIA devices.
  - `forward_gpu(self, *args, **kwargs)` — [`L352`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L352) — GPU forward — delegates to :meth:`forward_native`.
  - `forward_native(self, x: jnp.ndarray, conv_state: jnp.ndarray, kernel: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, d_conv: int, apply_silu: bool = True)` — [`L296`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L296) — Primary forward implementation; thin wrapper over the free function.
  - `forward_rocm(self, *args, **kwargs)` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L372) — ROCm forward — alias of :meth:`forward_gpu` for AMD devices.
  - `forward_tpu(self, *args, **kwargs)` — [`L343`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L343) — TPU forward — delegates to :meth:`forward_native`.
  - `get_impl_name(cls)` — [`L257`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L257) — Return the registry name for this operation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L267`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L267) — Describe the metadata and cache requirements of this op.
- uses (calls/refs, reference-scoped): [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`RECURRENT`](../requirements/types.md#CacheType.RECURRENT), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`SEQ_LENS`](../requirements/types.md#MetadataField.SEQ_LENS), [`STATE_INDICES`](../requirements/types.md#MetadataField.STATE_INDICES), [`ragged_causal_conv1d`](inference_conv1d.md#ragged_causal_conv1d)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm)

## Functions
- `ragged_causal_conv1d(x: jnp.ndarray, conv_state: jnp.ndarray, kernel: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, d_conv: int, apply_silu: bool = True)` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_conv1d.py#L53) — Applies causal depthwise conv1d over ragged sequences with rolling state.

