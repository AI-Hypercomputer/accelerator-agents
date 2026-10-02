---
title: 'Module: easydel/operations/kernels/kda.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/kda.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.kda`/
symbols:
  KernelDeltaAttnOp.get_requirements: KernelDeltaAttnOp#get_requirements().
  KernelDeltaAttnOp.forward_native: KernelDeltaAttnOp#forward_native().
  output: output.
  output_infer: output_infer.
  query: query.
  key_tensor: key_tensor.
  value: value.
  state_recurrent: state_recurrent.
  state_chunk: state_chunk.
  beta: beta.
  decay: decay.
  KDAOutput: KDAOutput#
  _chunk_kda_fwd: _chunk_kda_fwd().
  max_diff: max_diff.
  KernelDeltaAttnOp: KernelDeltaAttnOp#
  _MATMUL_PRECISION: _MATMUL_PRECISION.
  kda_op: kda_op.
  num_heads: num_heads.
  l2norm: l2norm().
  _recurrent_kda_fwd: _recurrent_kda_fwd().
  metadata: metadata.
  fused_kda_gate: fused_kda_gate().
  KernelDeltaAttnOp.forward_tpu: KernelDeltaAttnOp#forward_tpu().
  KernelDeltaAttnOp.forward_gpu: KernelDeltaAttnOp#forward_gpu().
  KernelDeltaAttnOp.forward_cpu: KernelDeltaAttnOp#forward_cpu().
  KernelDeltaAttnOp.forward_cuda: KernelDeltaAttnOp#forward_cuda().
  KernelDeltaAttnOp.forward_rocm: KernelDeltaAttnOp#forward_rocm().
  KernelDeltaAttnOp.__call__: KernelDeltaAttnOp#__call__().
  batch: batch.
  seq_len: seq_len.
  _single_step_kda_core: _single_step_kda_core().
  KDAOutput.recurrent_state: KDAOutput#recurrent_state.
  _chunk_kda_fwd.resolve_single_chunk: _chunk_kda_fwd().resolve_single_chunk().
  _chunk_kda_fwd.chunk_step: _chunk_kda_fwd().chunk_step().
  _single_step_kda_fwd_bthd: _single_step_kda_fwd_bthd().
  query_single: query_single.
  key_single: key_single.
  value_single: value_single.
  beta_single: beta_single.
  decay_single: decay_single.
  k5: k5.
  _single_step_kda_fwd: _single_step_kda_fwd().
  head_dim: head_dim.
  KDAOutput.q_conv_state: KDAOutput#q_conv_state.
  KDAOutput.k_conv_state: KDAOutput#k_conv_state.
  KDAOutput.v_conv_state: KDAOutput#v_conv_state.
  _recurrent_kda_fwd.step_fn: _recurrent_kda_fwd().step_fn().
  _chunk_kda_fwd.resolve_intra_chunk_row: _chunk_kda_fwd().resolve_intra_chunk_row().
  key: key.
  d_state: d_state.
  k1: k1.
  k2: k2.
  k3: k3.
  k4: k4.
  out_recurrent: out_recurrent.
  out_chunk: out_chunk.
  KernelDeltaAttnOp.get_impl_name: KernelDeltaAttnOp#get_impl_name().
---
# Module: [`easydel/operations/kernels/kda.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py)

## Classes
### `KDAOutput`  ·  implements/extends AttentionOutput
- def: [`easydel/operations/kernels/kda.py:102`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L102)
- doc: Output container for KDA operation.
- signature: `class KDAOutput(AttentionOutput):`
- members:
  - `k_conv_state` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L118)
  - `q_conv_state` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L117)
  - `recurrent_state` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L120)
  - `v_conv_state` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L119)
- uses (calls/refs, reference-scoped): [`AttentionOutput`](../_attention_outputs.md#AttentionOutput)
- used by: [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.__call__), [`forward_native`](kda.md#KernelDeltaAttnOp.forward_native), [`output`](kda.md#output), [`output_infer`](kda.md#output_infer), [`__call__`](kda.md#KernelDeltaAttnOp.__call__), [`forward_cpu`](kda.md#KernelDeltaAttnOp.forward_cpu), [`forward_cuda`](kda.md#KernelDeltaAttnOp.forward_cuda), [`forward_gpu`](kda.md#KernelDeltaAttnOp.forward_gpu), [`forward_rocm`](kda.md#KernelDeltaAttnOp.forward_rocm), [`forward_tpu`](kda.md#KernelDeltaAttnOp.forward_tpu)

### `KernelDeltaAttnOp`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/kda.py:456`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L456)
- doc: Kernel Delta Attention (KDA) operation for Kimi Linear.
- signature: `class KernelDeltaAttnOp(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch seq_len num_heads head_dim"], key: Float[Array, "batch seq_len num_heads head_dim"], value: Float[Array, "batch seq_len num_heads d_state"], beta: Float[Array, "batch seq_len num_heads"], decay: Float[Array, "batch seq_len num_heads"] | None = None, q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None, recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, chunk_size: int = 64, **kwargs)` — [`L647`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L647) — Execute the KDA operation.
  - `forward_cpu(self, *args, **kwargs)` — [`L635`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L635) — CPU forward pass. Delegates to forward_native.
  - `forward_cuda(self, *args, **kwargs)` — [`L639`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L639) — CUDA forward pass. Delegates to forward_native.
  - `forward_gpu(self, *args, **kwargs)` — [`L631`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L631) — GPU forward pass. Delegates to forward_native.
  - `forward_native(self, query: Float[Array, "batch seq_len num_heads head_dim"], key: Float[Array, "batch seq_len num_heads head_dim"], value: Float[Array, "batch seq_len num_heads d_state"], beta: Float[Array, "batch seq_len num_heads"], decay: Float[Array, "batch seq_len num_heads"] | None = None, q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None, recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, chunk_size: int = 64, **kwargs)` — [`L523`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L523) — Forward pass for KDA attention.
  - `forward_rocm(self, *args, **kwargs)` — [`L643`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L643) — ROCm forward pass. Delegates to forward_native.
  - `forward_tpu(self, *args, **kwargs)` — [`L627`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L627) — TPU forward pass. Delegates to forward_native.
  - `get_impl_name(cls)` — [`L488`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L488) — Returns the registered name of this operation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L497`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L497) — Returns requirements for KernelDeltaAttnOp (KDA).
- uses (calls/refs, reference-scoped): [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`KDACacheView`](../../caching/kda/cache.md#KDACacheView), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`query`](../_operation_meta.md#AttnShardingRules.query), [`optional_metadata`](../requirements/builder.md#RequirementsBuilder.optional_metadata), [`use_cache_view`](../requirements/builder.md#RequirementsBuilder.use_cache_view), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`RECURRENT`](../requirements/types.md#CacheType.RECURRENT), [`KDAOutput`](kda.md#KDAOutput), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`output`](../_operation_meta.md#AttnShardingRules.output), [`SEQ_LENS`](../requirements/types.md#MetadataField.SEQ_LENS), [`_chunk_kda_fwd`](kda.md#_chunk_kda_fwd), [`LOGITS_INDICES`](../requirements/types.md#MetadataField.LOGITS_INDICES), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`STATE_INDICES`](../requirements/types.md#MetadataField.STATE_INDICES), [`HAS_INITIAL_STATE`](../requirements/types.md#MetadataField.HAS_INITIAL_STATE), [`_single_step_kda_fwd_bthd`](kda.md#_single_step_kda_fwd_bthd)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`kda_op`](kda.md#kda_op), [`kda_op`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.kda_op)

## Functions
- `_chunk_kda_fwd(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True)` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L197) — Chunked forward pass for KDA.
- `_recurrent_kda_fwd(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True)` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L123) — Recurrent forward pass for KDA.
- `_single_step_kda_core(query: Float[Array, "batch num_heads head_dim"], key: Float[Array, "batch num_heads head_dim"], value: Float[Array, "batch num_heads d_state"], beta: Float[Array, "batch num_heads"], decay: Float[Array, "batch num_heads"] | None, recurrent_state: Float[Array, "batch num_heads head_dim d_state"], use_qk_l2norm: bool = True)` — [`L416`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L416) — Shared single-step KDA math over squeezed decode tensors.
- `_single_step_kda_fwd(query: Float[Array, "batch num_heads 1 head_dim"], key: Float[Array, "batch num_heads 1 head_dim"], value: Float[Array, "batch num_heads 1 d_state"], beta: Float[Array, "batch num_heads 1"], decay: Float[Array, "batch num_heads 1"] | None, recurrent_state: Float[Array, "batch num_heads head_dim d_state"], use_qk_l2norm: bool = True)` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L366) — Single-step recurrent forward pass for BHTD-layout decode inputs.
- `_single_step_kda_fwd_bthd(query: Float[Array, "batch 1 num_heads head_dim"], key: Float[Array, "batch 1 num_heads head_dim"], value: Float[Array, "batch 1 num_heads d_state"], beta: Float[Array, "batch 1 num_heads"], decay: Float[Array, "batch 1 num_heads"] | None, recurrent_state: Float[Array, "batch num_heads head_dim d_state"], use_qk_l2norm: bool = True)` — [`L391`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L391) — Single-step recurrent forward pass for BTHD-layout decode inputs.
- `chunk_step(state, inputs)` — [`L323`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L323)
- `fused_kda_gate(gate: Float[Array, ...], A_log: Float[Array, num_heads], dt_bias: Float[Array, num_heads])` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L78) — Compute KDA decay gate.
- `l2norm(x, axis=-1, eps=0.000001)` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L69) — L2 normalize along specified axis.
- `resolve_intra_chunk_row(attn_chunk, i)` — [`L282`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L282)
- `resolve_single_chunk(attn_single)` — [`L293`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L293)
- `step_fn(state, inputs)` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L178)

## Module values
- `_MATMUL_PRECISION` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L66)
- `batch` — [`L703`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L703)
- `beta` — [`L711`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L711)
- `beta_single` — [`L739`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L739)
- `d_state` — [`L703`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L703)
- `decay` — [`L712`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L712)
- `decay_single` — [`L740`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L740)
- `head_dim` — [`L703`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L703)
- `k1` — [`L706`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L706)
- `k2` — [`L706`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L706)
- `k3` — [`L706`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L706)
- `k4` — [`L706`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L706)
- `k5` — [`L706`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L706)
- `kda_op` — [`L720`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L720)
- `key` — [`L705`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L705)
- `key_single` — [`L737`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L737)
- `key_tensor` — [`L709`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L709)
- `max_diff` — [`L776`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L776)
- `metadata` — [`L714`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L714)
- `num_heads` — [`L703`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L703)
- `out_chunk` — [`L764`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L764)
- `out_recurrent` — [`L755`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L755)
- `output` — [`L723`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L723)
- `output_infer` — [`L742`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L742)
- `query` — [`L708`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L708)
- `query_single` — [`L736`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L736)
- `seq_len` — [`L703`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L703)
- `state_chunk` — [`L764`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L764)
- `state_recurrent` — [`L755`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L755)
- `value` — [`L710`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L710)
- `value_single` — [`L738`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/kda.py#L738)

