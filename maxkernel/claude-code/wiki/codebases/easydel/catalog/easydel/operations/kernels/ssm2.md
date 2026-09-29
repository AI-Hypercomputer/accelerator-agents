---
title: 'Module: easydel/operations/kernels/ssm2.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/ssm2.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.ssm2`/
symbols:
  SSM2Op.get_requirements: SSM2Op#get_requirements().
  output_infer: output_infer.
  output: output.
  output_norm: output_norm.
  x: x.
  B: B.
  C: C.
  gate: gate.
  dt: dt.
  num_heads: num_heads.
  SSM2Output: SSM2Output#
  SSM2Op: SSM2Op#
  SSM2Op.forward_native: SSM2Op#forward_native().
  A: A.
  D: D.
  ssm_op: ssm_op.
  SSM2Output.ssm_state: SSM2Output#ssm_state.
  SSM2Op.__call__: SSM2Op#__call__().
  metadata: metadata.
  SSM2Op.forward_tpu: SSM2Op#forward_tpu().
  SSM2Op.forward_gpu: SSM2Op#forward_gpu().
  SSM2Op.forward_cpu: SSM2Op#forward_cpu().
  batch: batch.
  seq_len: seq_len.
  n_groups: n_groups.
  _single_step_ssm2_fwd: _single_step_ssm2_fwd().
  k7: k7.
  head_dim: head_dim.
  ssm_state_size: ssm_state_size.
  key: key.
  k1: k1.
  k2: k2.
  k3: k3.
  k4: k4.
  k5: k5.
  k6: k6.
  SSM2Output.conv_state: SSM2Output#conv_state.
  SSM2Op.get_impl_name: SSM2Op#get_impl_name().
---
# Module: [`easydel/operations/kernels/ssm2.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py)

## Classes
### `SSM2Op`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/ssm2.py:132`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L132)
- doc: SSM2 (Mamba2-style) selective state space operation.
- signature: `class SSM2Op(OperationImpl):`
- members:
  - `__call__(self, x: Float[Array, "batch seq_len num_heads head_dim"], A: Float[Array, num_heads], B: Float[Array, "batch seq_len n_groups ssm_state_size"], C: Float[Array, "batch seq_len n_groups ssm_state_size"], D: Float[Array, num_heads], dt: Float[Array, "batch seq_len num_heads"], gate: Float[Array, "batch seq_len intermediate_size"] | None = None, conv_state: Float[Array, "batch conv_dim d_conv"] | None = None, ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None, n_groups: int = 1, use_gated_rmsnorm: bool = False, rmsnorm_eps: float = 0.00001, precision: lax.Precision | None = None, **kwargs)` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L274) — Execute the SSM2 operation.
  - `forward_cpu(self, *args, **kwargs)` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L270) — CPU forward pass. Delegates to forward_native.
  - `forward_gpu(self, *args, **kwargs)` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L266) — GPU forward pass. Delegates to forward_native.
  - `forward_native(self, x: Float[Array, "batch seq_len num_heads head_dim"], A: Float[Array, num_heads], B: Float[Array, "batch seq_len n_groups ssm_state_size"], C: Float[Array, "batch seq_len n_groups ssm_state_size"], D: Float[Array, num_heads], dt: Float[Array, "batch seq_len num_heads"], gate: Float[Array, "batch seq_len intermediate_size"] | None = None, conv_state: Float[Array, "batch conv_dim d_conv"] | None = None, ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None, n_groups: int = 1, use_gated_rmsnorm: bool = False, rmsnorm_eps: float = 0.00001, precision: lax.Precision | None = None, **kwargs)` — [`L190`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L190) — Forward pass for SSM2 operation using ejKernel.
  - `forward_tpu(self, *args, **kwargs)` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L262) — TPU forward pass. Delegates to forward_native.
  - `get_impl_name(cls)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L160) — Returns the registered names of this operation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L165) — Returns requirements for SSM2Op.
- uses (calls/refs, reference-scoped): [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`RecurrentCacheView`](../../caching/recurrent/cache.md#RecurrentCacheView), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`optional_metadata`](../requirements/builder.md#RequirementsBuilder.optional_metadata), [`use_cache_view`](../requirements/builder.md#RequirementsBuilder.use_cache_view), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`RECURRENT`](../requirements/types.md#CacheType.RECURRENT), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`SEQ_LENS`](../requirements/types.md#MetadataField.SEQ_LENS), [`LOGITS_INDICES`](../requirements/types.md#MetadataField.LOGITS_INDICES), [`num_heads`](ssm2.md#num_heads), [`SSM2Output`](ssm2.md#SSM2Output), [`STATE_INDICES`](../requirements/types.md#MetadataField.STATE_INDICES), [`HAS_INITIAL_STATE`](../requirements/types.md#MetadataField.HAS_INITIAL_STATE)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`ssm_op`](ssm2.md#ssm_op), [`ssm_op`](../../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.ssm_op), [`ssm_op`](../../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.ssm_op)

### `SSM2Output`  ·  implements/extends AttentionOutput
- def: [`easydel/operations/kernels/ssm2.py:117`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L117)
- doc: Output container for SSM2 operation.
- signature: `class SSM2Output(AttentionOutput):`
- members:
  - `conv_state` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L127)
  - `ssm_state` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L128)
- uses (calls/refs, reference-scoped): [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`register`](../_base_operation.md#OperationRegistry.register)
- used by: [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`__call__`](../../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.__call__), [`__call__`](../../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.__call__), [`output_infer`](ssm2.md#output_infer), [`output`](ssm2.md#output), [`forward_native`](ssm2.md#SSM2Op.forward_native), [`__call__`](ssm2.md#SSM2Op.__call__), [`forward_cpu`](ssm2.md#SSM2Op.forward_cpu), [`forward_gpu`](ssm2.md#SSM2Op.forward_gpu), [`forward_tpu`](ssm2.md#SSM2Op.forward_tpu)

## Functions
- `_single_step_ssm2_fwd(x: Float[Array, "batch num_heads head_dim"], A: Float[Array, num_heads], B: Float[Array, "batch n_groups ssm_state_size"], C: Float[Array, "batch n_groups ssm_state_size"], D: Float[Array, num_heads], dt: Float[Array, "batch num_heads"], ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"], n_groups: int = 1)` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L66) — Single-step SSM2 (Mamba2) forward pass.

## Module values
- `A` — [`L344`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L344)
- `B` — [`L345`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L345)
- `C` — [`L346`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L346)
- `D` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L347)
- `batch` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L338)
- `dt` — [`L348`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L348)
- `gate` — [`L349`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L349)
- `head_dim` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L338)
- `k1` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `k2` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `k3` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `k4` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `k5` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `k6` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `k7` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L341)
- `key` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L340)
- `metadata` — [`L351`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L351)
- `n_groups` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L338)
- `num_heads` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L338)
- `output` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L360)
- `output_infer` — [`L374`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L374)
- `output_norm` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L389)
- `seq_len` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L338)
- `ssm_op` — [`L357`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L357)
- `ssm_state_size` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L338)
- `x` — [`L343`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm2.py#L343)

