---
title: 'Module: easydel/operations/kernels/ssm1.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/ssm1.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.ssm1`/
symbols:
  SSM1Op.get_requirements: SSM1Op#get_requirements().
  output_infer: output_infer.
  output: output.
  hidden_states: hidden_states.
  B: B.
  C: C.
  dt: dt.
  SSM1Output: SSM1Output#
  A: A.
  gate: gate.
  SSM1Op: SSM1Op#
  SSM1Op.forward_native: SSM1Op#forward_native().
  SSM1Output.ssm_state: SSM1Output#ssm_state.
  D: D.
  ssm_op: ssm_op.
  metadata: metadata.
  SSM1Op.forward_tpu: SSM1Op#forward_tpu().
  SSM1Op.forward_gpu: SSM1Op#forward_gpu().
  SSM1Op.forward_cpu: SSM1Op#forward_cpu().
  SSM1Op.__call__: SSM1Op#__call__().
  batch: batch.
  seq_len: seq_len.
  d: d.
  k6: k6.
  n: n.
  _single_step_ssm1_fwd: _single_step_ssm1_fwd().
  key: key.
  k1: k1.
  k2: k2.
  k3: k3.
  k4: k4.
  k5: k5.
  SSM1Output.conv_state: SSM1Output#conv_state.
  SSM1Op.get_impl_name: SSM1Op#get_impl_name().
---
# Module: [`easydel/operations/kernels/ssm1.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py)

## Classes
### `SSM1Op`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/ssm1.py:123`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L123)
- doc: SSM1 (Mamba1-style) selective state space operation.
- signature: `class SSM1Op(OperationImpl):`
- members:
  - `__call__(self, hidden_states: Float[Array, "batch seq_len intermediate_size"], A: Float[Array, "intermediate_size ssm_state_size"], B: Float[Array, "batch seq_len ssm_state_size"], C: Float[Array, "batch seq_len ssm_state_size"], D: Float[Array, intermediate_size], discrete_time_step: Float[Array, "batch seq_len intermediate_size"], gate: Float[Array, "batch seq_len intermediate_size"] | None = None, conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None, ssm_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None, activation: str = "silu", **kwargs)` — [`L260`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L260) — Execute the SSM1 operation.
  - `forward_cpu(self, *args, **kwargs)` — [`L256`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L256) — CPU forward pass. Delegates to forward_native.
  - `forward_gpu(self, *args, **kwargs)` — [`L252`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L252) — GPU forward pass. Delegates to forward_native.
  - `forward_native(self, hidden_states: Float[Array, "batch seq_len intermediate_size"], A: Float[Array, "intermediate_size ssm_state_size"], B: Float[Array, "batch seq_len ssm_state_size"], C: Float[Array, "batch seq_len ssm_state_size"], D: Float[Array, intermediate_size], discrete_time_step: Float[Array, "batch seq_len intermediate_size"], gate: Float[Array, "batch seq_len intermediate_size"] | None = None, conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None, ssm_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None, activation: str = "silu", **kwargs)` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L181) — Forward pass for SSM1 operation using ejKernel.
  - `forward_tpu(self, *args, **kwargs)` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L248) — TPU forward pass. Delegates to forward_native.
  - `get_impl_name(cls)` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L151) — Returns the registered names of this operation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L156) — Returns requirements for SSM1Op.
- uses (calls/refs, reference-scoped): [`ACT2FN`](../../infra/utils.md#ACT2FN), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`RecurrentCacheView`](../../caching/recurrent/cache.md#RecurrentCacheView), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`optional_metadata`](../requirements/builder.md#RequirementsBuilder.optional_metadata), [`use_cache_view`](../requirements/builder.md#RequirementsBuilder.use_cache_view), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`RECURRENT`](../requirements/types.md#CacheType.RECURRENT), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`SEQ_LENS`](../requirements/types.md#MetadataField.SEQ_LENS), [`LOGITS_INDICES`](../requirements/types.md#MetadataField.LOGITS_INDICES), [`SSM1Output`](ssm1.md#SSM1Output), [`STATE_INDICES`](../requirements/types.md#MetadataField.STATE_INDICES), [`HAS_INITIAL_STATE`](../requirements/types.md#MetadataField.HAS_INITIAL_STATE)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`ssm_op`](ssm1.md#ssm_op), [`ssm_op`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaMixer.ssm_op), [`ssm_op`](../../modules/mamba/modeling_mamba.md#MambaMixer.ssm_op)

### `SSM1Output`  ·  implements/extends AttentionOutput
- def: [`easydel/operations/kernels/ssm1.py:108`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L108)
- doc: Output container for SSM1 operation.
- signature: `class SSM1Output(AttentionOutput):`
- members:
  - `conv_state` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L118)
  - `ssm_state` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L119)
- uses (calls/refs, reference-scoped): [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`register`](../_base_operation.md#OperationRegistry.register)
- used by: [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`__call__`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaMixer.__call__), [`__call__`](../../modules/mamba/modeling_mamba.md#MambaMixer.__call__), [`output_infer`](ssm1.md#output_infer), [`output`](ssm1.md#output), [`forward_native`](ssm1.md#SSM1Op.forward_native), [`__call__`](ssm1.md#SSM1Op.__call__), [`forward_cpu`](ssm1.md#SSM1Op.forward_cpu), [`forward_gpu`](ssm1.md#SSM1Op.forward_gpu), [`forward_tpu`](ssm1.md#SSM1Op.forward_tpu)

## Functions
- `_single_step_ssm1_fwd(hidden_states: Float[Array, "batch intermediate_size"], A: Float[Array, "intermediate_size ssm_state_size"], B: Float[Array, "batch ssm_state_size"], C: Float[Array, "batch ssm_state_size"], D: Float[Array, intermediate_size], dt: Float[Array, "batch intermediate_size"], ssm_state: Float[Array, "batch intermediate_size ssm_state_size"])` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L65) — Single-step SSM1 (Mamba1) forward pass.

## Module values
- `A` — [`L321`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L321)
- `B` — [`L322`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L322)
- `C` — [`L323`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L323)
- `D` — [`L324`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L324)
- `batch` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L315)
- `d` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L315)
- `dt` — [`L325`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L325)
- `gate` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L326)
- `hidden_states` — [`L320`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L320)
- `k1` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L318)
- `k2` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L318)
- `k3` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L318)
- `k4` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L318)
- `k5` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L318)
- `k6` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L318)
- `key` — [`L317`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L317)
- `metadata` — [`L328`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L328)
- `n` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L315)
- `output` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L337)
- `output_infer` — [`L350`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L350)
- `seq_len` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L315)
- `ssm_op` — [`L334`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ssm1.py#L334)

