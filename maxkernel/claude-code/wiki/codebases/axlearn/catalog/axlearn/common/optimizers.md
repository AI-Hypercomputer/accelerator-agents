---
title: 'Module: axlearn/common/optimizers.py'
type: catalog
provenance: extracted
module: axlearn/common/optimizers.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.optimizers`/
symbols:
  adastar_optimizer: adastar_optimizer().
  scale_by_schedule: scale_by_schedule().
  skip_and_clip_by_global_norm.partition_fn: skip_and_clip_by_global_norm().partition_fn().
  adastar_optimizer.update_fn: adastar_optimizer().update_fn().
  adafactor_optimizer: adafactor_optimizer().
  skip_and_clip_by_global_norm.update_fn: skip_and_clip_by_global_norm().update_fn().
  ema.partition_fn: ema().partition_fn().
  adamw_optimizer: adamw_optimizer().
  add_decayed_weights: add_decayed_weights().
  adamw_decoupled_optimizer: adamw_decoupled_optimizer().
  param_ema: param_ema().
  copy_partition: copy_partition().
  param_ema.partition_fn: param_ema().partition_fn().
  scale_by_lion.partition_fn: scale_by_lion().partition_fn().
  skip_and_clip_by_global_norm: skip_and_clip_by_global_norm().
  offload_optimizer: offload_optimizer().
  chain: chain().
  sgd_optimizer: sgd_optimizer().
  ema: ema().
  adastar_optimizer.update_fn._raw_updates: adastar_optimizer().update_fn()._raw_updates().
  adastar_optimizer.update_fn._smoothed_updates: adastar_optimizer().update_fn()._smoothed_updates().
  adastar_optimizer.partition_fn._partition: adastar_optimizer().partition_fn()._partition().
  _weight_decay_scales: _weight_decay_scales().
  with_partition_fn: with_partition_fn().
  lion_optimizer: lion_optimizer().
  named_chain: named_chain().
  skip_and_clip_by_global_norm.init_fn: skip_and_clip_by_global_norm().init_fn().
  adastar_optimizer.partition_fn: adastar_optimizer().partition_fn().
  adam_optimizer: adam_optimizer().
  adam_partition.partition_fn: adam_partition().partition_fn().
  add_decayed_weights.update_fn: add_decayed_weights().update_fn().
  l2_regularizer: l2_regularizer().
  scale_by_lion: scale_by_lion().
  scale_update_per_param: scale_update_per_param().
  scale_from_learning_rate: scale_from_learning_rate().
  ema._to_state: ema()._to_state().
  _log_per_layer_stats: _log_per_layer_stats().
  scale_update_per_param.update_fn: scale_update_per_param().update_fn().
  add_decayed_weights.partition_fn: add_decayed_weights().partition_fn().
  ema.update_fn._to_tensor_ema: ema().update_fn()._to_tensor_ema().
  ema.update_fn._update: ema().update_fn()._update().
  ema.partition_fn.get_scale_partition: ema().partition_fn().get_scale_partition().
  clip_by_global_norm: clip_by_global_norm().
  adastar_optimizer.init_fn._init: adastar_optimizer().init_fn()._init().
  adastar_optimizer.update2_fn._update2: adastar_optimizer().update2_fn()._update2().
  _no_op: _no_op().
  ema.partition_fn.get_ema_partition: ema().partition_fn().get_ema_partition().
  adastar_optimizer.init_fn: adastar_optimizer().init_fn().
  offload_optimizer.partition_fn: offload_optimizer().partition_fn().
  replace_with_updates: replace_with_updates().
  _compute_rms_norms: _compute_rms_norms().
  scale_by_param_block_rms: scale_by_param_block_rms().
  adastar_optimizer.update_fn._split_update_results: adastar_optimizer().update_fn()._split_update_results().
  trace_partition.partition_fn: trace_partition().partition_fn().
  scale_by_trust_ratio: scale_by_trust_ratio().
  ema.init_fn._init: ema().init_fn()._init().
  ema.update_fn._to_qint_tensor_ema: ema().update_fn()._to_qint_tensor_ema().
  param_ema.update_fn: param_ema().update_fn().
  adastar_optimizer.update2_fn: adastar_optimizer().update2_fn().
  adam_partition: adam_partition().
  SkipClipState.count: SkipClipState#count.
  SkipClipState.nonvalid_count: SkipClipState#nonvalid_count.
  SkipClipState.grad_norm_ema: SkipClipState#grad_norm_ema.
  SkipClipState.grad_norm_square_ema: SkipClipState#grad_norm_square_ema.
  SkipClipState.drop_stats: SkipClipState#drop_stats.
  _to_partitioned_transformation: _to_partitioned_transformation().
  trace_partition: trace_partition().
  _compute_covariance: _compute_covariance().
  clip_by_block_rms: clip_by_block_rms().
  opt_param_values: opt_param_values().
  ParamEmaState.count: ParamEmaState#count.
  adastar_optimizer._AdastarPerParamState: adastar_optimizer()._AdastarPerParamState#
  adastar_optimizer._AdastarState.pps: adastar_optimizer()._AdastarState#pps.
  named_chain.update_fn: named_chain().update_fn().
  with_partition_fn.init_fn: with_partition_fn().init_fn().
  per_param_scale_by_path.fn: per_param_scale_by_path().fn().
  l2_regularizer.update_fn: l2_regularizer().update_fn().
  skip_and_clip_by_global_norm.update_fn._is_valid_step: skip_and_clip_by_global_norm().update_fn()._is_valid_step().
  clip_by_block_rms.update_fn: clip_by_block_rms().update_fn().
  param_ema.init_fn: param_ema().init_fn().
  scale_by_lion.init_fn: scale_by_lion().init_fn().
  scale_by_lion.update_fn: scale_by_lion().update_fn().
  adastar_optimizer._AdastarPerParamState.gradient_ema: adastar_optimizer()._AdastarPerParamState#gradient_ema.
  adastar_optimizer._AdastarPerParamState.gradient_square_ema: adastar_optimizer()._AdastarPerParamState#gradient_square_ema.
  adastar_optimizer._AdastarPerParamState.update_ema: adastar_optimizer()._AdastarPerParamState#update_ema.
  offload_optimizer.update_fn: offload_optimizer().update_fn().
  per_param_scale_by_rms: per_param_scale_by_rms().
  scale: scale().
  adastar_optimizer._AdastarUpdateResult: adastar_optimizer()._AdastarUpdateResult#
  adastar_optimizer._AdastarUpdateResult.pps: adastar_optimizer()._AdastarUpdateResult#pps.
  offload_optimizer._move_fn: offload_optimizer()._move_fn().
  AddDecayedWeightsState: AddDecayedWeightsState#
  ParamEmaState.ema: ParamEmaState#ema.
  adastar_optimizer._AdastarState.count: adastar_optimizer()._AdastarState#count.
  _no_op.update_fn: _no_op().update_fn().
  with_partition_fn.update_fn: with_partition_fn().update_fn().
  replace_with_updates.update_fn: replace_with_updates().update_fn().
  scale_by_schedule.update_fn: scale_by_schedule().update_fn().
  scale_from_learning_rate.scale_fn: scale_from_learning_rate().scale_fn().
  scale_by_trust_ratio.update_fn: scale_by_trust_ratio().update_fn().
  _weight_decay_scales.maybe_override_scale: _weight_decay_scales().maybe_override_scale().
  add_decayed_weights.init_fn: add_decayed_weights().init_fn().
  ema._TensorEma: ema()._TensorEma#
  ema.init_fn: ema().init_fn().
  ema.update_fn: ema().update_fn().
  clip_by_global_norm.update_fn: clip_by_global_norm().update_fn().
  SkipClipState: SkipClipState#
  ParamEmaState: ParamEmaState#
  offload_optimizer.init_fn: offload_optimizer().init_fn().
  ema._TensorEma.value: ema()._TensorEma#value.
  ema._TensorEma.qstep_size: ema()._TensorEma#qstep_size.
  adastar_optimizer.update_fn._moment: adastar_optimizer().update_fn()._moment().
  per_param_scale_by_path: per_param_scale_by_path().
  drop_norm_by_grad_norm_ema: drop_norm_by_grad_norm_ema().
  drop_norm_by_grad_norm_stddev: drop_norm_by_grad_norm_stddev().
  SkipClipState.inner_state: SkipClipState#inner_state.
  EmaState.count: EmaState#count.
  EmaState.ema: EmaState#ema.
  EmaState.scale: EmaState#scale.
  skip_and_clip_by_global_norm.update_fn._stddev: skip_and_clip_by_global_norm().update_fn()._stddev().
  adastar_optimizer._AdastarUpdateResult.updates: adastar_optimizer()._AdastarUpdateResult#updates.
  DropNormThresholdFn: DropNormThresholdFn#
  ScaleByLionState: ScaleByLionState#
  adastar_optimizer._AdastarState: adastar_optimizer()._AdastarState#
  chain.partition: chain().partition().
  named_chain.init_fn: named_chain().init_fn().
  named_chain.partition_fn: named_chain().partition_fn().
  per_param_scale_by_rms.fn: per_param_scale_by_rms().fn().
  ema._UpdateResult.update: ema()._UpdateResult#update.
  ema._UpdateResult.tensor_ema: ema()._UpdateResult#tensor_ema.
  ema.update_fn._to_float: ema().update_fn()._to_float().
  drop_norm_by_grad_norm_ema.fn: drop_norm_by_grad_norm_ema().fn().
  drop_norm_by_grad_norm_stddev.fn: drop_norm_by_grad_norm_stddev().fn().
  skip_and_clip_by_global_norm.update_fn._moment: skip_and_clip_by_global_norm().update_fn()._moment().
  scale_by_param_block_rms.update_fn: scale_by_param_block_rms().update_fn().
  AddDecayedWeightsState.count: AddDecayedWeightsState#count.
  EmaState: EmaState#
  ema._should_quantize: ema()._should_quantize().
  ScaleByLionState.count: ScaleByLionState#count.
  ScaleByLionState.mu: ScaleByLionState#mu.
  DropNormThresholdFn.__call__: DropNormThresholdFn#__call__().
  ema._UpdateResult: ema()._UpdateResult#
  scale_by_schedule.init_fn: scale_by_schedule().init_fn().
  per_param_scale_by_path.fn.per_param_fn: per_param_scale_by_path().fn().per_param_fn().
  scale_by_trust_ratio.init_fn: scale_by_trust_ratio().init_fn().
  scale_by_trust_ratio.update_fn._scale_update: scale_by_trust_ratio().update_fn()._scale_update().
  scale_update_per_param.init_fn: scale_update_per_param().init_fn().
  add_decayed_weights.update_fn.f: add_decayed_weights().update_fn().f().
  l2_regularizer.init_fn: l2_regularizer().init_fn().
  clip_by_global_norm.init_fn: clip_by_global_norm().init_fn().
  clip_by_block_rms.init_fn: clip_by_block_rms().init_fn().
  clip_by_block_rms.update_fn._clip_fn: clip_by_block_rms().update_fn()._clip_fn().
  scale_by_param_block_rms.init_fn: scale_by_param_block_rms().init_fn().
  param_ema.update_fn._interpolate: param_ema().update_fn()._interpolate().
---
# Module: [`axlearn/common/optimizers.py`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py)

## Classes
### `AddDecayedWeightsState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/optimizers.py:463`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L463)
- signature: `class AddDecayedWeightsState(NamedTuple):`
- members:
  - `count` — [`L464`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L464)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`update_fn`](optimizers.md#add_decayed_weights.update_fn), [`partition_fn`](optimizers.md#add_decayed_weights.partition_fn), [`init_fn`](optimizers.md#add_decayed_weights.init_fn)

### `DropNormThresholdFn`  ·  implements/extends Protocol
- def: [`axlearn/common/optimizers.py:1187`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1187)
- doc: Protocol for drop norm threshold function.
- signature: `class DropNormThresholdFn(typing_extensions.Protocol):`
- members:
  - `__call__(self, *, count: Tensor, mean: Tensor, stddev: Tensor)` — [`L1190`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1190) — Returns the drop_norm thresholds given the gradient norm stats.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`skip_and_clip_by_global_norm`](optimizers.md#skip_and_clip_by_global_norm), [`_is_valid_step`](optimizers.md#skip_and_clip_by_global_norm.update_fn._is_valid_step), [`drop_norm_by_grad_norm_ema`](optimizers.md#drop_norm_by_grad_norm_ema), [`drop_norm_by_grad_norm_stddev`](optimizers.md#drop_norm_by_grad_norm_stddev)

### `EmaState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/optimizers.py:881`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L881)
- signature: `class EmaState(NamedTuple):`
- members:
  - `count` — [`L883`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L883)
  - `ema` — [`L884`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L884)
  - `scale` — [`L885`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L885)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTree`](utils.md#NestedTree)
- used by: [`partition_fn`](optimizers.md#ema.partition_fn), [`_to_state`](optimizers.md#ema._to_state)

### `ParamEmaState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/optimizers.py:1585`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1585)
- doc: Captures an exponential moving average for all params.
- signature: `class ParamEmaState(NamedTuple):`
- members:
  - `count` — [`L1588`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1588)
  - `ema` — [`L1589`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1589)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`TensorSpec`](utils.md#TensorSpec), [`NestedTree`](utils.md#NestedTree)
- used by: [`target_to_source`](state_builder.md#EmaParamsConverter.target_to_source), [`partition_fn`](optimizers.md#param_ema.partition_fn), [`update_fn`](optimizers.md#param_ema.update_fn), [`init_fn`](optimizers.md#param_ema.init_fn)

### `ScaleByLionState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/optimizers.py:1655`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1655)
- doc: State for the Lion algorithm.
- signature: `class ScaleByLionState(NamedTuple):`
- members:
  - `count` — [`L1658`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1658)
  - `mu` — [`L1659`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1659)
- used by: [`partition_fn`](optimizers.md#scale_by_lion.partition_fn), [`init_fn`](optimizers.md#scale_by_lion.init_fn), [`update_fn`](optimizers.md#scale_by_lion.update_fn)

### `SkipClipState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/optimizers.py:1239`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1239)
- doc: State returned by functions in skip_and_clip_by_global_norm().
- signature: `class SkipClipState(NamedTuple):`
- members:
  - `count` — [`L1242`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1242)
  - `drop_stats` — [`L1249`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1249)
  - `grad_norm_ema` — [`L1244`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1244)
  - `grad_norm_square_ema` — [`L1245`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1245)
  - `inner_state` — [`L1248`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1248)
  - `nonvalid_count` — [`L1243`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1243)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`TensorSpec`](utils.md#TensorSpec)
- used by: [`partition_fn`](optimizers.md#skip_and_clip_by_global_norm.partition_fn), [`update_fn`](optimizers.md#skip_and_clip_by_global_norm.update_fn), [`init_fn`](optimizers.md#skip_and_clip_by_global_norm.init_fn)

### `_AdastarPerParamState`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/optimizers.py:1859`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1859)
- signature: `class _AdastarPerParamState(flax_struct.PyTreeNode):`
- members:
  - `gradient_ema` — [`L1860`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1860)
  - `gradient_square_ema` — [`L1861`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1861)
  - `update_ema` — [`L1862`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1862)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`_partition`](optimizers.md#adastar_optimizer.partition_fn._partition), [`_raw_updates`](optimizers.md#adastar_optimizer.update_fn._raw_updates), [`_smoothed_updates`](optimizers.md#adastar_optimizer.update_fn._smoothed_updates), [`_init`](optimizers.md#adastar_optimizer.init_fn._init), [`_split_update_results`](optimizers.md#adastar_optimizer.update_fn._split_update_results), [`pps`](optimizers.md#adastar_optimizer._AdastarState.pps), [`pps`](optimizers.md#adastar_optimizer._AdastarUpdateResult.pps)

### `_AdastarState`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/optimizers.py:1864`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1864)
- signature: `class _AdastarState(flax_struct.PyTreeNode):`
- members:
  - `count` — [`L1865`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1865)
  - `pps` — [`L1866`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1866)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`_AdastarPerParamState`](optimizers.md#adastar_optimizer._AdastarPerParamState)
- used by: [`update_fn`](optimizers.md#adastar_optimizer.update_fn), [`partition_fn`](optimizers.md#adastar_optimizer.partition_fn), [`init_fn`](optimizers.md#adastar_optimizer.init_fn)

### `_AdastarUpdateResult`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/optimizers.py:1868`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1868)
- doc: Opaque container that is not traversed by jax.tree.map.
- signature: `class _AdastarUpdateResult(flax_struct.PyTreeNode):`
- members:
  - `pps` — [`L1872`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1872)
  - `updates` — [`L1871`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1871)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`as_schedule_fn`](schedule.md#as_schedule_fn), [`_AdastarPerParamState`](optimizers.md#adastar_optimizer._AdastarPerParamState)
- used by: [`_raw_updates`](optimizers.md#adastar_optimizer.update_fn._raw_updates), [`_smoothed_updates`](optimizers.md#adastar_optimizer.update_fn._smoothed_updates), [`_split_update_results`](optimizers.md#adastar_optimizer.update_fn._split_update_results)

### `_TensorEma`
- def: [`axlearn/common/optimizers.py:926`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L926)
- signature: `class _TensorEma:`
- members:
  - `qstep_size` — [`L929`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L929)
  - `value` — [`L928`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L928)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`_to_tensor_ema`](optimizers.md#ema.update_fn._to_tensor_ema), [`_init`](optimizers.md#ema.init_fn._init), [`_to_qint_tensor_ema`](optimizers.md#ema.update_fn._to_qint_tensor_ema), [`tensor_ema`](optimizers.md#ema._UpdateResult.tensor_ema)

### `_UpdateResult`
- def: [`axlearn/common/optimizers.py:956`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L956)
- signature: `class _UpdateResult:`
- members:
  - `tensor_ema` — [`L959`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L959)
  - `update` — [`L958`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L958)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`_TensorEma`](optimizers.md#ema._TensorEma)
- used by: [`_update`](optimizers.md#ema.update_fn._update)

## Functions
- `_clip_fn(u, norm)` — [`L1539`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1539)
- `_compute_covariance(x: NestedTensor, y: NestedTensor, *, summary_suffix: Optional[str] = None)` — [`L434`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L434) — Computes the covariance between leaf tensors in `x` and `y` and optionally adds summaries.
- `_compute_rms_norms(x: NestedTensor, *, summary_suffix: Optional[str] = None)` — [`L412`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L412) — Computes the RMS norm for each leaf tensor of `x` and optionally adds summaries.
- `_init(t)` — [`L939`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L939)
- `_init(param: OptParam)` — [`L1879`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1879)
- `_interpolate(param, ema_value)` — [`L1632`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1632)
- `_is_valid_step(g_norm: Tensor, drop_norm: Union[float, DropNormThresholdFn], *, norm_ema: Optional[Tensor], norm_square_ema: Optional[Tensor], count: Optional[Tensor], drop_stats: Optional[dict[str, Tensor]])` — [`L1360`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1360)
- `_log_per_layer_stats(stats: NestedTensor, *, summary_suffix: str)` — [`L398`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L398) — Expand the Nested Tensor `stats` and add summaries.
- `_moment(val: Tensor, norm_ema: Tensor, norm_square_ema: Tensor, count: Tensor)` — [`L1346`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1346)
- `_moment(x: Tensor, *, acc: Optional[Tensor], decay: Optional[float], debias: bool)` — [`L1896`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1896)
- `_move_fn(state: optax.OptState, dst: MemoryKind)` — [`L2158`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2158)
- `_no_op()` — [`L113`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L113)
- `_partition(param_spec: ParameterSpec)` — [`L2021`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2021)
- `_raw_updates(grad: Tensor, pps: _AdastarPerParamState)` — [`L1922`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1922) — Computes raw updates from gradients.
- `_scale_update(update, param)` — [`L375`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L375)
- `_should_quantize(t_shape: Sequence[int])` — [`L922`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L922)
- `_smoothed_updates(raw_updates: Tensor, pps: _AdastarPerParamState)` — [`L1947`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1947) — Computes smoothed updates from raw updates.
- `_split_update_results(update_results: Nested[_AdastarUpdateResult])` — [`L1906`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1906) — Splits a tree of _AdastarUpdateResult to (updates, state).
- `_stddev(mean: Tensor, mean_square: Tensor)` — [`L1343`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1343)
- `_to_float(value: Tensor, qstep_size: Tensor)` — [`L983`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L983)
- `_to_partitioned_transformation(transformation: ConfigOr[PartitionedGradientTransformation])` — [`L71`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L71)
- `_to_qint_tensor_ema(value: Tensor)` — [`L966`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L966)
- `_to_state(count: Tensor, ema_tree: NestedTree)` — [`L931`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L931)
- `_to_tensor_ema(value: Tensor)` — [`L977`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L977)
- `_update(value: Tensor, ema: Tensor, qstep_size: Tensor)` — [`L990`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L990)
- `_update2(u: Tensor, param: OptParam, weight_decay_scale: float = 1)` — [`L2041`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2041)
- `_weight_decay_scales(params: NestedOptParam, *, per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None)` — [`L519`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L519) — Returns a nested tree with float leaf nodes.
- `adafactor_optimizer(learning_rate: schedule.Schedule, *, b1: Optional[schedule.Schedule], b2: schedule.Schedule, multiply_by_parameter_scale: bool, clipping_threshold: Optional[float], dtype_momentum: Any = jnp.float32, weight_decay: Optional[float] = None, weight_decay_scale_by_learning_rate_exponent: Optional[float] = None, weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None, eps: float = 1e-30, factored: bool = True, apply_scale_by_trust_ratio: bool = False)` — [`L1058`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1058) — Adafactor optimizer.
- `adam_optimizer(learning_rate: schedule.Schedule, *, b1: float, b2: float, eps: float, l2_regularizer_weight: float = 0, l2_regularizer_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None, mu_dtype: Optional[jnp.dtype] = None)` — [`L859`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L859) — Adam optimizer with l2 regularization.
- `adam_partition(base: optax.GradientTransformation)` — [`L185`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L185)
- `adamw_decoupled_optimizer(learning_rate: float, *, b1: float, b2: float, eps: float, update_schedule: schedule.Schedule, weight_decay: float = 0, weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None, mu_dtype: Optional[jnp.dtype] = None, adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None)` — [`L796`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L796) — A "decoupled" version of the AdamW optimizer, with optional parameter scaling.
- `adamw_optimizer(learning_rate: schedule.Schedule, *, b1: float, b2: float, eps: float, weight_decay: float = 0, weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None, mu_dtype: Optional[jnp.dtype] = None, adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None)` — [`L741`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L741) — AdamW optimizer with parameter scaling.
- `adastar_optimizer(learning_rate: float, *, gradient_ema_decay: Optional[float], gradient_ema_debias: bool, gradient_square_ema_decay: float, gradient_square_ema_debias: bool, eps: float, eps_square: float, raw_update_clipping_threshold: Optional[float], update_ema_decay: Optional[float], update_ema_debias: bool, adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None, weight_decay: float = 0, weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None, update_schedule: schedule.Schedule, verbosity: int = 0)` — [`L1760`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1760) — An optimizer covering both {adamw_decoupled,adafactor}_optimizer (with factored=False).
- `add_decayed_weights(weight_decay: float, *, learning_rate_exponent: Optional[float] = None, learning_rate: Optional[schedule.Schedule] = None, per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None)` — [`L552`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L552) — Add parameter scaled by `weight_decay * (learning_rate ** learning_rate_exponent)`.
- `chain(*args)` — [`L81`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L81)
- `clip_by_block_rms(threshold: Optional[float], *, summary_suffix: Optional[str] = None)` — [`L1509`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1509) — Clip updates to a max rms for the gradient of each param vector or matrix.
- `clip_by_global_norm(max_norm: Optional[float] = None, *, drop_norm: Optional[float] = None, eps: float = 1e-8)` — [`L1149`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1149) — Scales gradients s.t. global norm <= max_norm, and drop gradients that exceed drop_norm.
- `copy_partition(specs: Nested[OptStateSpec], *, pattern: Union[None, str, re.Pattern] = None, memory_kind: Optional[MemoryKind] = None)` — [`L145`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L145) — Copies OptStateSpec and optionally assigns with a different memory kind.
- `drop_norm_by_grad_norm_ema(multipliers: tuple = (20, 40, 100))` — [`L1205`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1205) — Return drop norm thresholds which are multiples of grad norm ema.
- `drop_norm_by_grad_norm_stddev(*, min_count: int = 500, multipliers: tuple = (20, 40, 100))` — [`L1220`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1220) — Return drop norm thresholds based on grad norm stddev.
- `ema(decay: schedule.Schedule, debias: bool = True, accumulator_dtype: Optional[jnp.dtype] = jnp.float32)` — [`L888`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L888) — Compute an exponential moving average of updates.
- `f(g, p, s)` — [`L609`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L609)
- `fn(params: NestedOptParam)` — [`L304`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L304)
- `fn(params: NestedOptParam)` — [`L342`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L342)
- `fn(count: Tensor, mean: Tensor, stddev: Tensor)` — [`L1208`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1208)
- `fn(count: Tensor, mean: Tensor, stddev: Tensor)` — [`L1227`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1227)
- `get_ema_partition(param_spec: ParameterSpec)` — [`L1014`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1014)
- `get_scale_partition(param_spec: ParameterSpec)` — [`L1025`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1025)
- `init_fn(params)` — [`L96`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L96)
- `init_fn(params: NestedOptParam)` — [`L134`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L134)
- `init_fn(params)` — [`L247`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L247)
- `init_fn(params)` — [`L367`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L367)
- `init_fn(params)` — [`L484`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L484)
- `init_fn(params)` — [`L585`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L585)
- `init_fn(params)` — [`L660`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L660)
- `init_fn(params)` — [`L938`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L938)
- `init_fn(params)` — [`L1159`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1159)
- `init_fn(params)` — [`L1311`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1311)
- `init_fn(params)` — [`L1528`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1528)
- `init_fn(params)` — [`L1570`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1570)
- `init_fn(params)` — [`L1619`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1619)
- `init_fn(params)` — [`L1679`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1679)
- `init_fn(params: NestedOptParam)` — [`L1876`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1876) — Initializes the stage 1 state.
- `init_fn(params: NestedOptParam)` — [`L2155`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2155)
- `l2_regularizer(regularizer_weight: Optional[float] = 0, per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None)` — [`L636`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L636) — Adds l2_regularization scaled by `regularizer_weight` to gradients.
- `lion_optimizer(learning_rate: schedule.Schedule, b1: float, b2: float, mu_dtype: Optional[jnp.dtype] = None, weight_decay: float = 0, multiply_by_parameter_scale: bool = False, weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None)` — [`L1709`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1709) — Lion optimizer with parameter scaling.
- `maybe_override_scale(path: str, param: OptParam, curr_scale: float)` — [`L531`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L531)
- `named_chain(**kwargs)` — [`L93`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L93)
- `offload_optimizer(optimizer: ConfigOr[PartitionedGradientTransformation], *, pattern: Union[str, re.Pattern] = ".*", offload_src: MemoryKind = "device", offload_dst: MemoryKind = "pinned_host")` — [`L2095`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2095) — Offload the state of the wrapped optimizer that matches `pattern` to `offload_dst`.
- `opt_param_values(params: NestedOptParam)` — [`L127`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L127)
- `param_ema(*, decay: Optional[schedule.Schedule] = None)` — [`L1592`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1592) — Computes the EMA of model params.
- `partition(param_spec)` — [`L85`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L85)
- `partition_fn(param_spec)` — [`L107`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L107)
- `partition_fn(param_specs: Nested[ParameterSpec])` — [`L179`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L179)
- `partition_fn(param_specs: Nested[ParameterSpec])` — [`L188`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L188)
- `partition_fn(param_specs)` — [`L621`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L621)
- `partition_fn(param_specs: Nested[ParameterSpec])` — [`L1013`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1013)
- `partition_fn(param_specs: Nested[ParameterSpec])` — [`L1476`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1476)
- `partition_fn(param_specs: Nested[ParameterSpec])` — [`L1644`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1644)
- `partition_fn(param_specs: Nested[ParameterSpec])` — [`L1692`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1692)
- `partition_fn(param_specs)` — [`L2020`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2020)
- `partition_fn(param_spec: Nested[ParameterSpec])` — [`L2187`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2187)
- `per_param_fn(param_path: str)` — [`L305`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L305)
- `per_param_scale_by_path(*, scale_by_path: Sequence[tuple[str, float]], description: str, default_scale: float = 1)` — [`L287`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L287) — Computes per-parameter scales with regex-based rules.
- `per_param_scale_by_rms(*, min_scale: float = 0.0001)` — [`L330`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L330) — Computes per-parameter scales with its Root-Mean-Square (RMS).
- `replace_with_updates()` — [`L202`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L202) — Replaces params with updates (gradients).
- `scale(step_size: float)` — [`L222`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L222)
- `scale_by_lion(b1: float = 0.9, b2: float = 0.99, mu_dtype: Optional[Any] = None)` — [`L1662`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1662) — Rescale updates according to the Lion algorithm.
- `scale_by_param_block_rms(min_scale: float = 0.001)` — [`L1555`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1555) — Scale updates by rms of the gradient for each param vector or matrix.
- `scale_by_schedule(step_size_fn: schedule.Schedule, *, name: Optional[str] = None)` — [`L226`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L226) — Scales updates using a custom schedule for the step size.
- `scale_by_trust_ratio(min_norm: float = 0, trust_coefficient: float = 1, eps: float = 0)` — [`L348`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L348) — Scale updates by trust ratio`.
- `scale_fn(step)` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L276)
- `scale_from_learning_rate(learning_rate: schedule.Schedule, *, flip_sign=True)` — [`L271`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L271)
- `scale_update_per_param(per_param_scale: Callable[[NestedOptParam], Any])` — [`L467`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L467) — Scales updates based on `per_param_scale`.
- `sgd_optimizer(learning_rate: schedule.Schedule, *, decouple_weight_decay: bool, momentum: float = 0, weight_decay: float = 0, weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None)` — [`L685`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L685) — SGD optimizer implementation.
- `skip_and_clip_by_global_norm(inner: ConfigOr[PartitionedGradientTransformation], *, drop_norm: Optional[Union[float, ConfigOr[DropNormThresholdFn]]] = None, max_norm: Optional[float] = None, grad_norm_ema_decay: Optional[float] = None, eps: float = 1e-8)` — [`L1254`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1254) — Skip updates when global norm >= drop_norm, otherwise clip the global norm.
- `trace_partition(base: optax.GradientTransformation)` — [`L176`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L176)
- `update2_fn(updates, state: Tensor, params: NestedOptParam)` — [`L2038`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2038)
- `update_fn(updates: NestedTensor, state: dict[str, Any], params: NestedOptParam)` — [`L99`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L99)
- `update_fn(updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam)` — [`L114`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L114)
- `update_fn(updates: optax.Updates, state: optax.OptState, params: NestedOptParam)` — [`L137`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L137)
- `update_fn(updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam)` — [`L205`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L205)
- `update_fn(updates, state, params=None)` — [`L251`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L251)
- `update_fn(updates, state, params)` — [`L371`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L371)
- `update_fn(updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam)` — [`L489`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L489)
- `update_fn(updates: NestedTensor, state: AddDecayedWeightsState, params: NestedOptParam)` — [`L593`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L593)
- `update_fn(updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam)` — [`L664`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L664)
- `update_fn(updates, state, params=None)` — [`L961`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L961)
- `update_fn(updates, state, params=None)` — [`L1163`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1163)
- `update_fn(updates, state, params=None)` — [`L1337`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1337)
- `update_fn(updates, state, params=None)` — [`L1532`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1532)
- `update_fn(updates, state, params)` — [`L1574`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1574)
- `update_fn(updates, state, params)` — [`L1625`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1625)
- `update_fn(updates, state, params=None)` — [`L1683`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1683)
- `update_fn(grads: NestedTensor, state: _AdastarState, params: NestedOptParam)` — [`L1889`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L1889) — Applies (stage 1) gradient transformation to compute raw_updates.
- `update_fn(updates: optax.Updates, state: optax.OptState, params: NestedOptParam)` — [`L2181`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L2181)
- `with_partition_fn(base: optax.GradientTransformation, partition_fn: TransformPartitionSpecFn)` — [`L131`](../../../../../../raw/code/axlearn/axlearn/common/optimizers.py#L131)

