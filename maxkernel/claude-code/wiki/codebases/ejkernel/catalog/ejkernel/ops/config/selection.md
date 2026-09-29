---
title: 'Module: ejkernel/ops/config/selection.py'
type: catalog
provenance: extracted
module: ejkernel/ops/config/selection.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.config.selection`/
symbols:
  ConfigSelectorChain: ConfigSelectorChain#
  Tuner: Tuner#
  ConfigSelectorChain.choose: ConfigSelectorChain#choose().
  AutotunePolicy: AutotunePolicy#
  AutotunePolicy.allow_autotune: AutotunePolicy#allow_autotune.
  AutotunePolicy.cache_miss_fallback: AutotunePolicy#cache_miss_fallback.
  AutotunePolicy.validate_backward: AutotunePolicy#validate_backward.
  Tuner.measure: Tuner#measure().
  ConfigSelectorChain.f: ConfigSelectorChain#f().
  Tuner._time_forward: Tuner#_time_forward().
  ConfigSelectorChain.__init__: ConfigSelectorChain#__init__().
  Tuner.autotune: Tuner#autotune().
  ConfigSelectorChain._emit: ConfigSelectorChain#_emit().
  Cfg: Cfg.
  policy_override.__enter__: policy_override#__enter__().
  ConfigSelectorChain.persistent: ConfigSelectorChain#persistent.
  ConfigSelectorChain.policy: ConfigSelectorChain#policy.
  Tuner.loss: Tuner#loss().
  policy_override.__exit__: policy_override#__exit__().
  autotune_logger: autotune_logger.
  _is_autotune_progress_enabled: _is_autotune_progress_enabled().
  Tuner.iters: Tuner#iters.
  Tuner._block_all: Tuner#_block_all().
  ConfigSelectorChain.cache: ConfigSelectorChain#cache.
  Tuner.core: Tuner#core().
  forward_autotune_only.__enter__: forward_autotune_only#__enter__().
  forward_autotune_only.__exit__: forward_autotune_only#__exit__().
  log_autotune_progress.__enter__: log_autotune_progress#__enter__().
  log_autotune_progress.__exit__: log_autotune_progress#__exit__().
  ConfigSelectorChain.tuner: ConfigSelectorChain#tuner.
  ConfigSelectorChain.mk: ConfigSelectorChain#mk().
  policy_override: policy_override#
  _autotune_progress_enabled._autotune_progress_enabled: _autotune_progress_enabled._autotune_progress_enabled.
  _is_backward_autotune_enabled: _is_backward_autotune_enabled().
  set_autotune_progress: set_autotune_progress().
  Out: Out.
  policy_override.selector: policy_override#selector.
  forward_autotune_only._token: forward_autotune_only#_token.
  log_autotune_progress._token: log_autotune_progress#_token.
  Tuner.warmup: Tuner#warmup.
  ConfigSelectorChain.persist_autotune: ConfigSelectorChain#persist_autotune.
  _backward_autotune_enabled._backward_autotune_enabled: _backward_autotune_enabled._backward_autotune_enabled.
  policy_override.__init__: policy_override#__init__().
  policy_override._prev: policy_override#_prev.
  forward_autotune_only: forward_autotune_only#
  Tuner._restore_args_kwargs: Tuner#_restore_args_kwargs().
  ConfigSelectorChain.on_event: ConfigSelectorChain#on_event.
  ConfigSelectorChain._autotuned_keys: ConfigSelectorChain#_autotuned_keys.
  AutotunePolicy.allow_heuristics: AutotunePolicy#allow_heuristics.
  policy_override.updates: policy_override#updates.
  log_autotune_progress: log_autotune_progress#
  Tuner._is_arrayish: Tuner#_is_arrayish().
  Tuner._to_concrete: Tuner#_to_concrete().
  Tuner._is_diff: Tuner#_is_diff().
  Tuner._split: Tuner#_split().
  Tuner._merge: Tuner#_merge().
  Tuner._scalarize_output: Tuner#_scalarize_output().
  ConfigSelectorChain.forbid_reautotune: ConfigSelectorChain#forbid_reautotune.
  forward_autotune_only.__init__: forward_autotune_only#__init__().
  log_autotune_progress.__init__: log_autotune_progress#__init__().
  Tuner.__init__: Tuner#__init__().
  ConfigSelectorChain._is_arrayish: ConfigSelectorChain#_is_arrayish().
---
# Module: [`ejkernel/ops/config/selection.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py)

## Classes
### `AutotunePolicy`
- def: [`ejkernel/ops/config/selection.py:90`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L90) — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
- doc: Configuration policy for autotuning behavior.
- signature: `class AutotunePolicy:`
- members:
  - `allow_autotune` — [`L110`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L110) — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
  - `allow_heuristics` — [`L111`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L111)
  - `cache_miss_fallback` — [`L112`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L112) — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
  - `validate_backward` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L113) — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
- used by: [`choose`](selection.md#ConfigSelectorChain.choose), [`_quantized_matmul_executor`](../../modules/operations/quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule_v2.md#_executor._executor), [`_executor`](../../modules/operations/ragged_causal_conv1d.md#_executor._executor), [`_all_gather_matmul_executor`](../../modules/operations/all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`_chunked_prefill_paged_decode_executor`](../../modules/operations/chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`_decode_attention_executor`](../../modules/operations/decode_attention.md#_decode_attention_executor._decode_attention_executor), [`_dsa_executor`](../../modules/operations/deepseek_attn.md#_dsa_executor._dsa_executor), [`_executor`](../../modules/operations/attention.md#_executor._executor), [`_executor`](../../modules/operations/blocksparse_attention.md#_executor._executor), [`_executor`](../../modules/operations/fused_conv_decode.md#_executor._executor), [`_executor`](../../modules/operations/fused_cross_entropy.md#_executor._executor), [`_executor`](../../modules/operations/fused_kl_divergence.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule_grouped_decode.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/gdn_compute_schedule_v2.md#_executor._executor), [`_executor`](../../modules/operations/kernel_delta_attention.md#_executor._executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/rwkv4.md#_executor._executor), [`_executor`](../../modules/operations/rwkv6.md#_executor._executor), [`_executor`](../../modules/operations/scaled_dot_product_attention.md#_executor._executor), [`_executor_rwkv7`](../../modules/operations/rwkv7.md#_executor_rwkv7._executor_rwkv7), [`_executor_rwkv7_mul`](../../modules/operations/rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`_flash_executor`](../../modules/operations/flash_attention.md#_flash_executor._flash_executor), [`_gla_executor`](../../modules/operations/gated_linear_attention.md#_gla_executor._gla_executor), [`_grouped_matmul_executor`](../../modules/operations/grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`_lightning_executor`](../../modules/operations/lightning_attention.md#_lightning_executor._lightning_executor), [`_mean_pooling_executor`](../../modules/operations/pooling.md#_mean_pooling_executor._mean_pooling_executor), [`_mla_executor`](../../modules/operations/multi_head_latent_attention.md#_mla_executor._mla_executor), [`_mlrpa_executor`](../../modules/operations/multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`_mlrpa_v2_executor`](../../modules/operations/multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`_page_attention_executor`](../../modules/operations/page_attention.md#_page_attention_executor._page_attention_executor), [`_prefill_page_attention_executor`](../../modules/operations/prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`_ragged_decode_attention_executor`](../../modules/operations/ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_v2_tq_executor`](../../modules/operations/ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`_ragged_page_attention_v3_tq_executor`](../../modules/operations/ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`_recurrent_executor`](../../modules/operations/recurrent.md#_recurrent_executor._recurrent_executor), [`_reduce_scatter_matmul_executor`](../../modules/operations/reduce_scatter_matmul.md#_reduce_scatter_matmul_executor._reduce_scatter_matmul_executor)  (+8 more)

### `ConfigSelectorChain`
- def: [`ejkernel/ops/config/selection.py:566`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L566) — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
- members:
  - `__init__(self, cache: ConfigCache[Cfg], policy: AutotunePolicy | None = None, tuner: Tuner[Cfg] | None = None, persistent: PersistentCache[Cfg] | None = None, persist_autotune: bool = True, on_event: callable | None = None, forbid_reautotune: bool = True)` — [`L588`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L588) — Initialize configuration selector with cache and policy settings.
  - `_emit(self, event: str, **data)` — [`L813`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L813) — Invoke the optional monitoring callback with a selection event.
  - `_is_arrayish(x)` — [`L709`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L709) — Return True for JAX Arrays, NumPy arrays, and JAX Tracers.
  - `choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out])` — [`L618`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L618) — Select optimal configuration using the fallback hierarchy. — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
  - `f(*a, **k)` — [`L727`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L727) — Build the shard_map wrapper for config ``c`` and run it, applying any callback.
  - `mk(c, _static=static_fun_kwargs)` — [`L724`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L724) — Return a callable that benchmarks the kernel under shard_map with config ``c``.
  - `cache` — [`L609`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L609)
  - `forbid_reautotune` — [`L615`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L615)
  - `on_event` — [`L614`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L614)
  - `persist_autotune` — [`L613`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L613)
  - `persistent` — [`L612`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L612)
  - `policy` — [`L610`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L610)
  - `tuner` — [`L611`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L611)
- protocol/private: `_autotuned_keys`[`L616`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L616)
- uses (calls/refs, reference-scoped): [`Invocation`](../core/kernel.md#Invocation), [`Kernel`](../core/kernel.md#Kernel), [`kwargs`](../core/kernel.md#Invocation.kwargs), [`candidate_cfgs`](../core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../core/kernel.md#Kernel.heuristic_cfg), [`run`](../core/kernel.md#Kernel.run), [`ConfigCache`](cache.md#ConfigCache), [`PersistentCache`](persistent.md#PersistentCache), [`Tuner`](selection.md#Tuner), [`AutotunePolicy`](selection.md#AutotunePolicy), [`create_shard_map_wrapper`](../core/kernel.md#Kernel.create_shard_map_wrapper), [`allow_autotune`](selection.md#AutotunePolicy.allow_autotune), [`cache_miss_fallback`](selection.md#AutotunePolicy.cache_miss_fallback), [`validate_backward`](selection.md#AutotunePolicy.validate_backward), [`args`](../core/kernel.md#Invocation.args), [`op_id`](../core/kernel.md#Kernel.op_id), [`put`](persistent.md#PersistentCache.put), [`make_key`](../core/kernel.md#Invocation.make_key), [`get`](persistent.md#PersistentCache.get), [`_get_platform_method`](../core/kernel.md#_get_platform_method), [`override_cfg`](../core/kernel.md#Invocation.override_cfg), [`device_fingerprint`](../utils/fingerprint.md#device_fingerprint), [`autotune`](selection.md#Tuner.autotune), [`get_device_platform`](../utils/fingerprint.md#get_device_platform), [`key_builder`](../core/kernel.md#Kernel.key_builder), [`Cfg`](selection.md#Cfg), [`method`](../core/kernel.md#Invocation.method), [`put`](cache.md#ConfigCache.put), [`_cache_overlay`](cache.md#_cache_overlay._cache_overlay), [`get`](cache.md#ConfigCache.get), [`_is_backward_autotune_enabled`](selection.md#_is_backward_autotune_enabled), [`Out`](selection.md#Out), [`check_vma`](../core/kernel.md#Invocation.check_vma), [`in_specs`](../core/kernel.md#Invocation.in_specs), [`mesh`](../core/kernel.md#Invocation.mesh), [`out_specs`](../core/kernel.md#Invocation.out_specs), [`allow_heuristics`](selection.md#AutotunePolicy.allow_heuristics)
- used by: [`autotune_lowered`](../execution/offline.md#autotune_lowered), [`_quantized_matmul_executor`](../../modules/operations/quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule_v2.md#_executor._executor), [`_executor`](../../modules/operations/ragged_causal_conv1d.md#_executor._executor), [`_all_gather_matmul_executor`](../../modules/operations/all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`_chunked_prefill_paged_decode_executor`](../../modules/operations/chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`_decode_attention_executor`](../../modules/operations/decode_attention.md#_decode_attention_executor._decode_attention_executor), [`_dsa_executor`](../../modules/operations/deepseek_attn.md#_dsa_executor._dsa_executor), [`_executor`](../../modules/operations/attention.md#_executor._executor), [`_executor`](../../modules/operations/blocksparse_attention.md#_executor._executor), [`_executor`](../../modules/operations/fused_conv_decode.md#_executor._executor), [`_executor`](../../modules/operations/fused_cross_entropy.md#_executor._executor), [`_executor`](../../modules/operations/fused_kl_divergence.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule_grouped_decode.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/gdn_compute_schedule_v2.md#_executor._executor), [`_executor`](../../modules/operations/kernel_delta_attention.md#_executor._executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/rwkv4.md#_executor._executor), [`_executor`](../../modules/operations/rwkv6.md#_executor._executor), [`_executor`](../../modules/operations/scaled_dot_product_attention.md#_executor._executor), [`_executor_rwkv7`](../../modules/operations/rwkv7.md#_executor_rwkv7._executor_rwkv7), [`_executor_rwkv7_mul`](../../modules/operations/rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`_flash_executor`](../../modules/operations/flash_attention.md#_flash_executor._flash_executor), [`_gla_executor`](../../modules/operations/gated_linear_attention.md#_gla_executor._gla_executor), [`_grouped_matmul_executor`](../../modules/operations/grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`_lightning_executor`](../../modules/operations/lightning_attention.md#_lightning_executor._lightning_executor), [`_mean_pooling_executor`](../../modules/operations/pooling.md#_mean_pooling_executor._mean_pooling_executor), [`_mla_executor`](../../modules/operations/multi_head_latent_attention.md#_mla_executor._mla_executor), [`_mlrpa_executor`](../../modules/operations/multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`_mlrpa_v2_executor`](../../modules/operations/multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`_page_attention_executor`](../../modules/operations/page_attention.md#_page_attention_executor._page_attention_executor), [`_prefill_page_attention_executor`](../../modules/operations/prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`_ragged_decode_attention_executor`](../../modules/operations/ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_v2_tq_executor`](../../modules/operations/ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`_ragged_page_attention_v3_tq_executor`](../../modules/operations/ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`_recurrent_executor`](../../modules/operations/recurrent.md#_recurrent_executor._recurrent_executor), [`_reduce_scatter_matmul_executor`](../../modules/operations/reduce_scatter_matmul.md#_reduce_scatter_matmul_executor._reduce_scatter_matmul_executor)  (+9 more)

### `Tuner`  ·  implements/extends Generic
- def: [`ejkernel/ops/config/selection.py:266`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L266) — documented in [ejkernel-ops-config-selection](../../../../concepts/ejkernel-ops-config-selection.md)
- doc: Performance benchmarking engine used by `ConfigSelectorChain`.
- signature: `class Tuner(Generic[Cfg]):`
- members:
  - `__init__(self, warmup=1, iters=3)` — [`L287`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L287) — Initialize tuner with warmup and iteration settings.
  - `_block_all(x)` — [`L347`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L347) — Block until all arrays in the pytree are ready for synchronous timing.
  - `_is_arrayish(x)` — [`L319`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L319) — Return True for JAX Arrays, NumPy arrays, and JAX Tracers (abstract values).
  - `_is_diff(x)` — [`L403`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L403) — Check if a value is differentiable (floating-point or complex type).
  - `_merge(theta, nondiff)` — [`L426`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L426) — Merge differentiable and non-differentiable arrays back into original order.
  - `_restore_args_kwargs(array_leaves)` — [`L357`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L357) — Rebuild (args, kwargs) by merging dynamic array leaves with closed-over constants.
  - `_scalarize_output(output)` — [`L431`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L431) — Reduce array/scalar leaves from arbitrary pytrees into a scalar loss.
  - `_split(arrs)` — [`L419`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L419) — Split array leaves into differentiable and non-differentiable groups.
  - `_time_forward(jitted: bool = True)` — [`L369`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L369) — Time forward-only execution with optional JIT compilation.
  - `_to_concrete(x)` — [`L323`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L323) — Convert a tracer or abstract value to a concrete JAX array.
  - `autotune(self, make_fn, args, kwargs, candidates: Iterable[Cfg])` — [`L490`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L490) — Benchmark all candidate configurations and return the fastest one.
  - `core(*arrs)` — [`L380`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L380) — Reconstruct args/kwargs from array leaves and call the target function.
  - `loss(theta, nondiff)` — [`L458`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L458) — Compute scalar loss for backward pass validation timing.
  - `measure(self, fn, *args, **kwargs)` — [`L296`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L296) — Measure average execution time with optional backward validation.
  - `iters` — [`L294`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L294)
  - `warmup` — [`L294`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L294)
- uses (calls/refs, reference-scoped): [`Cfg`](selection.md#Cfg), [`_is_autotune_progress_enabled`](selection.md#_is_autotune_progress_enabled), [`autotune_logger`](selection.md#autotune_logger)
- used by: [`choose`](selection.md#ConfigSelectorChain.choose), [`autotune_lowered`](../execution/offline.md#autotune_lowered), [`_quantized_matmul_executor`](../../modules/operations/quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule_v2.md#_executor._executor), [`_executor`](../../modules/operations/ragged_causal_conv1d.md#_executor._executor), [`_all_gather_matmul_executor`](../../modules/operations/all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`_chunked_prefill_paged_decode_executor`](../../modules/operations/chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`_decode_attention_executor`](../../modules/operations/decode_attention.md#_decode_attention_executor._decode_attention_executor), [`_dsa_executor`](../../modules/operations/deepseek_attn.md#_dsa_executor._dsa_executor), [`_executor`](../../modules/operations/attention.md#_executor._executor), [`_executor`](../../modules/operations/blocksparse_attention.md#_executor._executor), [`_executor`](../../modules/operations/fused_conv_decode.md#_executor._executor), [`_executor`](../../modules/operations/fused_cross_entropy.md#_executor._executor), [`_executor`](../../modules/operations/fused_kl_divergence.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule_grouped_decode.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/gdn_compute_schedule_v2.md#_executor._executor), [`_executor`](../../modules/operations/kernel_delta_attention.md#_executor._executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/rwkv4.md#_executor._executor), [`_executor`](../../modules/operations/rwkv6.md#_executor._executor), [`_executor`](../../modules/operations/scaled_dot_product_attention.md#_executor._executor), [`_executor_rwkv7`](../../modules/operations/rwkv7.md#_executor_rwkv7._executor_rwkv7), [`_executor_rwkv7_mul`](../../modules/operations/rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`_flash_executor`](../../modules/operations/flash_attention.md#_flash_executor._flash_executor), [`_gla_executor`](../../modules/operations/gated_linear_attention.md#_gla_executor._gla_executor), [`_grouped_matmul_executor`](../../modules/operations/grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`_lightning_executor`](../../modules/operations/lightning_attention.md#_lightning_executor._lightning_executor), [`_mean_pooling_executor`](../../modules/operations/pooling.md#_mean_pooling_executor._mean_pooling_executor), [`_mla_executor`](../../modules/operations/multi_head_latent_attention.md#_mla_executor._mla_executor), [`_mlrpa_executor`](../../modules/operations/multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`_mlrpa_v2_executor`](../../modules/operations/multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`_page_attention_executor`](../../modules/operations/page_attention.md#_page_attention_executor._page_attention_executor), [`_prefill_page_attention_executor`](../../modules/operations/prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`_ragged_decode_attention_executor`](../../modules/operations/ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_v2_tq_executor`](../../modules/operations/ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`_ragged_page_attention_v3_tq_executor`](../../modules/operations/ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`_recurrent_executor`](../../modules/operations/recurrent.md#_recurrent_executor._recurrent_executor)  (+9 more)

### `forward_autotune_only`
- def: [`ejkernel/ops/config/selection.py:172`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L172)
- doc: Context manager that disables backward validation during autotuning.
- signature: `class forward_autotune_only:`
- protocol/private: `__enter__`[`L187`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L187), `__exit__`[`L191`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L191), `__init__`[`L184`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L184), `_token`[`L185`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L185)
- uses (calls/refs, reference-scoped): [`_backward_autotune_enabled`](selection.md#_backward_autotune_enabled._backward_autotune_enabled)

### `log_autotune_progress`
- def: [`ejkernel/ops/config/selection.py:220`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L220)
- doc: Context manager that enables tqdm progress bars during autotuning.
- signature: `class log_autotune_progress:`
- protocol/private: `__enter__`[`L239`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L239), `__exit__`[`L243`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L243), `__init__`[`L236`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L236), `_token`[`L237`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L237)
- uses (calls/refs, reference-scoped): [`_autotune_progress_enabled`](selection.md#_autotune_progress_enabled._autotune_progress_enabled)

### `policy_override`
- def: [`ejkernel/ops/config/selection.py:116`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L116)
- doc: Context manager for temporarily overriding autotuning policy settings.
- signature: `class policy_override:`
- members:
  - `__enter__(self)` — [`L151`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L151) — Enter context and apply policy overrides.
  - `__exit__(self, *exc)` — [`L162`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L162) — Exit context and restore original policy values.
  - `__init__(self, selector: ConfigSelectorChain, **updates)` — [`L140`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L140) — Initialize policy override context manager.
  - `selector` — [`L147`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L147)
  - `updates` — [`L148`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L148)
- protocol/private: `_prev`[`L149`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L149)
- uses (calls/refs, reference-scoped): [`ConfigSelectorChain`](selection.md#ConfigSelectorChain), [`policy`](selection.md#ConfigSelectorChain.policy)
- used by: [`_quantized_matmul_impl`](../../modules/operations/quantized_matmul.md#_quantized_matmul_impl)

## Functions
- `_is_autotune_progress_enabled()` — [`L209`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L209) — Return whether autotune progress bars are enabled.
- `_is_backward_autotune_enabled()` — [`L196`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L196) — Return whether backward-pass validation is currently enabled for autotuning.
- `set_autotune_progress(enabled: bool = True)` — [`L248`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L248) — Imperatively enable or disable autotune progress bars.

## Module values
- `Cfg` — [`L81`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L81)
- `Out` — [`L82`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L82)
- `_autotune_progress_enabled` — [`L86`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L86)
- `_backward_autotune_enabled` — [`L85`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L85)
- `autotune_logger` — [`L84`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/selection.py#L84)

