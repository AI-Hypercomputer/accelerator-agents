---
title: 'Module: ejkernel/ops/execution/executor.py'
type: catalog
provenance: extracted
module: ejkernel/ops/execution/executor.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.execution.executor`/
symbols:
  Executor: Executor#
  Executor.__call__: Executor#__call__().
  Executor.choose_config: Executor#choose_config().
  Executor._choose_heuristics_only: Executor#_choose_heuristics_only().
  Executor._prefer_cuda_cfg: Executor#_prefer_cuda_cfg().
  Executor._has_cuda_impl: Executor#_has_cuda_impl().
  Executor._has_cute_impl: Executor#_has_cute_impl().
  Executor.chooser: Executor#chooser.
  Executor.compile: Executor#compile().
  ConfigChooser.choose: ConfigChooser#choose().
  Executor._stamp_json: Executor#_stamp_json().
  Executor._stamp_hash: Executor#_stamp_hash().
  Executor.bwd_arrays: Executor#bwd_arrays().
  Executor.fn: Executor#fn().
  Executor.wrapped: Executor#wrapped().
  Executor.fwd_arrays: Executor#fwd_arrays().
  Executor._stamp: Executor#_stamp().
  Executor.primal_only_arrays: Executor#primal_only_arrays().
  Executor.run: Executor#run().
  ConfigChooser: ConfigChooser#
  Executor.__init__: Executor#__init__().
  Executor.stamp_prefix: Executor#stamp_prefix.
  Executor._platform_value: Executor#_platform_value().
  Executor._restore_args_kwargs: Executor#_restore_args_kwargs().
  Executor._is_nvidia_gpu: Executor#_is_nvidia_gpu().
  Executor.align_arg_grad: Executor#align_arg_grad().
---
# Module: [`ejkernel/ops/execution/executor.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py)

## Classes
### `ConfigChooser`  ·  implements/extends Protocol
- def: [`ejkernel/ops/execution/executor.py:83`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L83)
- doc: Protocol for configuration selection strategies.
- signature: `class ConfigChooser(Protocol):`
- members:
  - `choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out])` — [`L94`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L94) — Select optimal configuration for the given invocation.
- uses (calls/refs, reference-scoped): [`Invocation`](../core/kernel.md#Invocation), [`Kernel`](../core/kernel.md#Kernel), [`Cfg`](../core/types.md#Cfg), [`Out`](../core/types.md#Out)
- used by: [`__call__`](executor.md#Executor.__call__), [`choose_config`](executor.md#Executor.choose_config), [`__init__`](executor.md#Executor.__init__)

### `Executor`  ·  implements/extends Generic
- def: [`ejkernel/ops/execution/executor.py:107`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L107) — documented in [ejkernel-modules-base](../../../../concepts/ejkernel-modules-base.md)
- doc: Main execution engine for kernels with automatic configuration selection.
- signature: `class Executor(Generic[Cfg, Out]):`
- members:
  - `__call__(self, kernel: Kernel[Cfg, Out], *args, cfg: Cfg | None = None, stamp: bool = True, method: Literal["shard_map"] | None = None, mesh: jax.sharding.Mesh | None = None, in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None, out_specs: jax.sharding.PartitionSpec | None = None, check_vma: bool = False, **kwargs)` — [`L390`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L390) — Execute kernel with automatic configuration selection and management. — documented in [ejkernel-ops-execution-executor](../../../../concepts/ejkernel-ops-execution-executor.md)
  - `__init__(self, chooser: ConfigChooser, stamp_prefix: str = "ejkernel_ops")` — [`L124`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L124) — Initialize executor with configuration chooser and profiling settings.
  - `_choose_heuristics_only(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out])` — [`L662`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L662) — Select configuration using fast heuristics path without autotuning. — documented in [ejkernel-ops-execution-executor](../../../../concepts/ejkernel-ops-execution-executor.md)
  - `_has_cuda_impl(algorithm: str)` — [`L202`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L202) — Check if a native CUDA implementation exists for the given algorithm.
  - `_has_cute_impl(algorithm: str)` — [`L222`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L222) — Check if a CuTe DSL implementation exists for the given algorithm.
  - `_is_nvidia_gpu()` — [`L155`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L155) — Detect whether the current GPU device is an NVIDIA GPU.
  - `_platform_value(val)` — [`L135`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L135) — Convert a platform value to a lowercase string representation.
  - `_prefer_cuda_cfg(self, cfg: Cfg, kernel: Kernel[Cfg, Out], inv: Invocation[Cfg, Out])` — [`L241`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L241) — Upgrade configuration to prefer CUTE/CUDA when conditions are met. — documented in [ejkernel-ops-execution-executor](../../../../concepts/ejkernel-ops-execution-executor.md)
  - `_restore_args_kwargs(array_leaves)` — [`L500`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L500) — Rebuild (args, kwargs) by merging dynamic array leaves into closed constants.
  - `_stamp(self, name: str, fn: Callable)` — [`L367`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L367) — Add profiling metadata to function using JAX naming primitives.
  - `_stamp_hash(self, kernel, inv, fn, cfg)` — [`L310`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L310) — Add hash-based profiling metadata to function.
  - `_stamp_json(self, kernel, inv, fn, cfg)` — [`L330`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L330) — Add JSON-based profiling metadata to function.
  - `align_arg_grad(x, g)` — [`L526`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L526) — Align gradient structure with argument structure, using None for missing grads.
  - `bwd_arrays(payload, dy)` — [`L512`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L512) — Backward rule: rebuild args/kwargs, call kernel.vjp, and map grads to array inputs.
  - `choose_config(self, kernel: Kernel[Cfg, Out], *args, cfg: Cfg | None = None, **kwargs)` — [`L616`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L616) — Select configuration for kernel without executing it. — documented in [ejkernel-ops-execution-executor](../../../../concepts/ejkernel-ops-execution-executor.md)
  - `compile(self, kernel: Kernel[Cfg, Out], *example_args, cfg: Cfg | None = None, **example_kwargs)` — [`L720`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L720) — Compile kernel with pre-selected configuration.
  - `fn(*a, **k)` — [`L559`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L559) — Extract array leaves from args/kwargs and route through the custom VJP wrapper.
  - `fwd_arrays(*array_leaves)` — [`L506`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L506) — Forward rule: takes only array leaves, rebuilds args/kwargs inside.
  - `primal_only_arrays(*array_inputs)` — [`L552`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L552) — Compute forward pass output only, discarding residuals for custom VJP.
  - `run(*a, **k)` — [`L747`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L747) — Execute the kernel with the pre-selected optimal configuration.
  - `wrapped(*a, **k)` — [`L360`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L360) — Execute the function within a JAX named scope containing the JSON payload.
  - `chooser` — [`L131`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L131)
  - `stamp_prefix` — [`L132`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/executor.py#L132)
- uses (calls/refs, reference-scoped): [`Invocation`](../core/kernel.md#Invocation), [`Kernel`](../core/kernel.md#Kernel), [`kwargs`](../core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`heuristic_cfg`](../core/kernel.md#Kernel.heuristic_cfg), [`run`](../core/kernel.md#Kernel.run), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`create_shard_map_wrapper`](../core/kernel.md#Kernel.create_shard_map_wrapper), [`Cfg`](../core/types.md#Cfg), [`Out`](../core/types.md#Out), [`args`](../core/kernel.md#Invocation.args), [`backend`](../../kernels/_registry.md#KernelSpec.backend), [`op_id`](../core/kernel.md#Kernel.op_id), [`platform`](../../kernels/_registry.md#KernelSpec.platform), [`make_key`](../core/kernel.md#Invocation.make_key), [`_get_platform_method`](../core/kernel.md#_get_platform_method), [`override_cfg`](../core/kernel.md#Invocation.override_cfg), [`abstractify`](../utils/fingerprint.md#abstractify), [`device_fingerprint`](../utils/fingerprint.md#device_fingerprint), [`choose`](executor.md#ConfigChooser.choose), [`get_device_platform`](../utils/fingerprint.md#get_device_platform), [`_has_custom_vjp`](../core/kernel.md#_has_custom_vjp), [`key_builder`](../core/kernel.md#Kernel.key_builder), [`method`](../core/kernel.md#Invocation.method), [`CUDA`](../../kernels/_registry.md#Platform.CUDA), [`CUTE`](../../kernels/_registry.md#Platform.CUTE), [`fwd_with_residuals`](../core/kernel.md#Kernel.fwd_with_residuals), [`list_implementations`](../../kernels/_registry.md#KernelRegistry.list_implementations), [`vjp`](../core/kernel.md#Kernel.vjp), [`stable_json`](../utils/fingerprint.md#stable_json), [`GPU`](../../kernels/_registry.md#Backend.GPU), [`_cache_overlay`](../config/cache.md#_cache_overlay._cache_overlay), [`record_invocation`](../registry.md#record_invocation), [`ConfigChooser`](executor.md#ConfigChooser), [`check_vma`](../core/kernel.md#Invocation.check_vma), [`in_specs`](../core/kernel.md#Invocation.in_specs), [`mesh`](../core/kernel.md#Invocation.mesh), [`out_specs`](../core/kernel.md#Invocation.out_specs)  (+3 more)
- used by: [`_quantized_matmul_impl`](../../modules/operations/quantized_matmul.md#_quantized_matmul_impl), [`_quantized_matmul_executor`](../../modules/operations/quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule_v2.md#_executor._executor), [`_executor`](../../modules/operations/ragged_causal_conv1d.md#_executor._executor), [`_all_gather_matmul_executor`](../../modules/operations/all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`_chunked_prefill_paged_decode_executor`](../../modules/operations/chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`_decode_attention_executor`](../../modules/operations/decode_attention.md#_decode_attention_executor._decode_attention_executor), [`_dsa_executor`](../../modules/operations/deepseek_attn.md#_dsa_executor._dsa_executor), [`_executor`](../../modules/operations/attention.md#_executor._executor), [`_executor`](../../modules/operations/blocksparse_attention.md#_executor._executor), [`_executor`](../../modules/operations/fused_conv_decode.md#_executor._executor), [`_executor`](../../modules/operations/fused_cross_entropy.md#_executor._executor), [`_executor`](../../modules/operations/fused_kl_divergence.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule_grouped_decode.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/gdn_compute_schedule_v2.md#_executor._executor), [`_executor`](../../modules/operations/kernel_delta_attention.md#_executor._executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/rwkv4.md#_executor._executor), [`_executor`](../../modules/operations/rwkv6.md#_executor._executor), [`_executor`](../../modules/operations/scaled_dot_product_attention.md#_executor._executor), [`_executor_rwkv7`](../../modules/operations/rwkv7.md#_executor_rwkv7._executor_rwkv7), [`_executor_rwkv7_mul`](../../modules/operations/rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`_flash_executor`](../../modules/operations/flash_attention.md#_flash_executor._flash_executor), [`_gla_executor`](../../modules/operations/gated_linear_attention.md#_gla_executor._gla_executor), [`_grouped_matmul_executor`](../../modules/operations/grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`_lightning_executor`](../../modules/operations/lightning_attention.md#_lightning_executor._lightning_executor), [`_mean_pooling_executor`](../../modules/operations/pooling.md#_mean_pooling_executor._mean_pooling_executor), [`_mla_executor`](../../modules/operations/multi_head_latent_attention.md#_mla_executor._mla_executor), [`_mlrpa_executor`](../../modules/operations/multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`_mlrpa_v2_executor`](../../modules/operations/multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`_page_attention_executor`](../../modules/operations/page_attention.md#_page_attention_executor._page_attention_executor), [`_prefill_page_attention_executor`](../../modules/operations/prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`_ragged_decode_attention_executor`](../../modules/operations/ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_v2_tq_executor`](../../modules/operations/ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`_ragged_page_attention_v3_tq_executor`](../../modules/operations/ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`_recurrent_executor`](../../modules/operations/recurrent.md#_recurrent_executor._recurrent_executor), [`_reduce_scatter_matmul_executor`](../../modules/operations/reduce_scatter_matmul.md#_reduce_scatter_matmul_executor._reduce_scatter_matmul_executor)  (+6 more)

