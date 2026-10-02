---
title: 'Module: ejkernel/kernels/_registry.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_registry.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._registry`/
symbols:
  Backend: Backend#
  kernel_registry: kernel_registry.
  Platform: Platform#
  KernelRegistry.get: KernelRegistry#get().
  Backend.ANY: Backend#ANY.
  KernelRegistry.register: KernelRegistry#register().
  Platform.PALLAS: Platform#PALLAS.
  KernelRegistry.decorator: KernelRegistry#decorator().
  Backend.TPU: Backend#TPU.
  KernelSpec.backend: KernelSpec#backend.
  KernelSpec.platform: KernelSpec#platform.
  KernelRegistry.validate_signatures: KernelRegistry#validate_signatures().
  Platform.TRITON: Platform#TRITON.
  KernelRegistry: KernelRegistry#
  Platform.XLA: Platform#XLA.
  Platform.TILELANG: Platform#TILELANG.
  KernelRegistry._registry: KernelRegistry#_registry.
  _collect_unsupported_reasons: _collect_unsupported_reasons().
  Platform.CUDA: Platform#CUDA.
  Platform.CUTE: Platform#CUTE.
  KernelRegistry._wrapped: KernelRegistry#_wrapped().
  KernelRegistry.list_implementations: KernelRegistry#list_implementations().
  Backend.GPU: Backend#GPU.
  F: F.
  _get_ignored_params: _get_ignored_params().
  _IGNORED_PARAM_CACHE._IGNORED_PARAM_CACHE: _IGNORED_PARAM_CACHE._IGNORED_PARAM_CACHE.
  KernelSpec.implementation: KernelSpec#implementation.
  _types_are_equivalent: _types_are_equivalent().
  KernelRegistry.list_algorithms: KernelRegistry#list_algorithms().
  KernelSpec: KernelSpec#
  KernelSpec.priority: KernelSpec#priority.
  _TUNING_PARAM_NAMES._TUNING_PARAM_NAMES: _TUNING_PARAM_NAMES._TUNING_PARAM_NAMES.
  KernelRegistry.__init__: KernelRegistry#__init__().
  _normalize_type_string: _normalize_type_string().
  _get_ignored_params._DelVisitor: _get_ignored_params()._DelVisitor#
  _is_non_default: _is_non_default().
  KernelSpec.algorithm: KernelSpec#algorithm.
  _get_ignored_params._DelVisitor.visit_Delete: _get_ignored_params()._DelVisitor#visit_Delete().
  Backend.MPS: Backend#MPS.
  Backend.CPU: Backend#CPU.
---
# Module: [`ejkernel/kernels/_registry.py`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py)

## Classes
### `Backend`  ·  implements/extends StrEnum
- def: [`ejkernel/kernels/_registry.py:331`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L331) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
- doc: Target hardware backends for kernel execution.
- signature: `class Backend(StrEnum):`
- members:
  - `ANY` — [`L350`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L350) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
  - `CPU` — [`L349`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L349)
  - `GPU` — [`L346`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L346)
  - `MPS` — [`L347`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L347)
  - `TPU` — [`L348`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L348)
- used by: [`detect_platform`](../modules/base.md#detect_platform), [`get`](_registry.md#KernelRegistry.get), [`run`](../modules/operations/scaled_dot_product_attention.md#ScaledDotProductAttention.run), [`quantized_matmul`](_pallas/tpu/quantized_matmul/_interface.md#quantized_matmul), [`_fused_linear_cross_entropy_dispatch`](../modules/operations/fused_cross_entropy.md#_fused_linear_cross_entropy_dispatch), [`run`](../modules/operations/deepseek_attn.md#DeepSeekAttention.run), [`run`](../modules/operations/native_sparse_attention.md#NativeSparseAttention.run), [`run`](../modules/operations/pooling.md#MeanPooling.run), [`run`](../modules/operations/fused_cross_entropy.md#FusedCrossEntropy.run), [`run`](../modules/operations/fused_kl_divergence.md#FusedKLDivergence.run), [`run`](../modules/operations/gated_delta_rule.md#GatedDeltaRule.run), [`register`](_registry.md#KernelRegistry.register), [`run`](../modules/operations/gated_linear_attention.md#GLAttention.run), [`run`](../modules/operations/lightning_attention.md#LightningAttention.run), [`run`](../modules/operations/recurrent.md#RecurrentAttention.run), [`run`](../modules/operations/grouped_matmul.md#GroupedMatmul.run), [`run`](../modules/operations/page_attention.md#PageAttention.run), [`ring_attention`](_xla/ring_attention/_interface.md#ring_attention), [`run`](../modules/operations/reduce_scatter_matmul.md#ReduceScatterMatmul.run), [`quantized_matmul`](_xla/quantized_matmul/_interface.md#quantized_matmul), [`run`](../modules/operations/rwkv7.md#RWKV7.run), [`run`](../modules/operations/rwkv7.md#RWKV7Mul.run), [`deepseek_attn`](_pallas/tpu/deepseek_attn/_interface.md#deepseek_attn), [`run`](../modules/operations/multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.run), [`run`](../modules/operations/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.run), [`decorator`](_registry.md#KernelRegistry.decorator), [`flash_mla`](_pallas/tpu/flash_mla/_interface.md#flash_mla), [`run`](../modules/operations/all_gather_matmul.md#AllGatherMatmul.run), [`run`](../modules/operations/flash_attention.md#FlashAttention.run), [`run`](../modules/operations/ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.run), [`run`](../modules/operations/ragged_page_attention_v3.md#RaggedPageAttentionv3.run), [`run`](../modules/operations/unified_attention.md#UnifiedAttention.run), [`run`](../modules/operations/ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.run), [`gated_delta_rule`](_pallas/tpu/gated_delta_rule/_interface.md#gated_delta_rule), [`multi_latent_ragged_page_attention_v2`](_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.md#multi_latent_ragged_page_attention_v2), [`ring_attention`](_pallas/tpu/ring_attention/_interface.md#ring_attention), [`run`](../modules/operations/blocksparse_attention.md#BlockSparseAttention.run), [`run`](../modules/operations/chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.run), [`run`](../modules/operations/ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.run), [`run`](../modules/operations/ragged_page_attention_v2.md#RaggedPageAttentionv2.run)  (+41 more)

### `KernelRegistry`
- def: [`ejkernel/kernels/_registry.py:378`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L378)
- doc: Central registry for managing kernel implementations across platforms and backends.
- signature: `class KernelRegistry:`
- members:
  - `__init__(self)` — [`L406`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L406) — Initialize an empty kernel registry with no registered algorithms.
  - `_wrapped(*args, **kwargs)` — [`L463`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L463) — Validation wrapper that guards the kernel call.
  - `decorator(func: F)` — [`L453`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L453) — Inner decorator that wraps and registers *func*.
  - `get(self, algorithm: str, platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None, backend: Backend | Literal["gpu", "mps", "tpu", "cpu", "any"] | None = None)` — [`L499`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L499) — Retrieve the best matching kernel implementation. — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
  - `list_algorithms(self)` — [`L562`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L562) — List all registered algorithm names.
  - `list_implementations(self, algorithm: str)` — [`L571`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L571) — List all registered implementations for a given algorithm.
  - `register(self, algorithm: str, platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla"], backend: Backend | Literal["gpu", "mps", "tpu", "cpu", "any"], priority: int = 0)` — [`L411`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L411) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
  - `validate_signatures(self, algorithm: str | None, verbose: bool = False)` — [`L585`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L585) — Validate that all implementations of an algorithm have compatible signatures.
- protocol/private: `_registry`[`L408`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L408)
- uses (calls/refs, reference-scoped): [`Backend`](_registry.md#Backend), [`Platform`](_registry.md#Platform), [`ANY`](_registry.md#Backend.ANY), [`backend`](_registry.md#KernelSpec.backend), [`platform`](_registry.md#KernelSpec.platform), [`XLA`](_registry.md#Platform.XLA), [`_collect_unsupported_reasons`](_registry.md#_collect_unsupported_reasons), [`EjkernelRuntimeError`](../errors.md#EjkernelRuntimeError), [`F`](_registry.md#F), [`implementation`](_registry.md#KernelSpec.implementation), [`_types_are_equivalent`](_registry.md#_types_are_equivalent), [`KernelSpec`](_registry.md#KernelSpec), [`_TUNING_PARAM_NAMES`](_registry.md#_TUNING_PARAM_NAMES._TUNING_PARAM_NAMES), [`priority`](_registry.md#KernelSpec.priority), [`algorithm`](_registry.md#KernelSpec.algorithm)
- used by: [`kernel_registry`](_registry.md#kernel_registry), [`detect_platform`](../modules/base.md#detect_platform), [`quantized_matmul`](_pallas/tpu/quantized_matmul/_interface.md#quantized_matmul), [`run`](../modules/operations/attention.md#Attention.run), [`_fused_linear_cross_entropy_dispatch`](../modules/operations/fused_cross_entropy.md#_fused_linear_cross_entropy_dispatch), [`run`](../modules/operations/native_sparse_attention.md#NativeSparseAttention.run), [`run`](../modules/operations/pooling.md#MeanPooling.run), [`run`](../modules/operations/fused_cross_entropy.md#FusedCrossEntropy.run), [`run`](../modules/operations/fused_kl_divergence.md#FusedKLDivergence.run), [`run`](../modules/operations/gated_delta_rule.md#GatedDeltaRule.run), [`run`](../modules/operations/gated_linear_attention.md#GLAttention.run), [`run`](../modules/operations/lightning_attention.md#LightningAttention.run), [`run`](../modules/operations/recurrent.md#RecurrentAttention.run), [`run`](../modules/operations/page_attention.md#PageAttention.run), [`ring_attention`](_xla/ring_attention/_interface.md#ring_attention), [`quantized_matmul`](_xla/quantized_matmul/_interface.md#quantized_matmul), [`run`](../modules/operations/rwkv7.md#RWKV7.run), [`run`](../modules/operations/rwkv7.md#RWKV7Mul.run), [`deepseek_attn`](_pallas/tpu/deepseek_attn/_interface.md#deepseek_attn), [`flash_mla`](_pallas/tpu/flash_mla/_interface.md#flash_mla), [`gated_delta_rule`](_pallas/tpu/gated_delta_rule/_interface.md#gated_delta_rule), [`multi_latent_ragged_page_attention_v2`](_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.md#multi_latent_ragged_page_attention_v2), [`ring_attention`](_pallas/tpu/ring_attention/_interface.md#ring_attention), [`attention`](_xla/attention/_interface.md#attention), [`blocksparse_attention`](_pallas/tpu/blocksparse_attention/_interface.md#blocksparse_attention), [`_has_cuda_impl`](../ops/execution/executor.md#Executor._has_cuda_impl), [`_has_cute_impl`](../ops/execution/executor.md#Executor._has_cute_impl), [`_HBM_ANY`](_pallas/tpu/page_attention/_interface.md#_HBM_ANY), [`_HBM_ANY`](_pallas/tpu/prefill_page_attention/_interface.md#_HBM_ANY), [`ragged_gated_delta_rule`](_pallas/tpu/ragged_gated_delta_rule/_interface.md#ragged_gated_delta_rule), [`grouped_matmul`](_pallas/tpu/grouped_matmul/_interface.md#grouped_matmul), [`grouped_matmulv2`](_pallas/tpu/grouped_matmulv2/_interface.md#grouped_matmulv2), [`grouped_matmulv3`](_pallas/tpu/grouped_matmulv3/_interface.md#grouped_matmulv3), [`multi_latent_ragged_page_attention`](_pallas/tpu/multi_latent_ragged_page_attention/_interface.md#multi_latent_ragged_page_attention), [`ragged_decode_attention`](_pallas/tpu/ragged_decode_attention/_interface.md#ragged_decode_attention), [`ragged_page_attention_v3`](_pallas/tpu/ragged_page_attention_v3/_interface.md#ragged_page_attention_v3), [`grouped_matmulv3`](_xla/grouped_matmulv3/_interface.md#grouped_matmulv3), [`BlockTables`](_pallas/tpu/flash_attention/_interface.md#BlockTables), [`all_gather_matmul`](_pallas/tpu/all_gather_matmul/_interface.md#all_gather_matmul), [`fused_conv_decode`](_pallas/tpu/fused_conv_decode/_interface.md#fused_conv_decode)  (+50 more)

### `KernelSpec`
- def: [`ejkernel/kernels/_registry.py:354`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L354)
- doc: Immutable specification describing a single registered kernel implementation.
- signature: `class KernelSpec:`
- members:
  - `algorithm` — [`L373`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L373)
  - `backend` — [`L372`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L372) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
  - `implementation` — [`L374`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L374)
  - `platform` — [`L371`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L371) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
  - `priority` — [`L375`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L375)
- uses (calls/refs, reference-scoped): [`Backend`](_registry.md#Backend), [`Platform`](_registry.md#Platform)
- used by: [`detect_platform`](../modules/base.md#detect_platform), [`get`](_registry.md#KernelRegistry.get), [`decorator`](_registry.md#KernelRegistry.decorator), [`_has_cuda_impl`](../ops/execution/executor.md#Executor._has_cuda_impl), [`_has_cute_impl`](../ops/execution/executor.md#Executor._has_cute_impl), [`validate_signatures`](_registry.md#KernelRegistry.validate_signatures), [`list_implementations`](_registry.md#KernelRegistry.list_implementations), [`__init__`](_registry.md#KernelRegistry.__init__)

### `Platform`  ·  implements/extends StrEnum
- def: [`ejkernel/kernels/_registry.py:308`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L308) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)
- doc: Supported kernel implementation platforms.
- signature: `class Platform(StrEnum):`
- members:
  - `CUDA` — [`L325`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L325)
  - `CUTE` — [`L326`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L326)
  - `PALLAS` — [`L324`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L324)
  - `TILELANG` — [`L327`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L327)
  - `TRITON` — [`L323`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L323)
  - `XLA` — [`L328`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L328)
- used by: [`detect_platform`](../modules/base.md#detect_platform), [`get`](_registry.md#KernelRegistry.get), [`_candidate_cfgs_gpu_for_platform`](../modules/operations/quantized_matmul.md#QuantizedMatmul._candidate_cfgs_gpu_for_platform), [`_quantized_matmul_impl`](../modules/operations/quantized_matmul.md#_quantized_matmul_impl), [`run`](../modules/operations/scaled_dot_product_attention.md#ScaledDotProductAttention.run), [`quantized_matmul`](_pallas/tpu/quantized_matmul/_interface.md#quantized_matmul), [`run`](../modules/operations/quantized_matmul.md#QuantizedMatmul.run), [`run`](../modules/operations/attention.md#Attention.run), [`_fused_linear_cross_entropy_dispatch`](../modules/operations/fused_cross_entropy.md#_fused_linear_cross_entropy_dispatch), [`run`](../modules/operations/deepseek_attn.md#DeepSeekAttention.run), [`run`](../modules/operations/native_sparse_attention.md#NativeSparseAttention.run), [`run`](../modules/operations/pooling.md#MeanPooling.run), [`run`](../modules/operations/gated_delta_rule.md#GatedDeltaRule.run), [`register`](_registry.md#KernelRegistry.register), [`run`](../modules/operations/gated_linear_attention.md#GLAttention.run), [`run`](../modules/operations/lightning_attention.md#LightningAttention.run), [`run`](../modules/operations/recurrent.md#RecurrentAttention.run), [`run`](../modules/operations/page_attention.md#PageAttention.run), [`ring_attention`](_xla/ring_attention/_interface.md#ring_attention), [`candidate_cfgs_gpu`](../modules/operations/quantized_matmul.md#QuantizedMatmul.candidate_cfgs_gpu), [`quantized_matmul`](_xla/quantized_matmul/_interface.md#quantized_matmul), [`heuristic_cfg_gpu`](../modules/operations/quantized_matmul.md#QuantizedMatmul.heuristic_cfg_gpu), [`run`](../modules/operations/rwkv7.md#RWKV7.run), [`run`](../modules/operations/rwkv7.md#RWKV7Mul.run), [`deepseek_attn`](_pallas/tpu/deepseek_attn/_interface.md#deepseek_attn), [`decorator`](_registry.md#KernelRegistry.decorator), [`flash_mla`](_pallas/tpu/flash_mla/_interface.md#flash_mla), [`gated_delta_rule`](_pallas/tpu/gated_delta_rule/_interface.md#gated_delta_rule), [`multi_latent_ragged_page_attention_v2`](_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.md#multi_latent_ragged_page_attention_v2), [`ring_attention`](_pallas/tpu/ring_attention/_interface.md#ring_attention), [`attention`](_xla/attention/_interface.md#attention), [`candidate_cfgs_tpu`](../modules/operations/quantized_matmul.md#QuantizedMatmul.candidate_cfgs_tpu), [`blocksparse_attention`](_pallas/tpu/blocksparse_attention/_interface.md#blocksparse_attention), [`_resolve_inv_platform`](../modules/operations/quantized_matmul.md#QuantizedMatmul._resolve_inv_platform), [`_has_cuda_impl`](../ops/execution/executor.md#Executor._has_cuda_impl), [`_has_cute_impl`](../ops/execution/executor.md#Executor._has_cute_impl), [`platform`](_registry.md#KernelSpec.platform), [`_HBM_ANY`](_pallas/tpu/page_attention/_interface.md#_HBM_ANY), [`_HBM_ANY`](_pallas/tpu/prefill_page_attention/_interface.md#_HBM_ANY), [`ragged_gated_delta_rule`](_pallas/tpu/ragged_gated_delta_rule/_interface.md#ragged_gated_delta_rule)  (+19 more)

### `_DelVisitor`  ·  implements/extends NodeVisitor
- def: [`ejkernel/kernels/_registry.py:150`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L150)
- doc: AST visitor that collects names from `del` statements.
- signature: `class _DelVisitor(ast.NodeVisitor):`
- members:
  - `visit_Delete(self, node: ast.Delete)` — [`L153`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L153) — Record every ``ast.Name`` target in a ``del`` statement.
- used by: [`_get_ignored_params`](_registry.md#_get_ignored_params)

## Functions
- `_collect_unsupported_reasons(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any])` — [`L190`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L190) — Collect human-readable reasons why a kernel call is unsupported.
- `_get_ignored_params(func: Callable[..., Any])` — [`L115`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L115) — Detect parameter names explicitly deleted inside a function body.
- `_is_non_default(value: Any, default: Any)` — [`L165`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L165) — Determine whether a supplied argument value differs from its default.
- `_normalize_type_string(type_annotation: Any)` — [`L244`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L244) — Normalize type annotation string for comparison.
- `_types_are_equivalent(type1: Any, type2: Any)` — [`L282`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L282) — Check if two type annotations are equivalent.

## Module values
- `F` — [`L78`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L78)
- `_IGNORED_PARAM_CACHE` — [`L81`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L81)
- `_TUNING_PARAM_NAMES` — [`L84`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L84)
- `kernel_registry` — [`L714`](../../../../../../raw/code/ejkernel/ejkernel/kernels/_registry.py#L714) — documented in [ejkernel-kernels-_registry](../../../concepts/ejkernel-kernels-_registry.md)

