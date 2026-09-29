---
title: 'Module: easydel/operations/_operation_meta.py'
type: catalog
provenance: extracted
module: easydel/operations/_operation_meta.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations._operation_meta`/
symbols:
  OperationMetadata.get_shardings: OperationMetadata#get_shardings().
  OperationMetadata: OperationMetadata#
  OperationMetadata.mesh: OperationMetadata#mesh().
  AttnShardingRules.query: AttnShardingRules#query.
  OperationMetadata.from_config: OperationMetadata#from_config().
  OperationMetadata.__post_init__: OperationMetadata#__post_init__().
  OperationMetadata.get_operation_config: OperationMetadata#get_operation_config().
  OperationMetadata.runtime_dtype: OperationMetadata#runtime_dtype.
  OperationMetadata.set_attrs_carefully: OperationMetadata#set_attrs_carefully().
  BATCH: BATCH.
  AttnShardingRules.output: AttnShardingRules#output.
  NOT_GIVEN: NOT_GIVEN.
  OperationMetadata.partition_manager: OperationMetadata#partition_manager.
  AttnShardingRules.key: AttnShardingRules#key.
  AttnShardingRules.value: AttnShardingRules#value.
  OperationMetadata.backend: OperationMetadata#backend.
  OperationMetadata.base_config: OperationMetadata#base_config.
  __all__: __all__.
  OperationMetadata.partition_axis: OperationMetadata#partition_axis.
  OperationMetadata._stored_mesh: OperationMetadata#_stored_mesh.
  KV_LENGTH: KV_LENGTH.
  AttnShardingRules.softmax_aux: AttnShardingRules#softmax_aux.
  QUERY_LENGTH: QUERY_LENGTH.
  HEAD: HEAD.
  AttnShardingRules.bias: AttnShardingRules#bias.
  OperationMetadata.platform: OperationMetadata#platform.
  HEAD_DIM: HEAD_DIM.
  OperationMetadata.sequence_axis_name: OperationMetadata#sequence_axis_name.
  OperationMetadata._safety_check: OperationMetadata#_safety_check().
  AttnShardingRules: AttnShardingRules#
  AttnShardingRules.query3d: AttnShardingRules#query3d.
  OperationMetadata.requires_cache: OperationMetadata#requires_cache.
  KV_HEAD: KV_HEAD.
  OperationMetadata.operation_configs: OperationMetadata#operation_configs.
  RUNTIME_MODE_TYPES: RUNTIME_MODE_TYPES.
  KV_HEAD_DIM: KV_HEAD_DIM.
  BIAS_HEAD_SEQ: BIAS_HEAD_SEQ.
  BIAS_KV_SEQ: BIAS_KV_SEQ.
  EMPTY: EMPTY.
  AttnShardingRules.mask: AttnShardingRules#mask.
  AttnShardingRules.q_segment_ids: AttnShardingRules#q_segment_ids.
  AttnShardingRules.kv_segment_ids: AttnShardingRules#kv_segment_ids.
  OperationMetadata.runtime_softmax_dtype: OperationMetadata#runtime_softmax_dtype.
  logger: logger.
---
# Module: [`easydel/operations/_operation_meta.py`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py)

## Classes
### `AttnShardingRules`  ·  implements/extends NamedTuple
- def: [`easydel/operations/_operation_meta.py:58`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L58)
- doc: Named tuple containing JAX PartitionSpecs for all attention tensors.
- signature: `class AttnShardingRules(NamedTuple):`
- members:
  - `bias` — [`L79`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L79)
  - `key` — [`L77`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L77)
  - `kv_segment_ids` — [`L83`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L83)
  - `mask` — [`L80`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L80)
  - `output` — [`L81`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L81)
  - `q_segment_ids` — [`L82`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L82)
  - `query` — [`L76`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L76)
  - `query3d` — [`L75`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L75)
  - `softmax_aux` — [`L84`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L84)
  - `value` — [`L78`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L78)
- used by: [`get_shardings`](_operation_meta.md#OperationMetadata.get_shardings), [`forward_native`](kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`forward_native`](kernels/blocksparse_attention.md#BlockSparseAttn.forward_native), [`forward_native`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](kernels/ring_attention.md#RingAttn.forward_native), [`forward_native`](kernels/flash_attention.md#FlashAttn.forward_native), [`forward_native`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_native), [`forward_native`](kernels/kda.md#KernelDeltaAttnOp.forward_native), [`forward_native`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.forward_native), [`forward_native`](kernels/vanilla_attention.md#VanillaAttn.forward_native), [`forward_ragged`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_ragged), [`grouped_gdr_decode_shard_map_pallas`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.grouped_gdr_decode_shard_map_pallas), [`forward_native`](kernels/inference_gdn.md#RaggedGatedDeltaRule.forward_native)

### `OperationMetadata`
- def: [`easydel/operations/_operation_meta.py:88`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L88)
- doc: Holds configuration, context, and metadata for attention operations.
- signature: `class OperationMetadata:`
- members:
  - `__post_init__(self)` — [`L141`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L141) — Initializes default values and performs safety checks after dataclass creation.
  - `_safety_check(self)` — [`L183`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L183) — Ensures no essential attributes are left uninitialized (as NOT_GIVEN).
  - `from_config(cls, config: EasyDeLBaseConfig)` — [`L192`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L192) — Factory method to create OperationMetadata from an EasyDeLBaseConfig.
  - `get_operation_config(self, impl_name: str)` — [`L345`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L345) — Get ejkernel config for a specific operation by its registered name.
  - `get_shardings(self, mode: RUNTIME_MODE_TYPES, layout: tp.Literal["bthd", "bhtd", "thd"] = "bthd", qkv_mni_sharding: bool = False, softmax_aux: jaxtyping.Array | None = None)` — [`L228`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L228) — Generates JAX PartitionSpecs for attention tensors based on runtime mode.
  - `mesh(self)` — [`L217`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L217) — Get current mesh from base_config if available, otherwise return stored mesh.
  - `set_attrs_carefully(self, attr_name: str, default: tp.Any | None, pickup_name: str | None = None, use_base_config: bool = True)` — [`L316`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L316) — Internal helper to set an attribute if it's not already set (or is Ellipsis).
  - `backend` — [`L125`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L125)
  - `base_config` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L130)
  - `operation_configs` — [`L131`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L131)
  - `partition_axis` — [`L127`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L127)
  - `partition_manager` — [`L128`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L128)
  - `platform` — [`L124`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L124)
  - `requires_cache` — [`L137`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L137)
  - `runtime_dtype` — [`L120`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L120)
  - `runtime_softmax_dtype` — [`L121`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L121)
  - `sequence_axis_name` — [`L123`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L123)
- protocol/private: `_stored_mesh`[`L139`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L139)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseConfig`](../infra/base_config.md#EasyDeLBaseConfig), [`EasyDeLBackends`](../infra/etils.md#EasyDeLBackends), [`EasyDeLPlatforms`](../infra/etils.md#EasyDeLPlatforms), [`mesh`](../infra/base_config.md#EasyDeLBaseConfig.mesh), [`query`](_operation_meta.md#AttnShardingRules.query), [`BATCH`](_operation_meta.md#BATCH), [`output`](_operation_meta.md#AttnShardingRules.output), [`NOT_GIVEN`](_operation_meta.md#NOT_GIVEN), [`attn_dtype`](../infra/base_config.md#EasyDeLBaseConfig.attn_dtype), [`partition_axis`](../infra/base_config.md#EasyDeLBaseConfig.partition_axis), [`key`](_operation_meta.md#AttnShardingRules.key), [`value`](_operation_meta.md#AttnShardingRules.value), [`attn_softmax_dtype`](../infra/base_config.md#EasyDeLBaseConfig.attn_softmax_dtype), [`KV_LENGTH`](_operation_meta.md#KV_LENGTH), [`softmax_aux`](_operation_meta.md#AttnShardingRules.softmax_aux), [`HEAD`](_operation_meta.md#HEAD), [`QUERY_LENGTH`](_operation_meta.md#QUERY_LENGTH), [`backend`](../infra/base_config.md#EasyDeLBaseConfig.backend), [`bias`](_operation_meta.md#AttnShardingRules.bias), [`HEAD_DIM`](_operation_meta.md#HEAD_DIM), [`AttnShardingRules`](_operation_meta.md#AttnShardingRules), [`query3d`](_operation_meta.md#AttnShardingRules.query3d), [`KV_HEAD`](_operation_meta.md#KV_HEAD), [`BIAS_HEAD_SEQ`](_operation_meta.md#BIAS_HEAD_SEQ), [`BIAS_KV_SEQ`](_operation_meta.md#BIAS_KV_SEQ), [`EMPTY`](_operation_meta.md#EMPTY), [`KV_HEAD_DIM`](_operation_meta.md#KV_HEAD_DIM), [`RUNTIME_MODE_TYPES`](_operation_meta.md#RUNTIME_MODE_TYPES), [`kv_segment_ids`](_operation_meta.md#AttnShardingRules.kv_segment_ids), [`mask`](_operation_meta.md#AttnShardingRules.mask), [`platform`](../infra/base_config.md#EasyDeLBaseConfig.platform), [`q_segment_ids`](_operation_meta.md#AttnShardingRules.q_segment_ids), [`sequence_axis_name`](../infra/base_config.md#EasyDeLBaseConfig.sequence_axis_name)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`metadata`](_base_operation.md#BaseOperation.metadata), [`forward_native`](kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`_forward_v2_turboquant`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`forward_core`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.forward_core), [`forward_core`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.forward_core), [`_forward_v3_standard`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_standard), [`forward_native`](kernels/unified_attention.md#UnifiedAttn.forward_native), [`forward_native`](kernels/blocksparse_attention.md#BlockSparseAttn.forward_native), [`create_stable_sharding`](_operation_impl.md#OperationImpl.create_stable_sharding), [`__call__`](_base_operation.md#BaseOperation.__call__), [`forward_native`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](kernels/ring_attention.md#RingAttn.forward_native), [`forward_v2`](kernels/ragged_page_attention.md#_RaggedPageAttn.forward_v2), [`forward_native`](kernels/flash_attention.md#FlashAttn.forward_native), [`forward_native`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_native), [`forward_native`](kernels/kda.md#KernelDeltaAttnOp.forward_native), [`forward_native`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.forward_native), [`forward_native`](kernels/vanilla_attention.md#VanillaAttn.forward_native), [`__init__`](../layers/attention/_flexible.md#FlexibleAttentionModule.__init__), [`forward_ragged`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_ragged), [`grouped_gdr_decode_shard_map_pallas`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.grouped_gdr_decode_shard_map_pallas), [`forward_native`](kernels/inference_gdn.md#RaggedGatedDeltaRule.forward_native), [`get_instance_requirements`](_operation_impl.md#OperationImpl.get_instance_requirements), [`dt_bias`](../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.dt_bias), [`out_proj`](../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.out_proj), [`out_proj`](../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.out_proj), [`kernels_proj`](../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaIndexer.kernels_proj), [`create`](_base_operation.md#OperationRegistry.create), [`dt_bias`](../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.dt_bias), [`_mapped`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn._mapped), [`_mapped`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2._mapped), [`_mapped`](kernels/ragged_page_attention.md#_RaggedPageAttn._mapped), [`_mapped`](kernels/unified_attention.md#UnifiedAttn._mapped), [`grouped_gdr_decode`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.grouped_gdr_decode), [`_mapped`](kernels/inference_gdn.md#RaggedGatedDeltaRule._mapped), [`cpu_attn`](kernels/scaled_dot_product_attention.md#cpu_attn), [`gpu_attn`](kernels/scaled_dot_product_attention.md#gpu_attn), [`ring`](kernels/ring_attention.md#ring)  (+15 more)

## Module values
- `BATCH` — [`L46`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L46)
- `BIAS_HEAD_SEQ` — [`L53`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L53)
- `BIAS_KV_SEQ` — [`L54`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L54)
- `EMPTY` — [`L55`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L55)
- `HEAD` — [`L49`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L49)
- `HEAD_DIM` — [`L51`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L51)
- `KV_HEAD` — [`L50`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L50)
- `KV_HEAD_DIM` — [`L52`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L52)
- `KV_LENGTH` — [`L48`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L48)
- `NOT_GIVEN` — [`L44`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L44)
- `QUERY_LENGTH` — [`L47`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L47)
- `RUNTIME_MODE_TYPES` — [`L45`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L45)
- `__all__` — [`L22`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L22)
- `logger` — [`L43`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_meta.py#L43)

