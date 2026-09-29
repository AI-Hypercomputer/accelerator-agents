---
title: 'Module: ejkernel/modules/operations/configs.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/configs.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.configs`/
symbols:
  BaseOperationConfig.backend: BaseOperationConfig#backend.
  BaseOperationConfig.platform: BaseOperationConfig#platform.
  BaseOperationConfig: BaseOperationConfig#
  QuantizedMatmulConfig: QuantizedMatmulConfig#
  hash_fn: hash_fn().
  FlashAttentionConfig: FlashAttentionConfig#
  BlockSparseAttentionConfig: BlockSparseAttentionConfig#
  AllGatherMatmulConfig: AllGatherMatmulConfig#
  RaggedPageAttentionv2Config: RaggedPageAttentionv2Config#
  RaggedPageAttentionv3Config: RaggedPageAttentionv3Config#
  GatedDeltaRuleConfig: GatedDeltaRuleConfig#
  RaggedGatedDeltaRuleV2Config: RaggedGatedDeltaRuleV2Config#
  RaggedGatedDeltaRuleConfig: RaggedGatedDeltaRuleConfig#
  FusedKLDivergenceConfig: FusedKLDivergenceConfig#
  UnifiedAttentionConfig: UnifiedAttentionConfig#
  RaggedDecodeAttentionConfig: RaggedDecodeAttentionConfig#
  PrefillPageAttentionConfig: PrefillPageAttentionConfig#
  NativeSparseAttentionConfig: NativeSparseAttentionConfig#
  PageAttentionConfig: PageAttentionConfig#
  DecodeAttentionConfig: DecodeAttentionConfig#
  ChunkedPrefillPagedDecodeConfig: ChunkedPrefillPagedDecodeConfig#
  RWKV4Config: RWKV4Config#
  RWKV6Config: RWKV6Config#
  ScaledDotProductAttentionConfig: ScaledDotProductAttentionConfig#
  FusedCrossEntropyConfig: FusedCrossEntropyConfig#
  QuantizedMatmulConfig.block_m: QuantizedMatmulConfig#block_m.
  QuantizedMatmulConfig.block_n: QuantizedMatmulConfig#block_n.
  QuantizedMatmulConfig.block_k: QuantizedMatmulConfig#block_k.
  RecurrentAttentionConfig: RecurrentAttentionConfig#
  RingAttentionConfig: RingAttentionConfig#
  GroupedMatmulConfig: GroupedMatmulConfig#
  ReduceScatterMatmulConfig: ReduceScatterMatmulConfig#
  MeanPoolingConfig: MeanPoolingConfig#
  RaggedPageAttentionv2TurboQuantConfig: RaggedPageAttentionv2TurboQuantConfig#
  RaggedPageAttentionv3TurboQuantConfig: RaggedPageAttentionv3TurboQuantConfig#
  GLAttentionConfig: GLAttentionConfig#
  LightningAttentionConfig: LightningAttentionConfig#
  KernelDeltaAttentionConfig: KernelDeltaAttentionConfig#
  RWKV7Config: RWKV7Config#
  RWKV7MulConfig: RWKV7MulConfig#
  StateSpaceV1Config: StateSpaceV1Config#
  MultiLatentRaggedPageAttentionConfig: MultiLatentRaggedPageAttentionConfig#
  MultiLatentRaggedPageAttentionV2Config: MultiLatentRaggedPageAttentionV2Config#
  GatedDeltaRuleGroupedDecodeConfig: GatedDeltaRuleGroupedDecodeConfig#
  DeepSeekAttentionConfig: DeepSeekAttentionConfig#
  StateSpaceV2Config: StateSpaceV2Config#
  AttentionConfig: AttentionConfig#
  RaggedCausalConv1DConfig: RaggedCausalConv1DConfig#
  FlashMLAConfig: FlashMLAConfig#
  QuantizedMatmulConfig.use_bf16: QuantizedMatmulConfig#use_bf16.
  FusedConvDecodeConfig: FusedConvDecodeConfig#
  QuantizedMatmulConfig.num_warps: QuantizedMatmulConfig#num_warps.
  QuantizedMatmulConfig.num_stages: QuantizedMatmulConfig#num_stages.
  GDNComputeScheduleV2Config: GDNComputeScheduleV2Config#
  QuantizedMatmulConfig.split_k: QuantizedMatmulConfig#split_k.
  FlashAttentionConfig.fwd_params: FlashAttentionConfig#fwd_params.
  FlashAttentionConfig.bwd_params: FlashAttentionConfig#bwd_params.
  BlockSparseAttentionConfig.fwd_params: BlockSparseAttentionConfig#fwd_params.
  BlockSparseAttentionConfig.bwd_params: BlockSparseAttentionConfig#bwd_params.
  UnifiedAttentionConfig.seq_threshold_3d: UnifiedAttentionConfig#seq_threshold_3d.
  UnifiedAttentionConfig.num_par_softmax_segments: UnifiedAttentionConfig#num_par_softmax_segments.
  UnifiedAttentionConfig.block_dim: UnifiedAttentionConfig#block_dim.
  PrefillPageAttentionConfig.block_k: PrefillPageAttentionConfig#block_k.
  DecodeAttentionConfig.num_kv_splits: DecodeAttentionConfig#num_kv_splits.
  DecodeAttentionConfig.num_warps: DecodeAttentionConfig#num_warps.
  DecodeAttentionConfig.num_stages: DecodeAttentionConfig#num_stages.
  ChunkedPrefillPagedDecodeConfig.seq_threshold_3d: ChunkedPrefillPagedDecodeConfig#seq_threshold_3d.
  ChunkedPrefillPagedDecodeConfig.num_par_softmax_segments: ChunkedPrefillPagedDecodeConfig#num_par_softmax_segments.
  ChunkedPrefillPagedDecodeConfig.num_warps: ChunkedPrefillPagedDecodeConfig#num_warps.
  ChunkedPrefillPagedDecodeConfig.num_stages: ChunkedPrefillPagedDecodeConfig#num_stages.
  GroupedMatmulConfig.block_m: GroupedMatmulConfig#block_m.
  PrefillPageAttentionConfig.num_warps: PrefillPageAttentionConfig#num_warps.
  PrefillPageAttentionConfig.num_stages: PrefillPageAttentionConfig#num_stages.
  GatedDeltaRuleConfig.chunk_size: GatedDeltaRuleConfig#chunk_size.
  RaggedGatedDeltaRuleV2Config.chunk_size: RaggedGatedDeltaRuleV2Config#chunk_size.
  RingAttentionConfig.fwd_params: RingAttentionConfig#fwd_params.
  RingAttentionConfig.bwd_params: RingAttentionConfig#bwd_params.
  RaggedDecodeAttentionConfig.fwd_params: RaggedDecodeAttentionConfig#fwd_params.
  QuantizedMatmulConfig.tpu_path: QuantizedMatmulConfig#tpu_path.
  ScaledDotProductAttentionConfig.block_q: ScaledDotProductAttentionConfig#block_q.
  ScaledDotProductAttentionConfig.block_k: ScaledDotProductAttentionConfig#block_k.
  ScaledDotProductAttentionConfig.num_warps: ScaledDotProductAttentionConfig#num_warps.
  ScaledDotProductAttentionConfig.num_stages: ScaledDotProductAttentionConfig#num_stages.
  FlashAttentionConfig.__post_init__: FlashAttentionConfig#__post_init__().
  BlockSparseAttentionConfig.__post_init__: BlockSparseAttentionConfig#__post_init__().
  NativeSparseAttentionConfig.block_k: NativeSparseAttentionConfig#block_k.
  NativeSparseAttentionConfig.block_d: NativeSparseAttentionConfig#block_d.
  NativeSparseAttentionConfig.block_size: NativeSparseAttentionConfig#block_size.
  NativeSparseAttentionConfig.num_warps: NativeSparseAttentionConfig#num_warps.
  NativeSparseAttentionConfig.num_stages: NativeSparseAttentionConfig#num_stages.
  RingAttentionConfig.__post_init__: RingAttentionConfig#__post_init__().
  UnifiedAttentionConfig.num_warps: UnifiedAttentionConfig#num_warps.
  UnifiedAttentionConfig.num_stages: UnifiedAttentionConfig#num_stages.
  RaggedPageAttentionv2Config.num_kv_pages_per_block: RaggedPageAttentionv2Config#num_kv_pages_per_block.
  RaggedPageAttentionv2Config.num_queries_per_block: RaggedPageAttentionv2Config#num_queries_per_block.
  RaggedPageAttentionv2Config.num_warps: RaggedPageAttentionv2Config#num_warps.
  RaggedPageAttentionv2Config.num_stages: RaggedPageAttentionv2Config#num_stages.
  KernelDeltaAttentionConfig.chunk_size: KernelDeltaAttentionConfig#chunk_size.
  RaggedGatedDeltaRuleConfig.chunk_size: RaggedGatedDeltaRuleConfig#chunk_size.
  FusedConvDecodeConfig.activation: FusedConvDecodeConfig#activation.
  NativeSparseAttentionConfig.block_q: NativeSparseAttentionConfig#block_q.
  RecurrentAttentionConfig.block_k: RecurrentAttentionConfig#block_k.
  RecurrentAttentionConfig.block_d: RecurrentAttentionConfig#block_d.
  RecurrentAttentionConfig.num_warps: RecurrentAttentionConfig#num_warps.
  RecurrentAttentionConfig.num_stages: RecurrentAttentionConfig#num_stages.
  PageAttentionConfig.num_splits: PageAttentionConfig#num_splits.
  PageAttentionConfig.pages_per_compute_block: PageAttentionConfig#pages_per_compute_block.
  PageAttentionConfig.num_warps: PageAttentionConfig#num_warps.
  PageAttentionConfig.num_stages: PageAttentionConfig#num_stages.
  AllGatherMatmulConfig.block_n: AllGatherMatmulConfig#block_n.
  AllGatherMatmulConfig.block_k: AllGatherMatmulConfig#block_k.
  RaggedPageAttentionv2TurboQuantConfig.num_kv_pages_per_block: RaggedPageAttentionv2TurboQuantConfig#num_kv_pages_per_block.
  RaggedPageAttentionv2TurboQuantConfig.num_queries_per_block: RaggedPageAttentionv2TurboQuantConfig#num_queries_per_block.
  RaggedPageAttentionv2TurboQuantConfig.num_warps: RaggedPageAttentionv2TurboQuantConfig#num_warps.
  RaggedPageAttentionv2TurboQuantConfig.num_stages: RaggedPageAttentionv2TurboQuantConfig#num_stages.
  RaggedPageAttentionv3TurboQuantConfig.num_kv_pages_per_block: RaggedPageAttentionv3TurboQuantConfig#num_kv_pages_per_block.
  RaggedPageAttentionv3TurboQuantConfig.num_queries_per_block: RaggedPageAttentionv3TurboQuantConfig#num_queries_per_block.
  MultiLatentRaggedPageAttentionV2Config.num_kv_pages_per_block: MultiLatentRaggedPageAttentionV2Config#num_kv_pages_per_block.
  MultiLatentRaggedPageAttentionV2Config.num_queries_per_block: MultiLatentRaggedPageAttentionV2Config#num_queries_per_block.
  GLAttentionConfig.block_k: GLAttentionConfig#block_k.
  GLAttentionConfig.block_d: GLAttentionConfig#block_d.
  GLAttentionConfig.num_warps: GLAttentionConfig#num_warps.
  GLAttentionConfig.num_stages: GLAttentionConfig#num_stages.
  LightningAttentionConfig.block_k: LightningAttentionConfig#block_k.
  LightningAttentionConfig.block_d: LightningAttentionConfig#block_d.
  LightningAttentionConfig.num_warps: LightningAttentionConfig#num_warps.
  LightningAttentionConfig.num_stages: LightningAttentionConfig#num_stages.
  RWKV4Config.block_c: RWKV4Config#block_c.
  RecurrentAttentionConfig.block_q: RecurrentAttentionConfig#block_q.
  GroupedMatmulConfig.block_n: GroupedMatmulConfig#block_n.
  GroupedMatmulConfig.block_k: GroupedMatmulConfig#block_k.
  AllGatherMatmulConfig.num_warps: AllGatherMatmulConfig#num_warps.
  AllGatherMatmulConfig.num_stages: AllGatherMatmulConfig#num_stages.
  ReduceScatterMatmulConfig.block_m: ReduceScatterMatmulConfig#block_m.
  ReduceScatterMatmulConfig.block_n: ReduceScatterMatmulConfig#block_n.
  ReduceScatterMatmulConfig.block_k: ReduceScatterMatmulConfig#block_k.
  MeanPoolingConfig.block_size: MeanPoolingConfig#block_size.
  MeanPoolingConfig.block_dim: MeanPoolingConfig#block_dim.
  MeanPoolingConfig.num_warps: MeanPoolingConfig#num_warps.
  MeanPoolingConfig.num_stages: MeanPoolingConfig#num_stages.
  RaggedPageAttentionv3TurboQuantConfig.chunk_prefill_size: RaggedPageAttentionv3TurboQuantConfig#chunk_prefill_size.
  RaggedPageAttentionv3TurboQuantConfig.num_warps: RaggedPageAttentionv3TurboQuantConfig#num_warps.
  RaggedPageAttentionv3TurboQuantConfig.num_stages: RaggedPageAttentionv3TurboQuantConfig#num_stages.
  GLAttentionConfig.block_q: GLAttentionConfig#block_q.
  LightningAttentionConfig.block_q: LightningAttentionConfig#block_q.
  RaggedCausalConv1DConfig.d_conv: RaggedCausalConv1DConfig#d_conv.
  RaggedCausalConv1DConfig.apply_silu: RaggedCausalConv1DConfig#apply_silu.
  FusedConvDecodeConfig.d_conv: FusedConvDecodeConfig#d_conv.
  MultiLatentRaggedPageAttentionV2Config.__post_init__: MultiLatentRaggedPageAttentionV2Config#__post_init__().
  GroupedMatmulConfig.num_warps: GroupedMatmulConfig#num_warps.
  GroupedMatmulConfig.num_stages: GroupedMatmulConfig#num_stages.
  ReduceScatterMatmulConfig.num_warps: ReduceScatterMatmulConfig#num_warps.
  ReduceScatterMatmulConfig.num_stages: ReduceScatterMatmulConfig#num_stages.
  MultiLatentRaggedPageAttentionConfig.chunk_prefill_size: MultiLatentRaggedPageAttentionConfig#chunk_prefill_size.
  MultiLatentRaggedPageAttentionConfig.num_kv_pages_per_block: MultiLatentRaggedPageAttentionConfig#num_kv_pages_per_block.
  MultiLatentRaggedPageAttentionConfig.num_queries_per_block: MultiLatentRaggedPageAttentionConfig#num_queries_per_block.
  MultiLatentRaggedPageAttentionConfig.vmem_limit_bytes: MultiLatentRaggedPageAttentionConfig#vmem_limit_bytes.
  MultiLatentRaggedPageAttentionV2Config.chunk_prefill_size: MultiLatentRaggedPageAttentionV2Config#chunk_prefill_size.
  MultiLatentRaggedPageAttentionV2Config.vmem_limit_bytes: MultiLatentRaggedPageAttentionV2Config#vmem_limit_bytes.
  StateSpaceV1Config.block_d: StateSpaceV1Config#block_d.
  StateSpaceV1Config.block_e: StateSpaceV1Config#block_e.
  StateSpaceV2Config.block_e: StateSpaceV2Config#block_e.
  FusedCrossEntropyConfig.block_v: FusedCrossEntropyConfig#block_v.
  FusedCrossEntropyConfig.block_m: FusedCrossEntropyConfig#block_m.
  FusedKLDivergenceConfig.block_v: FusedKLDivergenceConfig#block_v.
  FusedKLDivergenceConfig.block_m: FusedKLDivergenceConfig#block_m.
  RaggedPageAttentionv3Config.num_kv_pages_per_block: RaggedPageAttentionv3Config#num_kv_pages_per_block.
  RaggedPageAttentionv3Config.num_queries_per_block: RaggedPageAttentionv3Config#num_queries_per_block.
  MultiLatentRaggedPageAttentionConfig.num_warps: MultiLatentRaggedPageAttentionConfig#num_warps.
  MultiLatentRaggedPageAttentionConfig.num_stages: MultiLatentRaggedPageAttentionConfig#num_stages.
  MultiLatentRaggedPageAttentionV2Config.num_warps: MultiLatentRaggedPageAttentionV2Config#num_warps.
  MultiLatentRaggedPageAttentionV2Config.num_stages: MultiLatentRaggedPageAttentionV2Config#num_stages.
  RWKV7Config.block_v: RWKV7Config#block_v.
  RWKV7Config.num_warps: RWKV7Config#num_warps.
  RWKV7Config.num_stages: RWKV7Config#num_stages.
  RWKV7MulConfig.block_v: RWKV7MulConfig#block_v.
  RWKV7MulConfig.num_warps: RWKV7MulConfig#num_warps.
  RWKV7MulConfig.num_stages: RWKV7MulConfig#num_stages.
  FlashMLAConfig.block_q: FlashMLAConfig#block_q.
  FlashMLAConfig.block_k: FlashMLAConfig#block_k.
  FlashMLAConfig.num_warps: FlashMLAConfig#num_warps.
  FlashMLAConfig.num_stages: FlashMLAConfig#num_stages.
  DeepSeekAttentionConfig.index_topk: DeepSeekAttentionConfig#index_topk.
  DeepSeekAttentionConfig.gemm_block: DeepSeekAttentionConfig#gemm_block.
  StateSpaceV2Config.n_groups: StateSpaceV2Config#n_groups.
  StateSpaceV2Config.use_gated_rmsnorm: StateSpaceV2Config#use_gated_rmsnorm.
  StateSpaceV2Config.rmsnorm_eps: StateSpaceV2Config#rmsnorm_eps.
  FusedCrossEntropyConfig.num_warps: FusedCrossEntropyConfig#num_warps.
  FusedCrossEntropyConfig.num_stages: FusedCrossEntropyConfig#num_stages.
  FusedKLDivergenceConfig.num_warps: FusedKLDivergenceConfig#num_warps.
  FusedKLDivergenceConfig.num_stages: FusedKLDivergenceConfig#num_stages.
  AttentionConfig.block_q: AttentionConfig#block_q.
  AttentionConfig.block_k: AttentionConfig#block_k.
  AttentionConfig.weights_block_q: AttentionConfig#weights_block_q.
  AttentionConfig.weights_block_k: AttentionConfig#weights_block_k.
  AttentionConfig.num_warps: AttentionConfig#num_warps.
  AttentionConfig.num_stages: AttentionConfig#num_stages.
  RaggedDecodeAttentionConfig.__post_init__: RaggedDecodeAttentionConfig#__post_init__().
  RaggedPageAttentionv3Config.num_warps: RaggedPageAttentionv3Config#num_warps.
  RaggedPageAttentionv3Config.num_stages: RaggedPageAttentionv3Config#num_stages.
  DeepSeekAttentionConfig.block_q: DeepSeekAttentionConfig#block_q.
  DeepSeekAttentionConfig.block_k: DeepSeekAttentionConfig#block_k.
  DeepSeekAttentionConfig.num_warps: DeepSeekAttentionConfig#num_warps.
  DeepSeekAttentionConfig.num_stages: DeepSeekAttentionConfig#num_stages.
  GatedDeltaRuleConfig.use_input_dtype_phase1_outputs: GatedDeltaRuleConfig#use_input_dtype_phase1_outputs.
  GroupedMatmulConfig.bypass_xla_tiling: GroupedMatmulConfig#bypass_xla_tiling.
  GatedDeltaRuleConfig.use_chunked: GatedDeltaRuleConfig#use_chunked.
  GatedDeltaRuleConfig.use_input_dtype_state: GatedDeltaRuleConfig#use_input_dtype_state.
  RaggedGatedDeltaRuleV2Config.kernel_tile_policy: RaggedGatedDeltaRuleV2Config#kernel_tile_policy.
  RaggedGatedDeltaRuleV2Config.use_fused_gdn_decode: RaggedGatedDeltaRuleV2Config#use_fused_gdn_decode.
  BaseOperationConfig.__hash__: BaseOperationConfig#__hash__.
  BaseOperationConfig.to_json: BaseOperationConfig#to_json().
  BaseOperationConfig.from_json: BaseOperationConfig#from_json().
  FlashAttentionConfig.__hash__: FlashAttentionConfig#__hash__.
  BlockSparseAttentionConfig.__hash__: BlockSparseAttentionConfig#__hash__.
  NativeSparseAttentionConfig.__hash__: NativeSparseAttentionConfig#__hash__.
  RecurrentAttentionConfig.__hash__: RecurrentAttentionConfig#__hash__.
  RingAttentionConfig.__hash__: RingAttentionConfig#__hash__.
  PageAttentionConfig.__hash__: PageAttentionConfig#__hash__.
  UnifiedAttentionConfig.__hash__: UnifiedAttentionConfig#__hash__.
  DecodeAttentionConfig.__hash__: DecodeAttentionConfig#__hash__.
  ChunkedPrefillPagedDecodeConfig.__hash__: ChunkedPrefillPagedDecodeConfig#__hash__.
  AttentionConfig.__hash__: AttentionConfig#__hash__.
  GroupedMatmulConfig.__hash__: GroupedMatmulConfig#__hash__.
  AllGatherMatmulConfig.__hash__: AllGatherMatmulConfig#__hash__.
  ReduceScatterMatmulConfig.__hash__: ReduceScatterMatmulConfig#__hash__.
  QuantizedMatmulConfig.__hash__: QuantizedMatmulConfig#__hash__.
  MeanPoolingConfig.__hash__: MeanPoolingConfig#__hash__.
  RaggedDecodeAttentionConfig.__hash__: RaggedDecodeAttentionConfig#__hash__.
  RaggedPageAttentionv2Config.__hash__: RaggedPageAttentionv2Config#__hash__.
  RaggedPageAttentionv2TurboQuantConfig.__hash__: RaggedPageAttentionv2TurboQuantConfig#__hash__.
  RaggedPageAttentionv3Config.__hash__: RaggedPageAttentionv3Config#__hash__.
  RaggedPageAttentionv3TurboQuantConfig.__hash__: RaggedPageAttentionv3TurboQuantConfig#__hash__.
  MultiLatentRaggedPageAttentionConfig.__hash__: MultiLatentRaggedPageAttentionConfig#__hash__.
  MultiLatentRaggedPageAttentionV2Config.__hash__: MultiLatentRaggedPageAttentionV2Config#__hash__.
  GLAttentionConfig.__hash__: GLAttentionConfig#__hash__.
  LightningAttentionConfig.__hash__: LightningAttentionConfig#__hash__.
  KernelDeltaAttentionConfig.__hash__: KernelDeltaAttentionConfig#__hash__.
  GatedDeltaRuleConfig.__hash__: GatedDeltaRuleConfig#__hash__.
  RaggedGatedDeltaRuleConfig.__hash__: RaggedGatedDeltaRuleConfig#__hash__.
  RaggedCausalConv1DConfig.__hash__: RaggedCausalConv1DConfig#__hash__.
  FusedConvDecodeConfig.__hash__: FusedConvDecodeConfig#__hash__.
  GatedDeltaRuleGroupedDecodeConfig.__hash__: GatedDeltaRuleGroupedDecodeConfig#__hash__.
  RaggedGatedDeltaRuleV2Config.__hash__: RaggedGatedDeltaRuleV2Config#__hash__.
  GDNComputeScheduleV2Config.__hash__: GDNComputeScheduleV2Config#__hash__.
  RWKV4Config.__hash__: RWKV4Config#__hash__.
  RWKV6Config.__hash__: RWKV6Config#__hash__.
  RWKV7Config.__hash__: RWKV7Config#__hash__.
  RWKV7MulConfig.__hash__: RWKV7MulConfig#__hash__.
  FlashMLAConfig.__hash__: FlashMLAConfig#__hash__.
  DeepSeekAttentionConfig.__hash__: DeepSeekAttentionConfig#__hash__.
  ScaledDotProductAttentionConfig.__hash__: ScaledDotProductAttentionConfig#__hash__.
  PrefillPageAttentionConfig.__hash__: PrefillPageAttentionConfig#__hash__.
  StateSpaceV1Config.__hash__: StateSpaceV1Config#__hash__.
  StateSpaceV2Config.__hash__: StateSpaceV2Config#__hash__.
  FusedCrossEntropyConfig.__hash__: FusedCrossEntropyConfig#__hash__.
  MultiLatentRaggedPageAttentionV2Config._normalize: MultiLatentRaggedPageAttentionV2Config#_normalize().
  get_safe_hash_int: get_safe_hash_int().
  BaseOperationConfig.to_dict: BaseOperationConfig#to_dict().
  BaseOperationConfig.from_dict: BaseOperationConfig#from_dict().
  RaggedPageAttentionv3Config.chunk_prefill_size: RaggedPageAttentionv3Config#chunk_prefill_size.
  FusedKLDivergenceConfig.__hash__: FusedKLDivergenceConfig#__hash__.
---
# Module: [`ejkernel/modules/operations/configs.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py)

## Classes
### `AllGatherMatmulConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:430`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L430)
- doc: Configuration for the fused All-Gather + Matmul operation.
- signature: `class AllGatherMatmulConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L449`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L449)
  - `block_n` — [`L448`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L448)
  - `num_stages` — [`L451`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L451)
  - `num_warps` — [`L450`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L450)
- protocol/private: `__hash__`[`L453`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L453)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_tpu`](all_gather_matmul.md#AllGatherMatmul.candidate_cfgs_tpu), [`run`](all_gather_matmul.md#AllGatherMatmul.run), [`_all_gather_matmul_executor`](all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`heuristic_cfg_tpu`](all_gather_matmul.md#AllGatherMatmul.heuristic_cfg_tpu), [`candidate_cfgs_gpu`](all_gather_matmul.md#AllGatherMatmul.candidate_cfgs_gpu), [`candidate_cfgs`](all_gather_matmul.md#AllGatherMatmul.candidate_cfgs), [`heuristic_cfg`](all_gather_matmul.md#AllGatherMatmul.heuristic_cfg), [`get_impl`](all_gather_matmul.md#AllGatherMatmul.get_impl), [`_inv_xy_rhs_transpose`](all_gather_matmul.md#_inv_xy_rhs_transpose), [`all_gather_matmul`](all_gather_matmul.md#all_gather_matmul), [`AllGatherMatmul`](all_gather_matmul.md#AllGatherMatmul), [`create_shard_map_wrapper`](all_gather_matmul.md#AllGatherMatmul.create_shard_map_wrapper)

### `AttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:373`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L373)
- doc: Configuration for basic Attention operation.
- signature: `class AttentionConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L391`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L391)
  - `block_q` — [`L390`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L390)
  - `num_stages` — [`L395`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L395)
  - `num_warps` — [`L394`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L394)
  - `weights_block_k` — [`L393`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L393)
  - `weights_block_q` — [`L392`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L392)
- protocol/private: `__hash__`[`L397`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L397)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](attention.md#Attention.run), [`heuristic_cfg`](attention.md#Attention.heuristic_cfg), [`candidate_cfgs_gpu`](attention.md#Attention.candidate_cfgs_gpu), [`candidate_cfgs`](attention.md#Attention.candidate_cfgs), [`_executor`](attention.md#_executor._executor), [`candidate_cfgs_tpu`](attention.md#Attention.candidate_cfgs_tpu), [`get_impl`](attention.md#Attention.get_impl), [`_seqlens_from_inv`](attention.md#Attention._seqlens_from_inv), [`Attention`](attention.md#Attention)

### `BaseOperationConfig`
- def: [`ejkernel/modules/operations/configs.py:112`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L112) — documented in [ejkernel-modules-operations-configs](../../../../concepts/ejkernel-modules-operations-configs.md)
- doc: Base configuration for all operations.
- signature: `class BaseOperationConfig:`
- members:
  - `from_dict(cls, data: dict)` — [`L148`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L148) — Deserialize a config from a plain dictionary.
  - `from_json(cls, s: str)` — [`L157`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L157) — Deserialize a config from a JSON string.
  - `to_dict(self)` — [`L143`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L143) — Serialize this config to a plain dictionary.
  - `to_json(self)` — [`L152`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L152) — Serialize this config to a JSON string.
  - `backend` — [`L139`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L139) — documented in [ejkernel-modules-operations-configs](../../../../concepts/ejkernel-modules-operations-configs.md)
  - `platform` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L138) — documented in [ejkernel-modules-operations-configs](../../../../concepts/ejkernel-modules-operations-configs.md)
- protocol/private: `__hash__`[`L141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L141)
- uses (calls/refs, reference-scoped): [`QuantizedMatmulConfig`](configs.md#QuantizedMatmulConfig), [`hash_fn`](configs.md#hash_fn), [`BlockSparseAttentionConfig`](configs.md#BlockSparseAttentionConfig), [`FlashAttentionConfig`](configs.md#FlashAttentionConfig), [`AllGatherMatmulConfig`](configs.md#AllGatherMatmulConfig), [`GatedDeltaRuleConfig`](configs.md#GatedDeltaRuleConfig), [`RaggedGatedDeltaRuleV2Config`](configs.md#RaggedGatedDeltaRuleV2Config), [`RaggedPageAttentionv2Config`](configs.md#RaggedPageAttentionv2Config), [`RaggedPageAttentionv3Config`](configs.md#RaggedPageAttentionv3Config), [`RaggedGatedDeltaRuleConfig`](configs.md#RaggedGatedDeltaRuleConfig), [`FusedKLDivergenceConfig`](configs.md#FusedKLDivergenceConfig), [`PrefillPageAttentionConfig`](configs.md#PrefillPageAttentionConfig), [`RaggedDecodeAttentionConfig`](configs.md#RaggedDecodeAttentionConfig), [`UnifiedAttentionConfig`](configs.md#UnifiedAttentionConfig), [`ChunkedPrefillPagedDecodeConfig`](configs.md#ChunkedPrefillPagedDecodeConfig), [`DecodeAttentionConfig`](configs.md#DecodeAttentionConfig), [`FusedCrossEntropyConfig`](configs.md#FusedCrossEntropyConfig), [`NativeSparseAttentionConfig`](configs.md#NativeSparseAttentionConfig), [`PageAttentionConfig`](configs.md#PageAttentionConfig), [`RWKV4Config`](configs.md#RWKV4Config), [`RWKV6Config`](configs.md#RWKV6Config), [`ScaledDotProductAttentionConfig`](configs.md#ScaledDotProductAttentionConfig), [`GLAttentionConfig`](configs.md#GLAttentionConfig), [`GroupedMatmulConfig`](configs.md#GroupedMatmulConfig), [`KernelDeltaAttentionConfig`](configs.md#KernelDeltaAttentionConfig), [`LightningAttentionConfig`](configs.md#LightningAttentionConfig), [`MeanPoolingConfig`](configs.md#MeanPoolingConfig), [`RWKV7Config`](configs.md#RWKV7Config), [`RWKV7MulConfig`](configs.md#RWKV7MulConfig), [`RaggedPageAttentionv2TurboQuantConfig`](configs.md#RaggedPageAttentionv2TurboQuantConfig), [`RaggedPageAttentionv3TurboQuantConfig`](configs.md#RaggedPageAttentionv3TurboQuantConfig), [`RecurrentAttentionConfig`](configs.md#RecurrentAttentionConfig), [`ReduceScatterMatmulConfig`](configs.md#ReduceScatterMatmulConfig), [`RingAttentionConfig`](configs.md#RingAttentionConfig), [`StateSpaceV1Config`](configs.md#StateSpaceV1Config), [`DeepSeekAttentionConfig`](configs.md#DeepSeekAttentionConfig), [`GatedDeltaRuleGroupedDecodeConfig`](configs.md#GatedDeltaRuleGroupedDecodeConfig), [`MultiLatentRaggedPageAttentionConfig`](configs.md#MultiLatentRaggedPageAttentionConfig), [`MultiLatentRaggedPageAttentionV2Config`](configs.md#MultiLatentRaggedPageAttentionV2Config), [`StateSpaceV2Config`](configs.md#StateSpaceV2Config)  (+5 more)
- used by: [`QuantizedMatmulConfig`](configs.md#QuantizedMatmulConfig), [`_candidate_cfgs_gpu_for_platform`](quantized_matmul.md#QuantizedMatmul._candidate_cfgs_gpu_for_platform), [`_pallas_tpu_candidate_cfgs`](quantized_matmul.md#_pallas_tpu_candidate_cfgs), [`_quantized_matmul_impl`](quantized_matmul.md#_quantized_matmul_impl), [`quantized_matmul`](quantized_matmul.md#quantized_matmul), [`_xla_candidate_cfgs`](quantized_matmul.md#_xla_candidate_cfgs), [`run`](scaled_dot_product_attention.md#ScaledDotProductAttention.run), [`candidate_cfgs_gpu`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_tpu), [`candidate_cfgs_tpu`](flash_attention.md#FlashAttention.candidate_cfgs_tpu), [`run`](quantized_matmul.md#QuantizedMatmul.run), [`candidate_cfgs_xla`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_xla), [`candidate_cfgs_xla`](flash_attention.md#FlashAttention.candidate_cfgs_xla), [`candidate_cfgs_gpu`](flash_attention.md#FlashAttention.candidate_cfgs_gpu), [`_pallas_tpu_heuristic_cfg`](quantized_matmul.md#_pallas_tpu_heuristic_cfg), [`candidate_cfgs_gpu`](ring_attention.md#RingAttention.candidate_cfgs_gpu), [`heuristic_cfg`](ring_attention.md#RingAttention.heuristic_cfg), [`_triton_heuristic_cfg`](quantized_matmul.md#_triton_heuristic_cfg), [`heuristic_cfg_gpu`](flash_attention.md#FlashAttention.heuristic_cfg_gpu), [`_cuda_heuristic_cfg`](quantized_matmul.md#_cuda_heuristic_cfg), [`_normalize_pallas_tpu_packed_cfg_forward`](quantized_matmul.md#_normalize_pallas_tpu_packed_cfg_forward), [`_tilelang_heuristic_cfg`](quantized_matmul.md#_tilelang_heuristic_cfg), [`candidate_cfgs`](ring_attention.md#RingAttention.candidate_cfgs), [`candidate_cfgs_tpu`](ring_attention.md#RingAttention.candidate_cfgs_tpu), [`heuristic_cfg`](blocksparse_attention.md#BlockSparseAttention.heuristic_cfg), [`candidate_cfgs`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs), [`candidate_cfgs`](flash_attention.md#FlashAttention.candidate_cfgs), [`heuristic_cfg`](flash_attention.md#FlashAttention.heuristic_cfg), [`heuristic_cfg_gpu`](blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_gpu), [`heuristic_cfg_tpu`](blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_tpu), [`heuristic_cfg_tpu`](flash_attention.md#FlashAttention.heuristic_cfg_tpu), [`run`](deepseek_attn.md#DeepSeekAttention.run), [`run`](native_sparse_attention.md#NativeSparseAttention.run), [`run`](pooling.md#MeanPooling.run), [`run`](fused_cross_entropy.md#FusedCrossEntropy.run), [`run`](fused_kl_divergence.md#FusedKLDivergence.run), [`run`](gated_delta_rule.md#GatedDeltaRule.run), [`run`](gated_linear_attention.md#GLAttention.run), [`run`](lightning_attention.md#LightningAttention.run), [`run`](recurrent.md#RecurrentAttention.run)  (+285 more)

### `BlockSparseAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:187`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L187) — documented in [ejkernel-modules-operations-configs](../../../../concepts/ejkernel-modules-operations-configs.md)
- doc: Configuration for Block Sparse Attention operation.
- signature: `class BlockSparseAttentionConfig(BaseOperationConfig):`
- members:
  - `__post_init__(self)` — [`L200`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L200) — Convert dict-typed forward/backward params to FwdParams/BwdParams.
  - `bwd_params` — [`L198`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L198)
  - `fwd_params` — [`L197`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L197)
- protocol/private: `__hash__`[`L207`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L207)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_gpu`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_tpu), [`candidate_cfgs_xla`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_xla), [`heuristic_cfg`](blocksparse_attention.md#BlockSparseAttention.heuristic_cfg), [`candidate_cfgs`](blocksparse_attention.md#BlockSparseAttention.candidate_cfgs), [`heuristic_cfg_gpu`](blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_gpu), [`heuristic_cfg_tpu`](blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_tpu), [`blocksparse_attention`](blocksparse_attention.md#blocksparse_attention), [`_executor`](blocksparse_attention.md#_executor._executor), [`run`](blocksparse_attention.md#BlockSparseAttention.run), [`get_impl`](blocksparse_attention.md#BlockSparseAttention.get_impl), [`create_shard_map_wrapper`](blocksparse_attention.md#BlockSparseAttention.create_shard_map_wrapper), [`BlockSparseAttention`](blocksparse_attention.md#BlockSparseAttention)

### `ChunkedPrefillPagedDecodeConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:348`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L348)
- doc: Configuration for chunked prefill + paged decode attention operation.
- signature: `class ChunkedPrefillPagedDecodeConfig(BaseOperationConfig):`
- members:
  - `num_par_softmax_segments` — [`L365`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L365)
  - `num_stages` — [`L367`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L367)
  - `num_warps` — [`L366`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L366)
  - `seq_threshold_3d` — [`L364`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L364)
- protocol/private: `__hash__`[`L369`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L369)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_gpu`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.candidate_cfgs_gpu), [`heuristic_cfg`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.heuristic_cfg), [`_chunked_prefill_paged_decode_executor`](chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`run`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.run), [`candidate_cfgs`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.candidate_cfgs), [`candidate_cfgs_tpu`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.candidate_cfgs_tpu), [`get_impl`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.get_impl), [`chunked_prefill_paged_decode`](chunked_prefill_paged_decode.md#chunked_prefill_paged_decode), [`ChunkedPrefillPagedDecode`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode), [`create_shard_map_wrapper`](chunked_prefill_paged_decode.md#ChunkedPrefillPagedDecode.create_shard_map_wrapper)

### `DecodeAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:329`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L329)
- doc: Configuration for vLLM-style paged decode attention operation.
- signature: `class DecodeAttentionConfig(BaseOperationConfig):`
- members:
  - `num_kv_splits` — [`L340`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L340)
  - `num_stages` — [`L342`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L342)
  - `num_warps` — [`L341`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L341)
- protocol/private: `__hash__`[`L344`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L344)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_decode_attention_executor`](decode_attention.md#_decode_attention_executor._decode_attention_executor), [`candidate_cfgs_gpu`](decode_attention.md#DecodeAttention.candidate_cfgs_gpu), [`heuristic_cfg`](decode_attention.md#DecodeAttention.heuristic_cfg), [`run`](decode_attention.md#DecodeAttention.run), [`candidate_cfgs`](decode_attention.md#DecodeAttention.candidate_cfgs), [`candidate_cfgs_tpu`](decode_attention.md#DecodeAttention.candidate_cfgs_tpu), [`get_impl`](decode_attention.md#DecodeAttention.get_impl), [`decode_attention`](decode_attention.md#decode_attention), [`DecodeAttention`](decode_attention.md#DecodeAttention), [`create_shard_map_wrapper`](decode_attention.md#DecodeAttention.create_shard_map_wrapper)

### `DeepSeekAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1093`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1093)
- doc: Configuration for DeepSeek Sparse Attention (DSA) operation.
- signature: `class DeepSeekAttentionConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L1119`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1119)
  - `block_q` — [`L1118`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1118)
  - `gemm_block` — [`L1120`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1120)
  - `index_topk` — [`L1117`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1117)
  - `num_stages` — [`L1122`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1122)
  - `num_warps` — [`L1121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1121)
- protocol/private: `__hash__`[`L1124`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1124)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](deepseek_attn.md#DeepSeekAttention.run), [`heuristic_cfg`](deepseek_attn.md#DeepSeekAttention.heuristic_cfg), [`candidate_cfgs_gpu`](deepseek_attn.md#DeepSeekAttention.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](deepseek_attn.md#DeepSeekAttention.candidate_cfgs_tpu), [`_dsa_executor`](deepseek_attn.md#_dsa_executor._dsa_executor), [`get_impl`](deepseek_attn.md#DeepSeekAttention.get_impl), [`_seq_len_from_inv`](deepseek_attn.md#DeepSeekAttention._seq_len_from_inv), [`deepseek_attn`](deepseek_attn.md#deepseek_attn), [`DeepSeekAttention`](deepseek_attn.md#DeepSeekAttention), [`candidate_cfgs`](deepseek_attn.md#DeepSeekAttention.candidate_cfgs)

### `FlashAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:163`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L163) — documented in [ejkernel-modules-operations-configs](../../../../concepts/ejkernel-modules-operations-configs.md)
- doc: Configuration for Flash Attention operation.
- signature: `class FlashAttentionConfig(BaseOperationConfig):`
- members:
  - `__post_init__(self)` — [`L176`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L176) — Convert dict-typed forward/backward params to FwdParams/BwdParams.
  - `bwd_params` — [`L174`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L174)
  - `fwd_params` — [`L173`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L173) — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)
- protocol/private: `__hash__`[`L183`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L183)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_tpu`](flash_attention.md#FlashAttention.candidate_cfgs_tpu), [`candidate_cfgs_xla`](flash_attention.md#FlashAttention.candidate_cfgs_xla), [`candidate_cfgs_gpu`](flash_attention.md#FlashAttention.candidate_cfgs_gpu), [`heuristic_cfg_gpu`](flash_attention.md#FlashAttention.heuristic_cfg_gpu), [`flash_attention`](flash_attention.md#flash_attention), [`candidate_cfgs`](flash_attention.md#FlashAttention.candidate_cfgs), [`heuristic_cfg`](flash_attention.md#FlashAttention.heuristic_cfg), [`heuristic_cfg_tpu`](flash_attention.md#FlashAttention.heuristic_cfg_tpu), [`run`](flash_attention.md#FlashAttention.run), [`_flash_executor`](flash_attention.md#_flash_executor._flash_executor), [`get_impl`](flash_attention.md#FlashAttention.get_impl), [`create_shard_map_wrapper`](flash_attention.md#FlashAttention.create_shard_map_wrapper), [`FlashAttention`](flash_attention.md#FlashAttention)

### `FlashMLAConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1072`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1072)
- doc: Configuration for Flash Multi-head Latent Attention operation.
- signature: `class FlashMLAConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L1085`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1085)
  - `block_q` — [`L1084`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1084)
  - `num_stages` — [`L1087`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1087)
  - `num_warps` — [`L1086`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1086)
- protocol/private: `__hash__`[`L1089`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1089)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_mla_executor`](multi_head_latent_attention.md#_mla_executor._mla_executor), [`candidate_cfgs_gpu`](multi_head_latent_attention.md#FlashMLA.candidate_cfgs_gpu), [`run`](multi_head_latent_attention.md#FlashMLA.run), [`heuristic_cfg`](multi_head_latent_attention.md#FlashMLA.heuristic_cfg), [`candidate_cfgs_tpu`](multi_head_latent_attention.md#FlashMLA.candidate_cfgs_tpu), [`get_impl`](multi_head_latent_attention.md#FlashMLA.get_impl), [`flash_mla`](multi_head_latent_attention.md#flash_mla), [`candidate_cfgs`](multi_head_latent_attention.md#FlashMLA.candidate_cfgs), [`FlashMLA`](multi_head_latent_attention.md#FlashMLA)

### `FusedConvDecodeConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:905`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L905)
- doc: Configuration for fused conv-state shift + depthwise conv decode.
- signature: `class FusedConvDecodeConfig(BaseOperationConfig):`
- members:
  - `activation` — [`L916`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L916)
  - `d_conv` — [`L915`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L915)
- protocol/private: `__hash__`[`L918`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L918)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](fused_conv_decode.md#_executor._executor), [`heuristic_cfg`](fused_conv_decode.md#FusedConvDecode.heuristic_cfg), [`run`](fused_conv_decode.md#FusedConvDecode.run), [`candidate_cfgs_tpu`](fused_conv_decode.md#FusedConvDecode.candidate_cfgs_tpu), [`fused_conv_decode`](fused_conv_decode.md#fused_conv_decode), [`get_impl`](fused_conv_decode.md#FusedConvDecode.get_impl), [`candidate_cfgs`](fused_conv_decode.md#FusedConvDecode.candidate_cfgs), [`FusedConvDecode`](fused_conv_decode.md#FusedConvDecode), [`candidate_cfgs_gpu`](fused_conv_decode.md#FusedConvDecode.candidate_cfgs_gpu)

### `FusedCrossEntropyConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1210`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1210)
- doc: Configuration for fused cross-entropy.
- signature: `class FusedCrossEntropyConfig(BaseOperationConfig):`
- members:
  - `block_m` — [`L1233`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1233)
  - `block_v` — [`L1232`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1232)
  - `num_stages` — [`L1235`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1235)
  - `num_warps` — [`L1234`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1234)
- protocol/private: `__hash__`[`L1237`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1237)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_fused_linear_cross_entropy_dispatch`](fused_cross_entropy.md#_fused_linear_cross_entropy_dispatch), [`run`](fused_cross_entropy.md#FusedCrossEntropy.run), [`fused_cross_entropy`](fused_cross_entropy.md#fused_cross_entropy), [`candidate_cfgs_gpu`](fused_cross_entropy.md#FusedCrossEntropy.candidate_cfgs_gpu), [`heuristic_cfg`](fused_cross_entropy.md#FusedCrossEntropy.heuristic_cfg), [`_executor`](fused_cross_entropy.md#_executor._executor), [`candidate_cfgs_tpu`](fused_cross_entropy.md#FusedCrossEntropy.candidate_cfgs_tpu), [`create_shard_map_wrapper`](fused_cross_entropy.md#FusedCrossEntropy.create_shard_map_wrapper), [`get_impl`](fused_cross_entropy.md#FusedCrossEntropy.get_impl), [`_shape_from_inv`](fused_cross_entropy.md#FusedCrossEntropy._shape_from_inv), [`candidate_cfgs`](fused_cross_entropy.md#FusedCrossEntropy.candidate_cfgs), [`FusedCrossEntropy`](fused_cross_entropy.md#FusedCrossEntropy)

### `FusedKLDivergenceConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1241`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1241)
- doc: Configuration for fused KL divergence (forward / reverse / JSD).
- signature: `class FusedKLDivergenceConfig(BaseOperationConfig):`
- members:
  - `block_m` — [`L1261`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1261)
  - `block_v` — [`L1260`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1260)
  - `num_stages` — [`L1263`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1263)
  - `num_warps` — [`L1262`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1262)
- protocol/private: `__hash__`[`L1265`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1265)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](fused_kl_divergence.md#FusedKLDivergence.run), [`candidate_cfgs_gpu`](fused_kl_divergence.md#FusedKLDivergence.candidate_cfgs_gpu), [`heuristic_cfg`](fused_kl_divergence.md#FusedKLDivergence.heuristic_cfg), [`_executor`](fused_kl_divergence.md#_executor._executor), [`candidate_cfgs_tpu`](fused_kl_divergence.md#FusedKLDivergence.candidate_cfgs_tpu), [`fused_kl_divergence`](fused_kl_divergence.md#fused_kl_divergence), [`create_shard_map_wrapper`](fused_kl_divergence.md#FusedKLDivergence.create_shard_map_wrapper), [`get_impl`](fused_kl_divergence.md#FusedKLDivergence.get_impl), [`_shape_from_inv`](fused_kl_divergence.md#FusedKLDivergence._shape_from_inv), [`candidate_cfgs`](fused_kl_divergence.md#FusedKLDivergence.candidate_cfgs), [`FusedKLDivergence`](fused_kl_divergence.md#FusedKLDivergence)

### `GDNComputeScheduleV2Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:962`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L962)
- doc: Configuration for GDN v2 schedule-table construction.
- signature: `class GDNComputeScheduleV2Config(BaseOperationConfig):`
- protocol/private: `__hash__`[`L974`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L974)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](gdn_compute_schedule_v2.md#_executor._executor), [`get_impl`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2.get_impl), [`compute_schedule_table_v2`](gdn_compute_schedule_v2.md#compute_schedule_table_v2), [`heuristic_cfg`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2.heuristic_cfg), [`candidate_cfgs_gpu`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2.candidate_cfgs_tpu), [`candidate_cfgs`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2.candidate_cfgs), [`GDNComputeScheduleV2`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2), [`run`](gdn_compute_schedule_v2.md#GDNComputeScheduleV2.run)

### `GLAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:770`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L770)
- doc: Configuration for Gated Linear Attention operation.
- signature: `class GLAttentionConfig(BaseOperationConfig):`
- members:
  - `block_d` — [`L785`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L785)
  - `block_k` — [`L784`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L784)
  - `block_q` — [`L783`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L783)
  - `num_stages` — [`L787`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L787)
  - `num_warps` — [`L786`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L786)
- protocol/private: `__hash__`[`L789`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L789)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](gated_linear_attention.md#GLAttention.run), [`candidate_cfgs_gpu`](gated_linear_attention.md#GLAttention.candidate_cfgs_gpu), [`_gla_executor`](gated_linear_attention.md#_gla_executor._gla_executor), [`candidate_cfgs`](gated_linear_attention.md#GLAttention.candidate_cfgs), [`candidate_cfgs_tpu`](gated_linear_attention.md#GLAttention.candidate_cfgs_tpu), [`heuristic_cfg`](gated_linear_attention.md#GLAttention.heuristic_cfg), [`get_impl`](gated_linear_attention.md#GLAttention.get_impl), [`gla_attention`](gated_linear_attention.md#gla_attention), [`GLAttention`](gated_linear_attention.md#GLAttention)

### `GatedDeltaRuleConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:832`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L832)
- doc: Configuration for Gated Delta Rule (GDR) operation.
- signature: `class GatedDeltaRuleConfig(BaseOperationConfig):`
- members:
  - `chunk_size` — [`L861`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L861)
  - `use_chunked` — [`L862`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L862)
  - `use_input_dtype_phase1_outputs` — [`L863`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L863)
  - `use_input_dtype_state` — [`L864`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L864)
- protocol/private: `__hash__`[`L866`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L866)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](gated_delta_rule.md#GatedDeltaRule.run), [`_executor`](gated_delta_rule.md#_executor._executor), [`candidate_cfgs_gpu`](gated_delta_rule.md#GatedDeltaRule.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](gated_delta_rule.md#GatedDeltaRule.candidate_cfgs_tpu), [`candidate_cfgs`](gated_delta_rule.md#GatedDeltaRule.candidate_cfgs), [`get_impl`](gated_delta_rule.md#GatedDeltaRule.get_impl), [`_wrapped_gdr`](gated_delta_rule.md#GatedDeltaRule._wrapped_gdr), [`heuristic_cfg`](gated_delta_rule.md#GatedDeltaRule.heuristic_cfg), [`gated_delta_rule`](gated_delta_rule.md#gated_delta_rule), [`resolve_platform`](gated_delta_rule.md#GatedDeltaRule.resolve_platform), [`GatedDeltaRule`](gated_delta_rule.md#GatedDeltaRule), [`create_shard_map_wrapper`](gated_delta_rule.md#GatedDeltaRule.create_shard_map_wrapper)

### `GatedDeltaRuleGroupedDecodeConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:922`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L922)
- doc: Configuration for grouped GDR single-step decode.
- signature: `class GatedDeltaRuleGroupedDecodeConfig(BaseOperationConfig):`
- protocol/private: `__hash__`[`L932`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L932)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](gated_delta_rule_grouped_decode.md#_executor._executor), [`get_impl`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.get_impl), [`run`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.run), [`heuristic_cfg`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.heuristic_cfg), [`gated_delta_rule_grouped_decode`](gated_delta_rule_grouped_decode.md#gated_delta_rule_grouped_decode), [`candidate_cfgs`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.candidate_cfgs), [`candidate_cfgs_tpu`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.candidate_cfgs_tpu), [`create_shard_map_wrapper`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.create_shard_map_wrapper), [`candidate_cfgs_gpu`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode.candidate_cfgs_gpu), [`GatedDeltaRuleGroupedDecode`](gated_delta_rule_grouped_decode.md#GatedDeltaRuleGroupedDecode)

### `GroupedMatmulConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:401`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L401)
- doc: Configuration for Grouped Matrix Multiplication operation.
- signature: `class GroupedMatmulConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L421`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L421)
  - `block_m` — [`L419`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L419)
  - `block_n` — [`L420`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L420)
  - `bypass_xla_tiling` — [`L424`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L424)
  - `num_stages` — [`L423`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L423)
  - `num_warps` — [`L422`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L422)
- protocol/private: `__hash__`[`L426`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L426)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](grouped_matmul.md#GroupedMatmul.run), [`candidate_cfgs_gpu`](grouped_matmul.md#GroupedMatmul.candidate_cfgs_gpu), [`_grouped_matmul_executor`](grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`candidate_cfgs`](grouped_matmul.md#GroupedMatmul.candidate_cfgs), [`heuristic_cfg`](grouped_matmul.md#GroupedMatmul.heuristic_cfg), [`candidate_cfgs_tpu`](grouped_matmul.md#GroupedMatmul.candidate_cfgs_tpu), [`get_impl`](grouped_matmul.md#GroupedMatmul.get_impl), [`create_shard_map_wrapper`](grouped_matmul.md#GroupedMatmul.create_shard_map_wrapper), [`grouped_matmul`](grouped_matmul.md#grouped_matmul), [`GroupedMatmul`](grouped_matmul.md#GroupedMatmul)

### `KernelDeltaAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:816`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L816)
- doc: Configuration for Kernel Delta Attention (KDA) operation.
- signature: `class KernelDeltaAttentionConfig(BaseOperationConfig):`
- members:
  - `chunk_size` — [`L826`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L826)
- protocol/private: `__hash__`[`L828`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L828)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](kernel_delta_attention.md#_executor._executor), [`candidate_cfgs_gpu`](kernel_delta_attention.md#KernelDeltaAttention.candidate_cfgs_gpu), [`run`](kernel_delta_attention.md#KernelDeltaAttention.run), [`candidate_cfgs`](kernel_delta_attention.md#KernelDeltaAttention.candidate_cfgs), [`get_impl`](kernel_delta_attention.md#KernelDeltaAttention.get_impl), [`candidate_cfgs_tpu`](kernel_delta_attention.md#KernelDeltaAttention.candidate_cfgs_tpu), [`heuristic_cfg`](kernel_delta_attention.md#KernelDeltaAttention.heuristic_cfg), [`kernel_delta_attention`](kernel_delta_attention.md#kernel_delta_attention), [`KernelDeltaAttention`](kernel_delta_attention.md#KernelDeltaAttention)

### `LightningAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:793`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L793)
- doc: Configuration for Lightning Attention operation.
- signature: `class LightningAttentionConfig(BaseOperationConfig):`
- members:
  - `block_d` — [`L808`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L808)
  - `block_k` — [`L807`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L807)
  - `block_q` — [`L806`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L806)
  - `num_stages` — [`L810`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L810)
  - `num_warps` — [`L809`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L809)
- protocol/private: `__hash__`[`L812`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L812)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](lightning_attention.md#LightningAttention.run), [`candidate_cfgs_gpu`](lightning_attention.md#LightningAttention.candidate_cfgs_gpu), [`_lightning_executor`](lightning_attention.md#_lightning_executor._lightning_executor), [`candidate_cfgs`](lightning_attention.md#LightningAttention.candidate_cfgs), [`candidate_cfgs_tpu`](lightning_attention.md#LightningAttention.candidate_cfgs_tpu), [`heuristic_cfg`](lightning_attention.md#LightningAttention.heuristic_cfg), [`get_impl`](lightning_attention.md#LightningAttention.get_impl), [`lightning_attention`](lightning_attention.md#lightning_attention), [`LightningAttention`](lightning_attention.md#LightningAttention)

### `MeanPoolingConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:516`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L516)
- doc: Configuration for Mean Pooling operation.
- signature: `class MeanPoolingConfig(BaseOperationConfig):`
- members:
  - `block_dim` — [`L529`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L529)
  - `block_size` — [`L528`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L528)
  - `num_stages` — [`L531`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L531)
  - `num_warps` — [`L530`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L530)
- protocol/private: `__hash__`[`L533`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L533)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](pooling.md#MeanPooling.run), [`_mean_pooling_executor`](pooling.md#_mean_pooling_executor._mean_pooling_executor), [`candidate_cfgs_gpu`](pooling.md#MeanPooling.candidate_cfgs_gpu), [`candidate_cfgs`](pooling.md#MeanPooling.candidate_cfgs), [`candidate_cfgs_tpu`](pooling.md#MeanPooling.candidate_cfgs_tpu), [`heuristic_cfg`](pooling.md#MeanPooling.heuristic_cfg), [`get_impl`](pooling.md#MeanPooling.get_impl), [`mean_pooling`](pooling.md#mean_pooling), [`MeanPooling`](pooling.md#MeanPooling)

### `MultiLatentRaggedPageAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:684`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L684)
- doc: Configuration for Multi-Latent Ragged Page Attention.
- signature: `class MultiLatentRaggedPageAttentionConfig(BaseOperationConfig):`
- members:
  - `chunk_prefill_size` — [`L698`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L698)
  - `num_kv_pages_per_block` — [`L699`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L699)
  - `num_queries_per_block` — [`L700`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L700)
  - `num_stages` — [`L703`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L703)
  - `num_warps` — [`L702`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L702)
  - `vmem_limit_bytes` — [`L701`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L701)
- protocol/private: `__hash__`[`L705`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L705)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_gpu`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.candidate_cfgs_gpu), [`run`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.run), [`candidate_cfgs`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.candidate_cfgs), [`candidate_cfgs_tpu`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.candidate_cfgs_tpu), [`heuristic_cfg`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.heuristic_cfg), [`_mlrpa_executor`](multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`get_impl`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention.get_impl), [`_estimate_kv_pages`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention._estimate_kv_pages), [`multi_latent_ragged_page_attention`](multi_latent_ragged_page_attention.md#multi_latent_ragged_page_attention), [`MultiLatentRaggedPageAttention`](multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttention)

### `MultiLatentRaggedPageAttentionV2Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:709`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L709)
- doc: Configuration for Multi-Latent Ragged Page Attention v2.
- signature: `class MultiLatentRaggedPageAttentionV2Config(BaseOperationConfig):`
- members:
  - `__post_init__(self)` — [`L732`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L732) — Normalize list-based JSON payloads back into tuples.
  - `_normalize(value: tuple[int, int, int] | list[int] | int | None, field_name: str)` — [`L735`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L735) — Convert JSON-deserialized lists back to tuples and validate length.
  - `chunk_prefill_size` — [`L725`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L725)
  - `num_kv_pages_per_block` — [`L726`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L726)
  - `num_queries_per_block` — [`L727`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L727)
  - `num_stages` — [`L730`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L730)
  - `num_warps` — [`L729`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L729)
  - `vmem_limit_bytes` — [`L728`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L728)
- protocol/private: `__hash__`[`L766`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L766)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_gpu`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.candidate_cfgs_gpu), [`candidate_cfgs`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.candidate_cfgs), [`candidate_cfgs_tpu`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.candidate_cfgs_tpu), [`heuristic_cfg`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.heuristic_cfg), [`run`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.run), [`_mlrpa_v2_executor`](multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`get_impl`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2.get_impl), [`_estimate_kv_pages`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2._estimate_kv_pages), [`multi_latent_ragged_page_attention_v2`](multi_latent_ragged_page_attention_v2.md#multi_latent_ragged_page_attention_v2), [`MultiLatentRaggedPageAttentionV2`](multi_latent_ragged_page_attention_v2.md#MultiLatentRaggedPageAttentionV2)

### `NativeSparseAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:211`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L211)
- doc: Configuration for Native Sparse Attention operation.
- signature: `class NativeSparseAttentionConfig(BaseOperationConfig):`
- members:
  - `block_d` — [`L227`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L227)
  - `block_k` — [`L226`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L226)
  - `block_q` — [`L225`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L225)
  - `block_size` — [`L228`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L228)
  - `num_stages` — [`L230`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L230)
  - `num_warps` — [`L229`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L229)
- protocol/private: `__hash__`[`L232`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L232)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](native_sparse_attention.md#NativeSparseAttention.run), [`candidate_cfgs_gpu`](native_sparse_attention.md#NativeSparseAttention.candidate_cfgs_gpu), [`candidate_cfgs_xla`](native_sparse_attention.md#NativeSparseAttention.candidate_cfgs_xla), [`_sparse_executor`](native_sparse_attention.md#_sparse_executor._sparse_executor), [`candidate_cfgs`](native_sparse_attention.md#NativeSparseAttention.candidate_cfgs), [`heuristic_cfg`](native_sparse_attention.md#NativeSparseAttention.heuristic_cfg), [`get_impl`](native_sparse_attention.md#NativeSparseAttention.get_impl), [`candidate_cfgs_tpu`](native_sparse_attention.md#NativeSparseAttention.candidate_cfgs_tpu), [`native_sparse_attention`](native_sparse_attention.md#native_sparse_attention), [`NativeSparseAttention`](native_sparse_attention.md#NativeSparseAttention)

### `PageAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:283`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L283)
- doc: Configuration for Page Attention operation.
- signature: `class PageAttentionConfig(BaseOperationConfig):`
- members:
  - `num_splits` — [`L295`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L295)
  - `num_stages` — [`L298`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L298)
  - `num_warps` — [`L297`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L297)
  - `pages_per_compute_block` — [`L296`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L296)
- protocol/private: `__hash__`[`L300`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L300)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](page_attention.md#PageAttention.run), [`_page_attention_executor`](page_attention.md#_page_attention_executor._page_attention_executor), [`candidate_cfgs_gpu`](page_attention.md#PageAttention.candidate_cfgs_gpu), [`candidate_cfgs`](page_attention.md#PageAttention.candidate_cfgs), [`heuristic_cfg`](page_attention.md#PageAttention.heuristic_cfg), [`candidate_cfgs_tpu`](page_attention.md#PageAttention.candidate_cfgs_tpu), [`get_impl`](page_attention.md#PageAttention.get_impl), [`page_attention`](page_attention.md#page_attention), [`PageAttention`](page_attention.md#PageAttention), [`create_shard_map_wrapper`](page_attention.md#PageAttention.create_shard_map_wrapper)

### `PrefillPageAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1149`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1149)
- doc: Configuration for Prefill Page Attention operation.
- signature: `class PrefillPageAttentionConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L1160`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1160)
  - `num_stages` — [`L1162`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1162)
  - `num_warps` — [`L1161`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1161)
- protocol/private: `__hash__`[`L1164`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1164)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_prefill_page_attention_executor`](prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`candidate_cfgs_gpu`](prefill_page_attention.md#PrefillPageAttention.candidate_cfgs_gpu), [`heuristic_cfg`](prefill_page_attention.md#PrefillPageAttention.heuristic_cfg), [`run`](prefill_page_attention.md#PrefillPageAttention.run), [`candidate_cfgs`](prefill_page_attention.md#PrefillPageAttention.candidate_cfgs), [`candidate_cfgs_tpu`](prefill_page_attention.md#PrefillPageAttention.candidate_cfgs_tpu), [`get_impl`](prefill_page_attention.md#PrefillPageAttention.get_impl), [`prefill_page_attention`](prefill_page_attention.md#prefill_page_attention), [`PrefillPageAttention`](prefill_page_attention.md#PrefillPageAttention), [`create_shard_map_wrapper`](prefill_page_attention.md#PrefillPageAttention.create_shard_map_wrapper)

### `QuantizedMatmulConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:485`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L485) — documented in [ejkernel-modules-operations-configs](../../../../concepts/ejkernel-modules-operations-configs.md)
- doc: Configuration for Quantized Matrix Multiplication operation.
- signature: `class QuantizedMatmulConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L505`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L505)
  - `block_m` — [`L503`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L503)
  - `block_n` — [`L504`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L504)
  - `num_stages` — [`L507`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L507)
  - `num_warps` — [`L506`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L506)
  - `split_k` — [`L509`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L509)
  - `tpu_path` — [`L510`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L510)
  - `use_bf16` — [`L508`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L508)
- protocol/private: `__hash__`[`L512`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L512)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_candidate_cfgs_gpu_for_platform`](quantized_matmul.md#QuantizedMatmul._candidate_cfgs_gpu_for_platform), [`_pallas_tpu_candidate_cfgs`](quantized_matmul.md#_pallas_tpu_candidate_cfgs), [`_quantized_matmul_impl`](quantized_matmul.md#_quantized_matmul_impl), [`quantized_matmul`](quantized_matmul.md#quantized_matmul), [`_xla_candidate_cfgs`](quantized_matmul.md#_xla_candidate_cfgs), [`run`](quantized_matmul.md#QuantizedMatmul.run), [`_pallas_tpu_heuristic_cfg`](quantized_matmul.md#_pallas_tpu_heuristic_cfg), [`_triton_heuristic_cfg`](quantized_matmul.md#_triton_heuristic_cfg), [`_cuda_heuristic_cfg`](quantized_matmul.md#_cuda_heuristic_cfg), [`_normalize_pallas_tpu_packed_cfg_forward`](quantized_matmul.md#_normalize_pallas_tpu_packed_cfg_forward), [`_tilelang_heuristic_cfg`](quantized_matmul.md#_tilelang_heuristic_cfg), [`_cute_heuristic_cfg`](quantized_matmul.md#_cute_heuristic_cfg), [`candidate_cfgs_gpu`](quantized_matmul.md#QuantizedMatmul.candidate_cfgs_gpu), [`_quantized_matmul_executor`](quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`heuristic_cfg_gpu`](quantized_matmul.md#QuantizedMatmul.heuristic_cfg_gpu), [`_infer_mkn`](quantized_matmul.md#_infer_mkn), [`candidate_cfgs_tpu`](quantized_matmul.md#QuantizedMatmul.candidate_cfgs_tpu), [`_resolve_inv_platform`](quantized_matmul.md#QuantizedMatmul._resolve_inv_platform), [`_inv_arg`](quantized_matmul.md#_inv_arg), [`get_impl`](quantized_matmul.md#QuantizedMatmul.get_impl), [`heuristic_cfg_tpu`](quantized_matmul.md#QuantizedMatmul.heuristic_cfg_tpu), [`_xla_heuristic_cfg`](quantized_matmul.md#_xla_heuristic_cfg), [`_score`](quantized_matmul.md#_pallas_tpu_candidate_cfgs._score), [`_score`](quantized_matmul.md#_xla_candidate_cfgs._score), [`QuantizedMatmul`](quantized_matmul.md#QuantizedMatmul), [`candidate_cfgs`](quantized_matmul.md#QuantizedMatmul.candidate_cfgs), [`candidate_cfgs_cpu`](quantized_matmul.md#QuantizedMatmul.candidate_cfgs_cpu), [`heuristic_cfg`](quantized_matmul.md#QuantizedMatmul.heuristic_cfg), [`heuristic_cfg_cpu`](quantized_matmul.md#QuantizedMatmul.heuristic_cfg_cpu)

### `RWKV4Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:978`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L978)
- doc: Configuration for RWKV-4 recurrence operation.
- signature: `class RWKV4Config(BaseOperationConfig):`
- members:
  - `block_c` — [`L996`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L996)
- protocol/private: `__hash__`[`L998`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L998)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](rwkv4.md#_executor._executor), [`candidate_cfgs`](rwkv4.md#RWKV4.candidate_cfgs), [`candidate_cfgs_gpu`](rwkv4.md#RWKV4.candidate_cfgs_gpu), [`run`](rwkv4.md#RWKV4.run), [`heuristic_cfg`](rwkv4.md#RWKV4.heuristic_cfg), [`get_impl`](rwkv4.md#RWKV4.get_impl), [`_channels_from_inv`](rwkv4.md#RWKV4._channels_from_inv), [`candidate_cfgs_tpu`](rwkv4.md#RWKV4.candidate_cfgs_tpu), [`rwkv4`](rwkv4.md#rwkv4), [`RWKV4`](rwkv4.md#RWKV4)

### `RWKV6Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1002`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1002)
- doc: Configuration for RWKV-6 recurrence operation.
- signature: `class RWKV6Config(BaseOperationConfig):`
- protocol/private: `__hash__`[`L1023`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1023)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](rwkv6.md#_executor._executor), [`get_impl`](rwkv6.md#RWKV6.get_impl), [`candidate_cfgs_gpu`](rwkv6.md#RWKV6.candidate_cfgs_gpu), [`run`](rwkv6.md#RWKV6.run), [`candidate_cfgs`](rwkv6.md#RWKV6.candidate_cfgs), [`candidate_cfgs_tpu`](rwkv6.md#RWKV6.candidate_cfgs_tpu), [`heuristic_cfg`](rwkv6.md#RWKV6.heuristic_cfg), [`rwkv6`](rwkv6.md#rwkv6), [`RWKV6`](rwkv6.md#RWKV6)

### `RWKV7Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1027`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1027)
- doc: Configuration for RWKV-7 recurrence operation.
- signature: `class RWKV7Config(BaseOperationConfig):`
- members:
  - `block_v` — [`L1041`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1041)
  - `num_stages` — [`L1043`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1043)
  - `num_warps` — [`L1042`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1042)
- protocol/private: `__hash__`[`L1045`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1045)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](rwkv7.md#RWKV7.run), [`_executor_rwkv7`](rwkv7.md#_executor_rwkv7._executor_rwkv7), [`candidate_cfgs_gpu`](rwkv7.md#RWKV7.candidate_cfgs_gpu), [`candidate_cfgs`](rwkv7.md#RWKV7.candidate_cfgs), [`candidate_cfgs_tpu`](rwkv7.md#RWKV7.candidate_cfgs_tpu), [`get_impl`](rwkv7.md#RWKV7.get_impl), [`heuristic_cfg`](rwkv7.md#RWKV7.heuristic_cfg), [`rwkv7`](rwkv7.md#rwkv7), [`RWKV7`](rwkv7.md#RWKV7)

### `RWKV7MulConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1049`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1049)
- doc: Configuration for RWKV-7 multiplicative recurrence operation.
- signature: `class RWKV7MulConfig(BaseOperationConfig):`
- members:
  - `block_v` — [`L1064`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1064)
  - `num_stages` — [`L1066`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1066)
  - `num_warps` — [`L1065`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1065)
- protocol/private: `__hash__`[`L1068`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1068)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](rwkv7.md#RWKV7Mul.run), [`_executor_rwkv7_mul`](rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`candidate_cfgs_gpu`](rwkv7.md#RWKV7Mul.candidate_cfgs_gpu), [`candidate_cfgs`](rwkv7.md#RWKV7Mul.candidate_cfgs), [`candidate_cfgs_tpu`](rwkv7.md#RWKV7Mul.candidate_cfgs_tpu), [`get_impl`](rwkv7.md#RWKV7Mul.get_impl), [`heuristic_cfg`](rwkv7.md#RWKV7Mul.heuristic_cfg), [`rwkv7_mul`](rwkv7.md#rwkv7_mul), [`RWKV7Mul`](rwkv7.md#RWKV7Mul)

### `RaggedCausalConv1DConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:888`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L888)
- doc: Configuration for ragged causal depthwise conv1d.
- signature: `class RaggedCausalConv1DConfig(BaseOperationConfig):`
- members:
  - `apply_silu` — [`L899`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L899)
  - `d_conv` — [`L898`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L898)
- protocol/private: `__hash__`[`L901`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L901)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](ragged_causal_conv1d.md#_executor._executor), [`ragged_causal_conv1d_head_sharded`](ragged_causal_conv1d.md#ragged_causal_conv1d_head_sharded), [`run`](ragged_causal_conv1d.md#RaggedCausalConv1D.run), [`heuristic_cfg`](ragged_causal_conv1d.md#RaggedCausalConv1D.heuristic_cfg), [`ragged_causal_conv1d`](ragged_causal_conv1d.md#ragged_causal_conv1d), [`get_impl`](ragged_causal_conv1d.md#RaggedCausalConv1D.get_impl), [`create_shard_map_wrapper`](ragged_causal_conv1d.md#RaggedCausalConv1D.create_shard_map_wrapper), [`candidate_cfgs`](ragged_causal_conv1d.md#RaggedCausalConv1D.candidate_cfgs), [`RaggedCausalConv1D`](ragged_causal_conv1d.md#RaggedCausalConv1D), [`candidate_cfgs_gpu`](ragged_causal_conv1d.md#RaggedCausalConv1D.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](ragged_causal_conv1d.md#RaggedCausalConv1D.candidate_cfgs_tpu)

### `RaggedDecodeAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:537`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L537)
- doc: Configuration for Ragged Decode Attention operation.
- signature: `class RaggedDecodeAttentionConfig(BaseOperationConfig):`
- members:
  - `__post_init__(self)` — [`L555`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L555) — Convert dict-typed forward params to FwdParams.
  - `fwd_params` — [`L553`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L553)
- protocol/private: `__hash__`[`L560`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L560)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`try_add`](ragged_decode_attention.md#RaggedDecodeAttention.try_add), [`candidate_cfgs`](ragged_decode_attention.md#RaggedDecodeAttention.candidate_cfgs), [`_ragged_decode_attention_executor`](ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`run`](ragged_decode_attention.md#RaggedDecodeAttention.run), [`_default_fwd_params`](ragged_decode_attention.md#RaggedDecodeAttention._default_fwd_params), [`get_impl`](ragged_decode_attention.md#RaggedDecodeAttention.get_impl), [`heuristic_cfg`](ragged_decode_attention.md#RaggedDecodeAttention.heuristic_cfg), [`candidate_cfgs_gpu`](ragged_decode_attention.md#RaggedDecodeAttention.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](ragged_decode_attention.md#RaggedDecodeAttention.candidate_cfgs_tpu), [`ragged_decode_attention`](ragged_decode_attention.md#ragged_decode_attention), [`RaggedDecodeAttention`](ragged_decode_attention.md#RaggedDecodeAttention), [`create_shard_map_wrapper`](ragged_decode_attention.md#RaggedDecodeAttention.create_shard_map_wrapper)

### `RaggedGatedDeltaRuleConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:870`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L870)
- doc: Configuration for Ragged Gated Delta Rule operation.
- signature: `class RaggedGatedDeltaRuleConfig(BaseOperationConfig):`
- members:
  - `chunk_size` — [`L882`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L882)
- protocol/private: `__hash__`[`L884`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L884)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_executor`](ragged_gated_delta_rule.md#_executor._executor), [`create_shard_map_wrapper`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.create_shard_map_wrapper), [`candidate_cfgs_gpu`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.candidate_cfgs_gpu), [`get_impl`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.get_impl), [`ragged_gated_delta_rule`](ragged_gated_delta_rule.md#ragged_gated_delta_rule), [`candidate_cfgs`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.candidate_cfgs), [`candidate_cfgs_tpu`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.candidate_cfgs_tpu), [`run`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.run), [`heuristic_cfg`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule.heuristic_cfg), [`RaggedGatedDeltaRule`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule), [`_wrapped`](ragged_gated_delta_rule.md#RaggedGatedDeltaRule._wrapped)

### `RaggedGatedDeltaRuleV2Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:936`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L936)
- doc: Configuration for the Qwen3-Next packed-inference ragged GDN op.
- signature: `class RaggedGatedDeltaRuleV2Config(BaseOperationConfig):`
- members:
  - `chunk_size` — [`L954`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L954)
  - `kernel_tile_policy` — [`L955`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L955)
  - `use_fused_gdn_decode` — [`L956`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L956)
- protocol/private: `__hash__`[`L958`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L958)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`ragged_gated_delta_rule_v2`](ragged_gated_delta_rule_v2.md#ragged_gated_delta_rule_v2), [`_executor`](ragged_gated_delta_rule_v2.md#_executor._executor), [`run`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.run), [`create_shard_map_wrapper`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.create_shard_map_wrapper), [`candidate_cfgs`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.candidate_cfgs), [`candidate_cfgs_gpu`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.candidate_cfgs_tpu), [`get_impl`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.get_impl), [`heuristic_cfg`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2.heuristic_cfg), [`_run_unsharded`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2._run_unsharded), [`RaggedGatedDeltaRuleV2`](ragged_gated_delta_rule_v2.md#RaggedGatedDeltaRuleV2), [`_is_tpu_pallas_request`](ragged_gated_delta_rule_v2.md#_is_tpu_pallas_request)

### `RaggedPageAttentionv2Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:564`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L564)
- doc: Configuration for Ragged Page Attention operation.
- signature: `class RaggedPageAttentionv2Config(BaseOperationConfig):`
- members:
  - `num_kv_pages_per_block` — [`L576`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L576)
  - `num_queries_per_block` — [`L577`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L577)
  - `num_stages` — [`L579`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L579)
  - `num_warps` — [`L578`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L578)
- protocol/private: `__hash__`[`L581`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L581)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_gpu`](ragged_page_attention_v2.md#RaggedPageAttentionv2.candidate_cfgs_gpu), [`_ragged_page_attention_executor`](ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`candidate_cfgs_tpu`](ragged_page_attention_v2.md#RaggedPageAttentionv2.candidate_cfgs_tpu), [`run`](ragged_page_attention_v2.md#RaggedPageAttentionv2.run), [`_xla_block_candidates_v2`](ragged_page_attention_v2.md#_xla_block_candidates_v2), [`candidate_cfgs`](ragged_page_attention_v2.md#RaggedPageAttentionv2.candidate_cfgs), [`heuristic_cfg`](ragged_page_attention_v2.md#RaggedPageAttentionv2.heuristic_cfg), [`get_impl`](ragged_page_attention_v2.md#RaggedPageAttentionv2.get_impl), [`ragged_page_attention_v2`](ragged_page_attention_v2.md#ragged_page_attention_v2), [`RaggedPageAttentionv2`](ragged_page_attention_v2.md#RaggedPageAttentionv2), [`create_shard_map_wrapper`](ragged_page_attention_v2.md#RaggedPageAttentionv2.create_shard_map_wrapper)

### `RaggedPageAttentionv2TurboQuantConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:585`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L585)
- doc: Configuration for Ragged Page Attention v2 with TurboQuant compression.
- signature: `class RaggedPageAttentionv2TurboQuantConfig(BaseOperationConfig):`
- members:
  - `num_kv_pages_per_block` — [`L611`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L611)
  - `num_queries_per_block` — [`L612`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L612)
  - `num_stages` — [`L614`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L614)
  - `num_warps` — [`L613`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L613)
- protocol/private: `__hash__`[`L616`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L616)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_ragged_page_attention_v2_tq_executor`](ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`run`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.run), [`candidate_cfgs_gpu`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.candidate_cfgs_tpu), [`candidate_cfgs`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.candidate_cfgs), [`heuristic_cfg`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.heuristic_cfg), [`get_impl`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.get_impl), [`ragged_page_attention_v2_turboquant`](ragged_page_attention_v2_turboquant.md#ragged_page_attention_v2_turboquant), [`RaggedPageAttentionv2TurboQuant`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant), [`create_shard_map_wrapper`](ragged_page_attention_v2_turboquant.md#RaggedPageAttentionv2TurboQuant.create_shard_map_wrapper)

### `RaggedPageAttentionv3Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:620`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L620)
- doc: Configuration for Ragged Page Attention operation.
- signature: `class RaggedPageAttentionv3Config(BaseOperationConfig):`
- members:
  - `chunk_prefill_size` — [`L632`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L632)
  - `num_kv_pages_per_block` — [`L633`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L633)
  - `num_queries_per_block` — [`L634`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L634)
  - `num_stages` — [`L636`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L636)
  - `num_warps` — [`L635`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L635)
- protocol/private: `__hash__`[`L638`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L638)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_extract_workload`](ragged_page_attention_v3.md#RaggedPageAttentionv3._extract_workload), [`run`](ragged_page_attention_v3.md#RaggedPageAttentionv3.run), [`_ragged_page_attention_executor`](ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_materialize_configs`](ragged_page_attention_v3.md#RaggedPageAttentionv3._materialize_configs), [`heuristic_cfg`](ragged_page_attention_v3.md#RaggedPageAttentionv3.heuristic_cfg), [`_build_candidate_configs`](ragged_page_attention_v3.md#RaggedPageAttentionv3._build_candidate_configs), [`candidate_cfgs_tpu`](ragged_page_attention_v3.md#RaggedPageAttentionv3.candidate_cfgs_tpu), [`_candidate_pairs`](ragged_page_attention_v3.md#RaggedPageAttentionv3._candidate_pairs), [`get_impl`](ragged_page_attention_v3.md#RaggedPageAttentionv3.get_impl), [`candidate_cfgs_gpu`](ragged_page_attention_v3.md#RaggedPageAttentionv3.candidate_cfgs_gpu), [`ragged_page_attention_v3`](ragged_page_attention_v3.md#ragged_page_attention_v3), [`RaggedPageAttentionv3`](ragged_page_attention_v3.md#RaggedPageAttentionv3), [`candidate_cfgs`](ragged_page_attention_v3.md#RaggedPageAttentionv3.candidate_cfgs), [`create_shard_map_wrapper`](ragged_page_attention_v3.md#RaggedPageAttentionv3.create_shard_map_wrapper)

### `RaggedPageAttentionv3TurboQuantConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:642`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L642)
- doc: Configuration for Ragged Page Attention v3 with TurboQuant compression.
- signature: `class RaggedPageAttentionv3TurboQuantConfig(BaseOperationConfig):`
- members:
  - `chunk_prefill_size` — [`L674`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L674)
  - `num_kv_pages_per_block` — [`L675`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L675)
  - `num_queries_per_block` — [`L676`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L676)
  - `num_stages` — [`L678`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L678)
  - `num_warps` — [`L677`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L677)
- protocol/private: `__hash__`[`L680`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L680)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.run), [`candidate_cfgs_gpu`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.candidate_cfgs_gpu), [`_ragged_page_attention_v3_tq_executor`](ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`candidate_cfgs_tpu`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.candidate_cfgs_tpu), [`candidate_cfgs`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.candidate_cfgs), [`heuristic_cfg`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.heuristic_cfg), [`get_impl`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.get_impl), [`ragged_page_attention_v3_turboquant`](ragged_page_attention_v3_turboquant.md#ragged_page_attention_v3_turboquant), [`RaggedPageAttentionv3TurboQuant`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant), [`create_shard_map_wrapper`](ragged_page_attention_v3_turboquant.md#RaggedPageAttentionv3TurboQuant.create_shard_map_wrapper)

### `RecurrentAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:236`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L236)
- doc: Configuration for Recurrent Attention operation.
- signature: `class RecurrentAttentionConfig(BaseOperationConfig):`
- members:
  - `block_d` — [`L251`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L251)
  - `block_k` — [`L250`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L250)
  - `block_q` — [`L249`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L249)
  - `num_stages` — [`L253`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L253)
  - `num_warps` — [`L252`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L252)
- protocol/private: `__hash__`[`L255`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L255)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](recurrent.md#RecurrentAttention.run), [`candidate_cfgs_gpu`](recurrent.md#RecurrentAttention.candidate_cfgs_gpu), [`_recurrent_executor`](recurrent.md#_recurrent_executor._recurrent_executor), [`candidate_cfgs`](recurrent.md#RecurrentAttention.candidate_cfgs), [`candidate_cfgs_tpu`](recurrent.md#RecurrentAttention.candidate_cfgs_tpu), [`heuristic_cfg`](recurrent.md#RecurrentAttention.heuristic_cfg), [`get_impl`](recurrent.md#RecurrentAttention.get_impl), [`recurrent_attention`](recurrent.md#recurrent_attention), [`RecurrentAttention`](recurrent.md#RecurrentAttention)

### `ReduceScatterMatmulConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:457`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L457)
- doc: Configuration for the fused Matmul + Reduce-Scatter operation.
- signature: `class ReduceScatterMatmulConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L477`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L477)
  - `block_m` — [`L475`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L475)
  - `block_n` — [`L476`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L476)
  - `num_stages` — [`L479`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L479)
  - `num_warps` — [`L478`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L478)
- protocol/private: `__hash__`[`L481`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L481)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](reduce_scatter_matmul.md#ReduceScatterMatmul.run), [`candidate_cfgs_gpu`](reduce_scatter_matmul.md#ReduceScatterMatmul.candidate_cfgs_gpu), [`_reduce_scatter_matmul_executor`](reduce_scatter_matmul.md#_reduce_scatter_matmul_executor._reduce_scatter_matmul_executor), [`candidate_cfgs`](reduce_scatter_matmul.md#ReduceScatterMatmul.candidate_cfgs), [`candidate_cfgs_tpu`](reduce_scatter_matmul.md#ReduceScatterMatmul.candidate_cfgs_tpu), [`heuristic_cfg`](reduce_scatter_matmul.md#ReduceScatterMatmul.heuristic_cfg), [`get_impl`](reduce_scatter_matmul.md#ReduceScatterMatmul.get_impl), [`reduce_scatter_matmul`](reduce_scatter_matmul.md#reduce_scatter_matmul), [`ReduceScatterMatmul`](reduce_scatter_matmul.md#ReduceScatterMatmul), [`create_shard_map_wrapper`](reduce_scatter_matmul.md#ReduceScatterMatmul.create_shard_map_wrapper)

### `RingAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:259`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L259)
- doc: Configuration for Ring Attention operation.
- signature: `class RingAttentionConfig(BaseOperationConfig):`
- members:
  - `__post_init__(self)` — [`L272`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L272) — Convert dict-typed forward/backward params to FwdParams/BwdParams.
  - `bwd_params` — [`L270`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L270)
  - `fwd_params` — [`L269`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L269)
- protocol/private: `__hash__`[`L279`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L279)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`candidate_cfgs_gpu`](ring_attention.md#RingAttention.candidate_cfgs_gpu), [`heuristic_cfg`](ring_attention.md#RingAttention.heuristic_cfg), [`candidate_cfgs`](ring_attention.md#RingAttention.candidate_cfgs), [`candidate_cfgs_tpu`](ring_attention.md#RingAttention.candidate_cfgs_tpu), [`ring_attention`](ring_attention.md#ring_attention), [`_ring_executor`](ring_attention.md#_ring_executor._ring_executor), [`run`](ring_attention.md#RingAttention.run), [`get_impl`](ring_attention.md#RingAttention.get_impl), [`RingAttention`](ring_attention.md#RingAttention), [`create_shard_map_wrapper`](ring_attention.md#RingAttention.create_shard_map_wrapper)

### `ScaledDotProductAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1128`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1128)
- doc: Configuration for Scaled Dot Product Attention operation.
- signature: `class ScaledDotProductAttentionConfig(BaseOperationConfig):`
- members:
  - `block_k` — [`L1141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1141)
  - `block_q` — [`L1140`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1140)
  - `num_stages` — [`L1143`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1143)
  - `num_warps` — [`L1142`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1142)
- protocol/private: `__hash__`[`L1145`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1145)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`run`](scaled_dot_product_attention.md#ScaledDotProductAttention.run), [`_executor`](scaled_dot_product_attention.md#_executor._executor), [`candidate_cfgs_gpu`](scaled_dot_product_attention.md#ScaledDotProductAttention.candidate_cfgs_gpu), [`candidate_cfgs`](scaled_dot_product_attention.md#ScaledDotProductAttention.candidate_cfgs), [`candidate_cfgs_tpu`](scaled_dot_product_attention.md#ScaledDotProductAttention.candidate_cfgs_tpu), [`heuristic_cfg`](scaled_dot_product_attention.md#ScaledDotProductAttention.heuristic_cfg), [`get_impl`](scaled_dot_product_attention.md#ScaledDotProductAttention.get_impl), [`ScaledDotProductAttention`](scaled_dot_product_attention.md#ScaledDotProductAttention), [`create_shard_map_wrapper`](scaled_dot_product_attention.md#ScaledDotProductAttention.create_shard_map_wrapper)

### `StateSpaceV1Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1168`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1168)
- doc: Configuration for SSM1 (Mamba1-style) Selective State Space operation.
- signature: `class StateSpaceV1Config(BaseOperationConfig):`
- members:
  - `block_d` — [`L1181`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1181)
  - `block_e` — [`L1182`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1182)
- protocol/private: `__hash__`[`L1184`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1184)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_state_space_v1_executor`](state_space_v1.md#_state_space_v1_executor._state_space_v1_executor), [`candidate_cfgs_gpu`](state_space_v1.md#StateSpaceV1.candidate_cfgs_gpu), [`candidate_cfgs`](state_space_v1.md#StateSpaceV1.candidate_cfgs), [`run`](state_space_v1.md#StateSpaceV1.run), [`heuristic_cfg`](state_space_v1.md#StateSpaceV1.heuristic_cfg), [`get_impl`](state_space_v1.md#StateSpaceV1.get_impl), [`candidate_cfgs_tpu`](state_space_v1.md#StateSpaceV1.candidate_cfgs_tpu), [`_DN_from_inv`](state_space_v1.md#StateSpaceV1._DN_from_inv), [`state_space_v1`](state_space_v1.md#state_space_v1), [`StateSpaceV1`](state_space_v1.md#StateSpaceV1)

### `StateSpaceV2Config`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:1188`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1188)
- doc: Configuration for SSM2 (Mamba2-style) Selective State Space operation.
- signature: `class StateSpaceV2Config(BaseOperationConfig):`
- members:
  - `block_e` — [`L1204`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1204)
  - `n_groups` — [`L1201`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1201)
  - `rmsnorm_eps` — [`L1203`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1203)
  - `use_gated_rmsnorm` — [`L1202`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1202)
- protocol/private: `__hash__`[`L1206`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L1206)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`_state_space_v2_executor`](state_space_v2.md#_state_space_v2_executor._state_space_v2_executor), [`candidate_cfgs`](state_space_v2.md#StateSpaceV2.candidate_cfgs), [`candidate_cfgs_gpu`](state_space_v2.md#StateSpaceV2.candidate_cfgs_gpu), [`run`](state_space_v2.md#StateSpaceV2.run), [`heuristic_cfg`](state_space_v2.md#StateSpaceV2.heuristic_cfg), [`candidate_cfgs_tpu`](state_space_v2.md#StateSpaceV2.candidate_cfgs_tpu), [`get_impl`](state_space_v2.md#StateSpaceV2.get_impl), [`state_space_v2`](state_space_v2.md#state_space_v2), [`StateSpaceV2`](state_space_v2.md#StateSpaceV2)

### `UnifiedAttentionConfig`  ·  implements/extends BaseOperationConfig
- def: [`ejkernel/modules/operations/configs.py:304`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L304)
- doc: Configuration for vLLM-style unified (paged) attention operation.
- signature: `class UnifiedAttentionConfig(BaseOperationConfig):`
- members:
  - `block_dim` — [`L321`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L321)
  - `num_par_softmax_segments` — [`L320`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L320)
  - `num_stages` — [`L323`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L323)
  - `num_warps` — [`L322`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L322)
  - `seq_threshold_3d` — [`L319`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L319)
- protocol/private: `__hash__`[`L325`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L325)
- uses (calls/refs, reference-scoped): [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`hash_fn`](configs.md#hash_fn)
- used by: [`BaseOperationConfig`](configs.md#BaseOperationConfig), [`heuristic_cfg`](unified_attention.md#UnifiedAttention.heuristic_cfg), [`run`](unified_attention.md#UnifiedAttention.run), [`candidate_cfgs_gpu`](unified_attention.md#UnifiedAttention.candidate_cfgs_gpu), [`_unified_attention_executor`](unified_attention.md#_unified_attention_executor._unified_attention_executor), [`candidate_cfgs_tpu`](unified_attention.md#UnifiedAttention.candidate_cfgs_tpu), [`get_impl`](unified_attention.md#UnifiedAttention.get_impl), [`unified_attention`](unified_attention.md#unified_attention), [`UnifiedAttention`](unified_attention.md#UnifiedAttention), [`candidate_cfgs`](unified_attention.md#UnifiedAttention.candidate_cfgs), [`create_shard_map_wrapper`](unified_attention.md#UnifiedAttention.create_shard_map_wrapper)

## Functions
- `get_safe_hash_int(text, algorithm="md5")` — [`L67`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L67) — Generate a hash of text using specified algorithm with safety checks.
- `hash_fn(self)` — [`L91`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/configs.py#L91) — Generate a hash for an object based on its dictionary values.

