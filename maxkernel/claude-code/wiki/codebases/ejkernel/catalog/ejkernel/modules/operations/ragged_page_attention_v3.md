---
title: 'Module: ejkernel/modules/operations/ragged_page_attention_v3.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ragged_page_attention_v3.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ragged_page_attention_v3`/
symbols:
  RaggedPageAttentionv3._extract_workload: RaggedPageAttentionv3#_extract_workload().
  _lookup_tuned_pair: _lookup_tuned_pair().
  RaggedPageAttentionv3.run: RaggedPageAttentionv3#run().
  _ragged_page_attention_executor._ragged_page_attention_executor: _ragged_page_attention_executor._ragged_page_attention_executor.
  RaggedPageAttentionv3._materialize_configs: RaggedPageAttentionv3#_materialize_configs().
  RaggedPageAttentionv3.heuristic_cfg: RaggedPageAttentionv3#heuristic_cfg().
  RaggedPageAttentionv3._build_candidate_configs: RaggedPageAttentionv3#_build_candidate_configs().
  RaggedPageAttentionv3.candidate_cfgs_tpu: RaggedPageAttentionv3#candidate_cfgs_tpu().
  RaggedPageAttentionv3._candidate_pairs: RaggedPageAttentionv3#_candidate_pairs().
  RaggedPageAttentionv3.get_impl: RaggedPageAttentionv3#get_impl().
  RaggedPageAttentionv3._generate_candidate_pairs: RaggedPageAttentionv3#_generate_candidate_pairs().
  _resolve_inv_arg: _resolve_inv_arg().
  _suggest_block_sizes: _suggest_block_sizes().
  RaggedPageAttentionv3.candidate_cfgs_gpu: RaggedPageAttentionv3#candidate_cfgs_gpu().
  _tpu_kv_candidates: _tpu_kv_candidates().
  ragged_page_attention_v3: ragged_page_attention_v3().
  _RPAWorkload.head_dim: _RPAWorkload#head_dim.
  RaggedPageAttentionv3: RaggedPageAttentionv3#
  _expand_axis_candidates._push: _expand_axis_candidates()._push().
  RaggedPageAttentionv3.candidate_cfgs: RaggedPageAttentionv3#candidate_cfgs().
  _RPAWorkload.pages_per_seq: _RPAWorkload#pages_per_seq.
  _RPAWorkload: _RPAWorkload#
  _RPAWorkload.max_num_tokens: _RPAWorkload#max_num_tokens.
  _is_tpu: _is_tpu().
  RaggedPageAttentionv3.create_shard_map_wrapper: RaggedPageAttentionv3#create_shard_map_wrapper().
  _expand_axis_candidates: _expand_axis_candidates().
  _RPAWorkload.page_size: _RPAWorkload#page_size.
  _ARGUMENT_INDEX: _ARGUMENT_INDEX.
  _expand_axis_candidates._quantize: _expand_axis_candidates()._quantize().
  RaggedPageAttentionv3._wrapped_ragged_page_attn: RaggedPageAttentionv3#_wrapped_ragged_page_attn().
  _RPAWorkload.q_dtype: _RPAWorkload#q_dtype.
  _RPAWorkload.kv_dtype: _RPAWorkload#kv_dtype.
  _RPAWorkload.num_q_heads: _RPAWorkload#num_q_heads.
  _RPAWorkload.num_kv_heads: _RPAWorkload#num_kv_heads.
  RaggedPageAttentionv3.__init__: RaggedPageAttentionv3#__init__().
  RaggedPageAttentionv3.candidate_cfgs_shard_map_tpu: RaggedPageAttentionv3#candidate_cfgs_shard_map_tpu.
  RaggedPageAttentionv3.candidate_cfgs_shard_map_gpu: RaggedPageAttentionv3#candidate_cfgs_shard_map_gpu.
  _ARGUMENT_ORDER: _ARGUMENT_ORDER.
  _round_up_to_step: _round_up_to_step().
  RaggedPageAttentionv3.version: RaggedPageAttentionv3#version.
---
# Module: [`ejkernel/modules/operations/ragged_page_attention_v3.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py)

## Classes
### `RaggedPageAttentionv3`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ragged_page_attention_v3.py:352`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L352)
- doc: Ragged Page Attention v3 with fused KV-cache write.
- signature: `class RaggedPageAttentionv3(Kernel[RaggedPageAttentionv3Config, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L384`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L384) — Initialize Ragged Page Attention module.
  - `_build_candidate_configs(self, inv: Invocation[RaggedPageAttentionv3Config, Array], *, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"], backend: Backend | Literal["any"], num_warps: int | None, num_stages: int | None, prefer_tuned: bool, max_candidates: int)` — [`L892`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L892) — Build candidate configurations for autotuning from an invocation.
  - `_candidate_pairs(self, inv: Invocation[RaggedPageAttentionv3Config, Array], *, prefer_tuned: bool, max_candidates: int)` — [`L751`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L751) — Generate candidate (kv_pages, query_tokens) block size pairs.
  - `_extract_workload(self, inv: Invocation[RaggedPageAttentionv3Config, Array])` — [`L702`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L702) — Extract workload characteristics from an invocation.
  - `_generate_candidate_pairs(self, workload: _RPAWorkload, base_pair: tuple[int, int], *, max_candidates: int)` — [`L783`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L783) — Generate candidate block size pairs from a base configuration.
  - `_materialize_configs(self, pairs: list[tuple[int, int]], *, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"], backend: Backend | Literal["any"], num_warps: int | None, num_stages: int | None)` — [`L842`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L842) — Convert block size pairs into configuration objects.
  - `_wrapped_ragged_page_attn(queries: Float[Array, "total_tokens num_q_heads head_dim"], keys: Float[Array, "total_tokens num_kv_heads head_dim"], values: Float[Array, "total_tokens num_kv_heads head_dim"], kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], softmax_aux: Float[Array, num_q_heads] | None)` — [`L482`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L482) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `candidate_cfgs(self, inv: Invocation[RaggedPageAttentionv3Config, Array])` — [`L950`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L950) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RaggedPageAttentionv3Config, Array])` — [`L978`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L978) — Generate GPU candidates for every registered GPU backend.
  - `candidate_cfgs_tpu(self, inv: Invocation[RaggedPageAttentionv3Config, Array])` — [`L1046`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L1046) — Generate candidate configurations for autotuning on TPU (Pallas backend).
  - `create_shard_map_wrapper(self, queries: Float[Array, "total_tokens num_q_heads head_dim"], keys: Float[Array, "total_tokens num_kv_heads head_dim"], values: Float[Array, "total_tokens num_kv_heads head_dim"], kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], softmax_aux: Float[Array, num_q_heads] | None = None, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, vmem_limit_bytes: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedPageAttentionv3Config | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | tuple[PartitionSpec, PartitionSpec] | None = None, check_vma: bool = False)` — [`L392`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L392) — Create a shard_map wrapper for distributed ragged page attention.
  - `get_impl(self, cfg: RaggedPageAttentionv3Config)` — [`L540`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L540) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RaggedPageAttentionv3Config, Array])` — [`L928`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L928) — Provide default configuration optimized for ragged page attention.
  - `run(self, queries: Float[Array, "total_tokens num_q_heads head_dim"], keys: Float[Array, "total_tokens num_kv_heads head_dim"], values: Float[Array, "total_tokens num_kv_heads head_dim"], kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], softmax_aux: Float[Array, num_q_heads] | None = None, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, vmem_limit_bytes: int | None = None, *, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedPageAttentionv3Config)` — [`L555`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L555) — Execute ragged page attention over variable-length sequences.
  - `candidate_cfgs_shard_map_gpu` — [`L1113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L1113)
  - `candidate_cfgs_shard_map_tpu` — [`L1112`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L1112)
  - `version` — [`L382`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L382)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`_lookup_tuned_pair`](ragged_page_attention_v3.md#_lookup_tuned_pair), [`RaggedPageAttentionv3Config`](configs.md#RaggedPageAttentionv3Config), [`_resolve_inv_arg`](ragged_page_attention_v3.md#_resolve_inv_arg), [`_suggest_block_sizes`](ragged_page_attention_v3.md#_suggest_block_sizes), [`_tpu_kv_candidates`](ragged_page_attention_v3.md#_tpu_kv_candidates), [`head_dim`](ragged_page_attention_v3.md#_RPAWorkload.head_dim), [`pages_per_seq`](ragged_page_attention_v3.md#_RPAWorkload.pages_per_seq), [`_RPAWorkload`](ragged_page_attention_v3.md#_RPAWorkload), [`max_num_tokens`](ragged_page_attention_v3.md#_RPAWorkload.max_num_tokens), [`num_kv_pages_per_block`](configs.md#RaggedPageAttentionv3Config.num_kv_pages_per_block), [`num_queries_per_block`](configs.md#RaggedPageAttentionv3Config.num_queries_per_block), [`_is_tpu`](ragged_page_attention_v3.md#_is_tpu), [`num_stages`](configs.md#RaggedPageAttentionv3Config.num_stages), [`num_warps`](configs.md#RaggedPageAttentionv3Config.num_warps), [`_expand_axis_candidates`](ragged_page_attention_v3.md#_expand_axis_candidates), [`page_size`](ragged_page_attention_v3.md#_RPAWorkload.page_size), [`kv_dtype`](ragged_page_attention_v3.md#_RPAWorkload.kv_dtype), [`num_kv_heads`](ragged_page_attention_v3.md#_RPAWorkload.num_kv_heads), [`num_q_heads`](ragged_page_attention_v3.md#_RPAWorkload.num_q_heads), [`q_dtype`](ragged_page_attention_v3.md#_RPAWorkload.q_dtype), [`chunk_prefill_size`](configs.md#RaggedPageAttentionv3Config.chunk_prefill_size)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ragged_page_attention_v3`](ragged_page_attention_v3.md#ragged_page_attention_v3)

### `_RPAWorkload`
- def: [`ejkernel/modules/operations/ragged_page_attention_v3.py:95`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L95)
- doc: Internal dataclass representing workload characteristics for ragged page attention.
- signature: `class _RPAWorkload:`
- members:
  - `head_dim` — [`L116`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L116)
  - `kv_dtype` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L113)
  - `max_num_tokens` — [`L118`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L118)
  - `num_kv_heads` — [`L115`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L115)
  - `num_q_heads` — [`L114`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L114)
  - `page_size` — [`L117`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L117)
  - `pages_per_seq` — [`L119`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L119)
  - `q_dtype` — [`L112`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L112)
- used by: [`_extract_workload`](ragged_page_attention_v3.md#RaggedPageAttentionv3._extract_workload), [`_lookup_tuned_pair`](ragged_page_attention_v3.md#_lookup_tuned_pair), [`candidate_cfgs_tpu`](ragged_page_attention_v3.md#RaggedPageAttentionv3.candidate_cfgs_tpu), [`_generate_candidate_pairs`](ragged_page_attention_v3.md#RaggedPageAttentionv3._generate_candidate_pairs), [`_suggest_block_sizes`](ragged_page_attention_v3.md#_suggest_block_sizes), [`_tpu_kv_candidates`](ragged_page_attention_v3.md#_tpu_kv_candidates)

## Functions
- `_expand_axis_candidates(base: int | None, limit: int, *, seeds: tuple[int, ...], min_value: int = 1, step: int | None = None)` — [`L239`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L239) — Expand a base value into a list of candidate values for autotuning.
- `_is_tpu()` — [`L295`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L295) — Check if the current JAX backend is TPU.
- `_lookup_tuned_pair(workload: _RPAWorkload)` — [`L307`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L307) — Look up pre-tuned block sizes for the given workload.
- `_push(val: float | int | None)` — [`L276`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L276) — Quantize and append a candidate value if not already present.
- `_quantize(val: float | int)` — [`L266`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L266) — Round and clamp a value to the nearest valid block size within limits.
- `_resolve_inv_arg(inv: Invocation, name: str)` — [`L136`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L136) — Resolve an argument from an Invocation by name.
- `_round_up_to_step(value: int, step: int)` — [`L160`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L160) — Round a value up to the nearest multiple of step.
- `_suggest_block_sizes(workload: _RPAWorkload, aggressive: bool)` — [`L175`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L175) — Suggest initial block sizes based on workload characteristics.
- `_tpu_kv_candidates(workload: _RPAWorkload)` — [`L220`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L220) — Return TPU-safe KV-page candidates for the workload.
- `ragged_page_attention_v3(queries: Float[Array, "total_tokens num_q_heads head_dim"], keys: Float[Array, "total_tokens num_kv_heads head_dim"], values: Float[Array, "total_tokens num_kv_heads head_dim"], kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], softmax_aux: Float[Array, num_q_heads] | None = None,, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, vmem_limit_bytes: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedPageAttentionv3Config | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | tuple[PartitionSpec, PartitionSpec] | None = None)` — [`L1130`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L1130) — Execute ragged page attention v3 with automatic optimization and optional sharding.

## Module values
- `_ARGUMENT_INDEX` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L133)
- `_ARGUMENT_ORDER` — [`L122`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L122)
- `_ragged_page_attention_executor` — [`L1116`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v3.py#L1116)

