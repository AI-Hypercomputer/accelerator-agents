---
title: 'Module: ejkernel/modules/operations/scaled_dot_product_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/scaled_dot_product_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.scaled_dot_product_attention`/
symbols:
  ScaledDotProductAttention.run: ScaledDotProductAttention#run().
  _executor._executor: _executor._executor.
  ScaledDotProductAttention.candidate_cfgs_gpu: ScaledDotProductAttention#candidate_cfgs_gpu().
  ScaledDotProductAttention.candidate_cfgs: ScaledDotProductAttention#candidate_cfgs().
  ScaledDotProductAttention.heuristic_cfg: ScaledDotProductAttention#heuristic_cfg().
  ScaledDotProductAttention.candidate_cfgs_tpu: ScaledDotProductAttention#candidate_cfgs_tpu().
  ScaledDotProductAttention.get_impl: ScaledDotProductAttention#get_impl().
  scaled_dot_product_attention: scaled_dot_product_attention().
  ScaledDotProductAttention: ScaledDotProductAttention#
  ScaledDotProductAttention.create_shard_map_wrapper: ScaledDotProductAttention#create_shard_map_wrapper().
  ScaledDotProductAttention.__init__: ScaledDotProductAttention#__init__().
  ScaledDotProductAttention._wrapped_sdpa: ScaledDotProductAttention#_wrapped_sdpa().
---
# Module: [`ejkernel/modules/operations/scaled_dot_product_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py)

## Classes
### `ScaledDotProductAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/scaled_dot_product_attention.py:102`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L102)
- doc: Standard scaled dot-product attention kernel wrapper.
- signature: `class ScaledDotProductAttention(Kernel[ScaledDotProductAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L141) — Initialize ScaledDotProductAttention module.
  - `_wrapped_sdpa(query, key, value, bias, cum_seqlens_q, cum_seqlens_k, attention_mask)` — [`L316`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L316) — Shard-map compatible wrapper that delegates to impl with captured params.
  - `candidate_cfgs(self, inv: Invocation[ScaledDotProductAttentionConfig, Array])` — [`L379`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L379) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[ScaledDotProductAttentionConfig, Array])` — [`L415`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L415) — Generate GPU candidates for SDPA.
  - `candidate_cfgs_tpu(self, inv: Invocation[ScaledDotProductAttentionConfig, Array])` — [`L487`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L487) — Generate TPU candidates for XLA SDPA.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, cum_seqlens_q: Int[Array, batch] | None = None, cum_seqlens_k: Int[Array, batch] | None = None, *, mesh: jax.sharding.Mesh, in_specs: tuple[jax.sharding.PartitionSpec, ...], out_specs: jax.sharding.PartitionSpec, check_vma: bool = False, cfg: ScaledDotProductAttentionConfig, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, softmax_scale: float | None = None, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None)` — [`L259`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L259) — Create a shard_map wrapper for distributed ScaledDotProductAttention execution.
  - `get_impl(self, cfg: ScaledDotProductAttentionConfig)` — [`L149`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L149) — Get kernel implementation from registry based on configuration.
  - `heuristic_cfg(self, inv: Invocation[ScaledDotProductAttentionConfig, Array])` — [`L360`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L360) — Provide default configuration based on invocation context.
  - `run(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, softmax_scale: float | None = None, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, cum_seqlens_q: Int[Array, batch] | None = None, cum_seqlens_k: Int[Array, batch] | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: ScaledDotProductAttentionConfig)` — [`L167`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L167) — Execute scaled dot-product attention with the given configuration.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`kv_blocksize`](../../ops/utils/datacarrier.md#FwdParams.kv_blocksize), [`kv_blocksize`](../../ops/utils/datacarrier.md#BwdParams.kv_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#BwdParams.q_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#FwdParams.q_blocksize), [`num_stages`](../../ops/utils/datacarrier.md#FwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#FwdParams.num_warps), [`ScaledDotProductAttentionConfig`](configs.md#ScaledDotProductAttentionConfig), [`num_stages`](../../ops/utils/datacarrier.md#BwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#BwdParams.num_warps), [`block_k`](configs.md#ScaledDotProductAttentionConfig.block_k), [`block_q`](configs.md#ScaledDotProductAttentionConfig.block_q), [`num_stages`](configs.md#ScaledDotProductAttentionConfig.num_stages), [`num_warps`](configs.md#ScaledDotProductAttentionConfig.num_warps), [`TILELANG`](../../kernels/_registry.md#Platform.TILELANG)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`scaled_dot_product_attention`](scaled_dot_product_attention.md#scaled_dot_product_attention)

## Functions
- `scaled_dot_product_attention(query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, cum_seqlens_q: Int[Array, batch] | None = None, cum_seqlens_k: Int[Array, batch] | None = None,, *, mask_info: MaskInfo | None = None, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, softmax_scale: float | None = None, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, mesh: jax.sharding.Mesh | None = None, in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None, out_specs: jax.sharding.PartitionSpec | None = None)` — [`L511`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L511) — Execute scaled dot-product attention with automatic optimization.

## Module values
- `_executor` — [`L501`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/scaled_dot_product_attention.py#L501)

