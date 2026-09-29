---
title: 'Module: ejkernel/modules/operations/attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.attention`/
symbols:
  Attention.run: Attention#run().
  Attention.heuristic_cfg: Attention#heuristic_cfg().
  Attention.candidate_cfgs_gpu: Attention#candidate_cfgs_gpu().
  Attention.candidate_cfgs: Attention#candidate_cfgs().
  _executor._executor: _executor._executor.
  Attention.candidate_cfgs_tpu: Attention#candidate_cfgs_tpu().
  Attention.get_impl: Attention#get_impl().
  attention: attention().
  Attention._seqlens_from_inv: Attention#_seqlens_from_inv().
  Attention: Attention#
  Attention._heuristic_weights_block: Attention#_heuristic_weights_block().
  Attention.__init__: Attention#__init__().
---
# Module: [`ejkernel/modules/operations/attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py)

## Classes
### `Attention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/attention.py:99`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L99)
- doc: Attention with custom optimization logic.
- signature: `class Attention(Kernel[AttentionConfig, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L133) — Initialize  Attention module.
  - `_heuristic_weights_block(seq_len: int)` — [`L274`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L274) — Operation-side dense-weights tile heuristic — single source of truth.
  - `_seqlens_from_inv(inv: Invocation[AttentionConfig, Array])` — [`L265`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L265) — Pull ``(seq_len_q, seq_len_k)`` from the invocation's q/k tensors.
  - `candidate_cfgs(self, inv: Invocation[AttentionConfig, Array])` — [`L295`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L295) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[AttentionConfig, Array])` — [`L311`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L311) — Generate GPU candidates for dense attention with weights.
  - `candidate_cfgs_tpu(self, inv: Invocation[AttentionConfig, Array])` — [`L372`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L372) — Return TPU candidates for the portable XLA dense-attention path.
  - `get_impl(self, cfg: AttentionConfig)` — [`L137`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L137) — Get kernel implementation from registry based on configuration.
  - `heuristic_cfg(self, inv: Invocation[AttentionConfig, Array])` — [`L281`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L281) — Cold-start configuration with shape-aware ``weights_block_q/k``.
  - `run(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads vhead_dim"], attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, softmax_aux: Float[Array, num_sinks] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dtype: DTypeLike | None = jnp.bfloat16, softmax_dtype: DTypeLike | None = None, dropout_prob: float = 0, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, *, cfg: AttentionConfig)` — [`L155`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L155) — Execute standard multi-head attention with the given configuration.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`kv_blocksize`](../../ops/utils/datacarrier.md#FwdParams.kv_blocksize), [`kv_blocksize`](../../ops/utils/datacarrier.md#BwdParams.kv_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#BwdParams.q_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#FwdParams.q_blocksize), [`num_stages`](../../ops/utils/datacarrier.md#FwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#FwdParams.num_warps), [`num_stages`](../../ops/utils/datacarrier.md#BwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#BwdParams.num_warps), [`AttentionConfig`](configs.md#AttentionConfig), [`TILELANG`](../../kernels/_registry.md#Platform.TILELANG), [`block_k`](configs.md#AttentionConfig.block_k), [`block_q`](configs.md#AttentionConfig.block_q), [`num_stages`](configs.md#AttentionConfig.num_stages), [`num_warps`](configs.md#AttentionConfig.num_warps), [`weights_block_k`](configs.md#AttentionConfig.weights_block_k), [`weights_block_q`](configs.md#AttentionConfig.weights_block_q)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`attention`](attention.md#attention)

## Functions
- `attention(query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads vhead_dim"], bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, dropout_rng: PRNGKeyArray | None = None, softmax_aux: Float[Array, num_sinks] | None = None,, *, mask_info: MaskInfo | None = None, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, deterministic: bool = True, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dtype: DTypeLike | None = jnp.bfloat16, softmax_dtype: DTypeLike | None = None, dropout_prob: float = 0, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L398`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L398) — Execute standard multi-head attention with automatic optimization.

## Module values
- `_executor` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/attention.py#L388)

