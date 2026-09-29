---
title: 'Module: ejkernel/modules/operations/kernel_delta_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/kernel_delta_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.kernel_delta_attention`/
symbols:
  _executor._executor: _executor._executor.
  KernelDeltaAttention.run: KernelDeltaAttention#run().
  KernelDeltaAttention.candidate_cfgs_gpu: KernelDeltaAttention#candidate_cfgs_gpu().
  KernelDeltaAttention.get_impl: KernelDeltaAttention#get_impl().
  KernelDeltaAttention.candidate_cfgs: KernelDeltaAttention#candidate_cfgs().
  KernelDeltaAttention.heuristic_cfg: KernelDeltaAttention#heuristic_cfg().
  KernelDeltaAttention.candidate_cfgs_tpu: KernelDeltaAttention#candidate_cfgs_tpu().
  kernel_delta_attention: kernel_delta_attention().
  KernelDeltaAttention: KernelDeltaAttention#
  kda_attention: kda_attention.
  KernelDeltaAttention.__init__: KernelDeltaAttention#__init__().
---
# Module: [`ejkernel/modules/operations/kernel_delta_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py)

## Classes
### `KernelDeltaAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/kernel_delta_attention.py:90`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L90)
- doc: Kernel Delta Attention (KDA) operation.
- signature: `class KernelDeltaAttention(Kernel[KernelDeltaAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L105`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L105) — Initialize KernelDeltaAttention operation.
  - `candidate_cfgs(self, inv: Invocation[KernelDeltaAttentionConfig, Array])` — [`L223`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L223) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[KernelDeltaAttentionConfig, Array])` — [`L237`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L237) — Generate GPU candidates for KDA.
  - `candidate_cfgs_tpu(self, inv: Invocation[KernelDeltaAttentionConfig, Array])` — [`L255`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L255) — Generate TPU candidates for the XLA chunked KDA implementation.
  - `get_impl(self, cfg: KernelDeltaAttentionConfig)` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L113) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[KernelDeltaAttentionConfig, Array])` — [`L208`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L208) — Provide default configuration based on heuristics.
  - `run(self, query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_heads qk_head_dim"], value: Float[Array, "batch seq_len num_heads v_head_dim"], beta: Float[Array, "batch seq_len num_heads"], decay: Float[Array, "batch seq_len num_heads"] | None = None, initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None, *, softmax_scale: float | None = None, chunk_size: int = 64, use_qk_l2norm: bool = True, use_chunked: bool = True, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: KernelDeltaAttentionConfig)` — [`L128`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L128) — Execute Kernel Delta Attention operation.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`KernelDeltaAttentionConfig`](configs.md#KernelDeltaAttentionConfig), [`chunk_size`](configs.md#KernelDeltaAttentionConfig.chunk_size)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`kernel_delta_attention`](kernel_delta_attention.md#kernel_delta_attention)

## Functions
- `kernel_delta_attention(query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_heads qk_head_dim"], value: Float[Array, "batch seq_len num_heads v_head_dim"], beta: Float[Array, "batch seq_len num_heads"], decay: Float[Array, "batch seq_len num_heads"] | None = None, initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,, *, softmax_scale: float | None = None, chunk_size: int = 64, use_qk_l2norm: bool = True, use_chunked: bool = True, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: KernelDeltaAttentionConfig | None = None)` — [`L272`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L272) — Execute Kernel Delta Attention (KDA) with automatic optimization.

## Module values
- `_executor` — [`L262`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L262)
- `kda_attention` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/kernel_delta_attention.py#L388)

