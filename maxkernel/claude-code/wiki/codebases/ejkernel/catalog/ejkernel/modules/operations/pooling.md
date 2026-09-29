---
title: 'Module: ejkernel/modules/operations/pooling.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/pooling.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.pooling`/
symbols:
  MeanPooling.run: MeanPooling#run().
  _mean_pooling_executor._mean_pooling_executor: _mean_pooling_executor._mean_pooling_executor.
  MeanPooling.candidate_cfgs_gpu: MeanPooling#candidate_cfgs_gpu().
  MeanPooling.candidate_cfgs: MeanPooling#candidate_cfgs().
  MeanPooling.heuristic_cfg: MeanPooling#heuristic_cfg().
  MeanPooling.candidate_cfgs_tpu: MeanPooling#candidate_cfgs_tpu().
  MeanPooling.get_impl: MeanPooling#get_impl().
  mean_pooling: mean_pooling().
  MeanPooling: MeanPooling#
  MeanPooling.__init__: MeanPooling#__init__().
---
# Module: [`ejkernel/modules/operations/pooling.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py)

## Classes
### `MeanPooling`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/pooling.py:84`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L84)
- doc: Mean Pooling with custom optimization logic.
- signature: `class MeanPooling(Kernel[MeanPoolingConfig, Array]):`
- members:
  - `__init__(self)` — [`L101`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L101) — Initialize Mean Pooling module.
  - `candidate_cfgs(self, inv: Invocation[MeanPoolingConfig, Array])` — [`L201`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L201) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[MeanPoolingConfig, Array])` — [`L240`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L240) — Generate GPU candidates for mean pooling across registered GPU backends.
  - `candidate_cfgs_tpu(self, inv: Invocation[MeanPoolingConfig, Array])` — [`L313`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L313) — Generate TPU candidates for the XLA mean-pooling path.
  - `get_impl(self, cfg: MeanPoolingConfig)` — [`L109`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L109) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[MeanPoolingConfig, Array])` — [`L179`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L179) — Provide default configuration with block sizes.
  - `run(self, x: Float[Array, "... hidden_dim"], chunk_size: int = 32, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: MeanPoolingConfig)` — [`L124`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L124) — Execute mean pooling over sequence dimension.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`MeanPoolingConfig`](configs.md#MeanPoolingConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`XLA`](../../kernels/_registry.md#Platform.XLA), [`TILELANG`](../../kernels/_registry.md#Platform.TILELANG), [`block_dim`](configs.md#MeanPoolingConfig.block_dim), [`block_size`](configs.md#MeanPoolingConfig.block_size), [`num_stages`](configs.md#MeanPoolingConfig.num_stages), [`num_warps`](configs.md#MeanPoolingConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`mean_pooling`](pooling.md#mean_pooling)

## Functions
- `mean_pooling(x: Float[Array, "... hidden_dim"], cu_seqlens: Int[Array, num_seqs_plus_one] | None = None,, *, chunk_size: int = 32, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: MeanPoolingConfig | None = None)` — [`L341`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L341) — Execute mean pooling with automatic optimization.

## Module values
- `_mean_pooling_executor` — [`L327`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/pooling.py#L327)

