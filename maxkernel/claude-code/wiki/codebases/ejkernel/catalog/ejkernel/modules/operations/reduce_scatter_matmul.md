---
title: 'Module: ejkernel/modules/operations/reduce_scatter_matmul.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/reduce_scatter_matmul.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.reduce_scatter_matmul`/
symbols:
  ReduceScatterMatmul.run: ReduceScatterMatmul#run().
  ReduceScatterMatmul.candidate_cfgs_gpu: ReduceScatterMatmul#candidate_cfgs_gpu().
  _reduce_scatter_matmul_executor._reduce_scatter_matmul_executor: _reduce_scatter_matmul_executor._reduce_scatter_matmul_executor.
  ReduceScatterMatmul.candidate_cfgs: ReduceScatterMatmul#candidate_cfgs().
  ReduceScatterMatmul.heuristic_cfg: ReduceScatterMatmul#heuristic_cfg().
  ReduceScatterMatmul.candidate_cfgs_tpu: ReduceScatterMatmul#candidate_cfgs_tpu().
  ReduceScatterMatmul.get_impl: ReduceScatterMatmul#get_impl().
  reduce_scatter_matmul: reduce_scatter_matmul().
  ReduceScatterMatmul: ReduceScatterMatmul#
  ReduceScatterMatmul.create_shard_map_wrapper: ReduceScatterMatmul#create_shard_map_wrapper().
  _resolve_tp_size: _resolve_tp_size().
  ReduceScatterMatmul._wrapped: ReduceScatterMatmul#_wrapped().
  _largest_divisor_leq: _largest_divisor_leq().
  ReduceScatterMatmul.__init__: ReduceScatterMatmul#__init__().
  _infer_axis_size: _infer_axis_size().
---
# Module: [`ejkernel/modules/operations/reduce_scatter_matmul.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py)

## Classes
### `ReduceScatterMatmul`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/reduce_scatter_matmul.py:154`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L154)
- doc: Fused distributed reduce-scatter matmul kernel.
- signature: `class ReduceScatterMatmul(Kernel[ReduceScatterMatmulConfig, Array]):`
- members:
  - `candidate_cfgs(self, inv: Invocation[ReduceScatterMatmulConfig, Array])` — [`L333`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L333) — Return candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[ReduceScatterMatmulConfig, Array])` — [`L350`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L350) — Return GPU candidates for TileLang and XLA reduce-scatter matmul.
  - `candidate_cfgs_tpu(self, inv: Invocation[ReduceScatterMatmulConfig, Array])` — [`L405`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L405) — Return TPU candidates for Pallas and XLA reduce-scatter matmul paths.
  - `create_shard_map_wrapper(self, x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"], axis_name: str, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, tp_size: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: ReduceScatterMatmulConfig, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L190`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L190) — Build a shard_map-wrapped callable and its input arguments.
  - `get_impl(self, cfg: ReduceScatterMatmulConfig)` — [`L175`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L175) — Get the kernel implementation for the given configuration.
  - `heuristic_cfg(self, inv: Invocation[ReduceScatterMatmulConfig, Array])` — [`L321`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L321) — Return default heuristic configuration for any platform.
  - `run(self, x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"], axis_name: str, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, tp_size: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: ReduceScatterMatmulConfig)` — [`L255`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L255) — Execute reduce-scatter matmul with the selected backend.
- protocol/private: `__init__`[`L172`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L172), `_wrapped`[`L229`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L229)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`op_id`](../../ops/core/kernel.md#Kernel.op_id), [`ReduceScatterMatmulConfig`](configs.md#ReduceScatterMatmulConfig), [`block_k`](configs.md#ReduceScatterMatmulConfig.block_k), [`block_m`](configs.md#ReduceScatterMatmulConfig.block_m), [`block_n`](configs.md#ReduceScatterMatmulConfig.block_n), [`num_stages`](configs.md#ReduceScatterMatmulConfig.num_stages), [`num_warps`](configs.md#ReduceScatterMatmulConfig.num_warps), [`_resolve_tp_size`](reduce_scatter_matmul.md#_resolve_tp_size), [`_largest_divisor_leq`](reduce_scatter_matmul.md#_largest_divisor_leq)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`reduce_scatter_matmul`](reduce_scatter_matmul.md#reduce_scatter_matmul)

## Functions
- `_infer_axis_size(axis_name: str)` — [`L106`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L106) — Infer collective axis size from the active ``shard_map`` context.
- `_largest_divisor_leq(x: int, upper: int, candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1))` — [`L76`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L76) — Return the largest divisor of ``x`` that is also <= ``upper``.
- `_resolve_tp_size(tp_size: int | None, axis_name: str)` — [`L130`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L130) — Resolve tensor-parallel world size.
- `reduce_scatter_matmul(x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"], axis_name: str,, *, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, tp_size: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: ReduceScatterMatmulConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L437`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L437) — Reduce-scatter matmul with automatic backend selection and autotuning.

## Module values
- `_reduce_scatter_matmul_executor` — [`L423`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/reduce_scatter_matmul.py#L423)

