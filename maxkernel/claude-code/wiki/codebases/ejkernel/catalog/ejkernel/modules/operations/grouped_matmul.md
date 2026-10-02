---
title: 'Module: ejkernel/modules/operations/grouped_matmul.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/grouped_matmul.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.grouped_matmul`/
symbols:
  GroupedMatmul.run: GroupedMatmul#run().
  GroupedMatmul.candidate_cfgs_gpu: GroupedMatmul#candidate_cfgs_gpu().
  _grouped_matmul_executor._grouped_matmul_executor: _grouped_matmul_executor._grouped_matmul_executor.
  GroupedMatmul.heuristic_cfg: GroupedMatmul#heuristic_cfg().
  GroupedMatmul.candidate_cfgs: GroupedMatmul#candidate_cfgs().
  GroupedMatmul.candidate_cfgs_tpu: GroupedMatmul#candidate_cfgs_tpu().
  GroupedMatmul.get_impl: GroupedMatmul#get_impl().
  GroupedMatmul.create_shard_map_wrapper: GroupedMatmul#create_shard_map_wrapper().
  grouped_matmul: grouped_matmul().
  GroupedMatmul: GroupedMatmul#
  GroupedMatmul._wrapped_grouped_matmul: GroupedMatmul#_wrapped_grouped_matmul().
  GroupedMatmul.__init__: GroupedMatmul#__init__().
  GroupedMatmul.callback: GroupedMatmul#callback().
---
# Module: [`ejkernel/modules/operations/grouped_matmul.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py)

## Classes
### `GroupedMatmul`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/grouped_matmul.py:100`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L100)
- doc: Grouped Matrix Multiplication with custom optimization logic.
- signature: `class GroupedMatmul(Kernel[GroupedMatmulConfig, Array]):`
- members:
  - `__init__(self, version: int = 1)` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L121) — Initialize Grouped Matmul module.
  - `_wrapped_grouped_matmul(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards])` — [`L217`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L217) — Shard-local grouped matmul forwarding to self.run.
  - `callback(out, cfg)` — [`L255`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L255) — Strip padding rows from the output if padding was applied.
  - `candidate_cfgs(self, inv: Invocation[GroupedMatmulConfig, Array])` — [`L397`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L397) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[GroupedMatmulConfig, Array])` — [`L439`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L439) — Generate GPU candidates for grouped matmul across TileLang and XLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[GroupedMatmulConfig, Array])` — [`L506`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L506) — Generate TPU candidates for Pallas and XLA grouped matmul.
  - `create_shard_map_wrapper(self, lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None, rhs_bias: Float[Array, "num_groups 1 n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, do_padding: bool = True, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, out_shard_callback: Callable[[Float[Array, "m n"]], Float[Array, "m n"]] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: GroupedMatmulConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L157`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L157) — Create a shard_map wrapper for distributed grouped matmul execution.
  - `get_impl(self, cfg: GroupedMatmulConfig)` — [`L142`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L142) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[GroupedMatmulConfig, Array])` — [`L374`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L374) — Provide default configuration with block sizes.
  - `run(self, lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None, rhs_bias: Float[Array, "num_groups 1 n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, do_padding: bool = True, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, out_shard_callback: Callable[[Float[Array, "m n"]], Float[Array, "m n"]] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: GroupedMatmulConfig)` — [`L263`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L263) — Execute grouped matrix multiplication.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`op_id`](../../ops/core/kernel.md#Kernel.op_id), [`GroupedMatmulConfig`](configs.md#GroupedMatmulConfig), [`block_m`](configs.md#GroupedMatmulConfig.block_m), [`block_k`](configs.md#GroupedMatmulConfig.block_k), [`block_n`](configs.md#GroupedMatmulConfig.block_n), [`num_stages`](configs.md#GroupedMatmulConfig.num_stages), [`num_warps`](configs.md#GroupedMatmulConfig.num_warps), [`bypass_xla_tiling`](configs.md#GroupedMatmulConfig.bypass_xla_tiling)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`grouped_matmul`](grouped_matmul.md#grouped_matmul)

## Functions
- `grouped_matmul(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None,, *, preferred_element_type: DTypeLike = jnp.float32, rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None, rhs_bias: Float[Array, "num_groups 1 n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, do_padding: bool = True, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, use_v2: bool = False, use_v3: bool = False, out_shard_callback: Callable[[Float[Array, "m n"]], Float[Array, "m n"]] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GroupedMatmulConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L544`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L544) — Execute grouped matrix multiplication with automatic optimization.

## Module values
- `_grouped_matmul_executor` — [`L530`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/grouped_matmul.py#L530)

