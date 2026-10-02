---
title: 'Module: ejkernel/modules/operations/all_gather_matmul.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/all_gather_matmul.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.all_gather_matmul`/
symbols:
  AllGatherMatmul.run: AllGatherMatmul#run().
  AllGatherMatmul.candidate_cfgs_tpu: AllGatherMatmul#candidate_cfgs_tpu().
  AllGatherMatmul.heuristic_cfg_tpu: AllGatherMatmul#heuristic_cfg_tpu().
  _all_gather_matmul_executor._all_gather_matmul_executor: _all_gather_matmul_executor._all_gather_matmul_executor.
  AllGatherMatmul.candidate_cfgs_gpu: AllGatherMatmul#candidate_cfgs_gpu().
  AllGatherMatmul.candidate_cfgs: AllGatherMatmul#candidate_cfgs().
  AllGatherMatmul.heuristic_cfg: AllGatherMatmul#heuristic_cfg().
  AllGatherMatmul.get_impl: AllGatherMatmul#get_impl().
  _inv_xy_rhs_transpose: _inv_xy_rhs_transpose().
  all_gather_matmul: all_gather_matmul().
  AllGatherMatmul: AllGatherMatmul#
  _largest_divisor_leq: _largest_divisor_leq().
  AllGatherMatmul.create_shard_map_wrapper: AllGatherMatmul#create_shard_map_wrapper().
  AllGatherMatmul._wrapped: AllGatherMatmul#_wrapped().
  AllGatherMatmul.__init__: AllGatherMatmul#__init__().
  AllGatherMatmul.candidate_cfgs_shard_map_tpu: AllGatherMatmul#candidate_cfgs_shard_map_tpu.
---
# Module: [`ejkernel/modules/operations/all_gather_matmul.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py)

## Classes
### `AllGatherMatmul`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/all_gather_matmul.py:120`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L120)
- doc: Fused all-gather + matmul for tensor-parallel linear layers.
- signature: `class AllGatherMatmul(Kernel[AllGatherMatmulConfig, Array]):`
- members:
  - `candidate_cfgs(self, inv: Invocation[AllGatherMatmulConfig, Array])` — [`L313`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L313) — Return candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[AllGatherMatmulConfig, Array])` — [`L329`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L329) — Return GPU candidates for TileLang and XLA all-gather matmul paths.
  - `candidate_cfgs_tpu(self, inv: Invocation[AllGatherMatmulConfig, Array])` — [`L380`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L380) — Return TPU/Pallas candidate configurations for autotuning.
  - `create_shard_map_wrapper(self, x: Float[Array, "m_local k"], y: Float[Array, "k n_local"] | Float[Array, "n_local k"], axis_name: str, rhs_transpose: bool = False, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, tp_size: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: AllGatherMatmulConfig, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L153`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L153) — Build a shard_map-wrapped callable and its input arguments.
  - `get_impl(self, cfg: AllGatherMatmulConfig)` — [`L144`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L144) — Get the kernel implementation for the given configuration.
  - `heuristic_cfg(self, inv: Invocation[AllGatherMatmulConfig, Array])` — [`L286`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L286) — Return default heuristic configuration for any platform.
  - `heuristic_cfg_tpu(self, inv: Invocation[AllGatherMatmulConfig, Array])` — [`L297`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L297) — Return TPU/Pallas-oriented default configuration.
  - `run(self, x: Float[Array, "m_local k"], y: Float[Array, "k n_local"] | Float[Array, "n_local k"], axis_name: str, rhs_transpose: bool = False, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, tp_size: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: AllGatherMatmulConfig)` — [`L226`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L226) — Execute all-gather matmul with the selected backend.
  - `candidate_cfgs_shard_map_tpu` — [`L412`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L412)
- protocol/private: `__init__`[`L141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L141), `_wrapped`[`L196`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L196)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`AllGatherMatmulConfig`](configs.md#AllGatherMatmulConfig), [`op_id`](../../ops/core/kernel.md#Kernel.op_id), [`_inv_xy_rhs_transpose`](all_gather_matmul.md#_inv_xy_rhs_transpose), [`block_k`](configs.md#AllGatherMatmulConfig.block_k), [`block_n`](configs.md#AllGatherMatmulConfig.block_n), [`num_stages`](configs.md#AllGatherMatmulConfig.num_stages), [`num_warps`](configs.md#AllGatherMatmulConfig.num_warps), [`_largest_divisor_leq`](all_gather_matmul.md#_largest_divisor_leq)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`all_gather_matmul`](all_gather_matmul.md#all_gather_matmul)

## Functions
- `_inv_xy_rhs_transpose(inv: Invocation[AllGatherMatmulConfig, Array])` — [`L93`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L93) — Extract ``x``, ``y``, and ``rhs_transpose`` from an invocation.
- `_largest_divisor_leq(x: int, upper: int, candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1))` — [`L65`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L65) — Return the largest divisor of ``x`` that is also ``<= upper``.
- `all_gather_matmul(x: Float[Array, "m_local k"], y: Float[Array, "k n_local"] | Float[Array, "n_local k"], axis_name: str,, *, rhs_transpose: bool = False, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT, tp_size: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: AllGatherMatmulConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L429`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L429) — All-gather matmul with automatic backend selection and autotuning.

## Module values
- `_all_gather_matmul_executor` — [`L415`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/all_gather_matmul.py#L415)

