---
title: 'Module: ejkernel/modules/operations/ragged_causal_conv1d.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ragged_causal_conv1d.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ragged_causal_conv1d`/
symbols:
  _executor._executor: _executor._executor.
  ragged_causal_conv1d_head_sharded: ragged_causal_conv1d_head_sharded().
  RaggedCausalConv1D.run: RaggedCausalConv1D#run().
  RaggedCausalConv1D.heuristic_cfg: RaggedCausalConv1D#heuristic_cfg().
  ragged_causal_conv1d: ragged_causal_conv1d().
  RaggedCausalConv1D.get_impl: RaggedCausalConv1D#get_impl().
  RaggedCausalConv1D.create_shard_map_wrapper: RaggedCausalConv1D#create_shard_map_wrapper().
  RaggedCausalConv1D.candidate_cfgs: RaggedCausalConv1D#candidate_cfgs().
  RaggedCausalConv1D: RaggedCausalConv1D#
  RaggedCausalConv1D.candidate_cfgs_gpu: RaggedCausalConv1D#candidate_cfgs_gpu().
  RaggedCausalConv1D.candidate_cfgs_tpu: RaggedCausalConv1D#candidate_cfgs_tpu().
  mesh_axis_size: mesh_axis_size().
  ragged_causal_conv1d_op: ragged_causal_conv1d_op.
  RaggedCausalConv1D._wrapped: RaggedCausalConv1D#_wrapped().
  RaggedCausalConv1D.__init__: RaggedCausalConv1D#__init__().
  RaggedCausalConv1D.candidate_cfgs_xla: RaggedCausalConv1D#candidate_cfgs_xla.
  RaggedCausalConv1D.candidate_cfgs_any: RaggedCausalConv1D#candidate_cfgs_any.
  RaggedCausalConv1D.candidate_cfgs_shard_map: RaggedCausalConv1D#candidate_cfgs_shard_map.
  RaggedCausalConv1D.candidate_cfgs_shard_map_xla: RaggedCausalConv1D#candidate_cfgs_shard_map_xla.
  RaggedCausalConv1D.candidate_cfgs_shard_map_gpu: RaggedCausalConv1D#candidate_cfgs_shard_map_gpu.
  RaggedCausalConv1D.candidate_cfgs_shard_map_tpu: RaggedCausalConv1D#candidate_cfgs_shard_map_tpu.
  _reorder_concatenated_tensor_for_sharding: _reorder_concatenated_tensor_for_sharding().
  RaggedCausalConv1D.version: RaggedCausalConv1D#version.
---
# Module: [`ejkernel/modules/operations/ragged_causal_conv1d.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py)

## Classes
### `RaggedCausalConv1D`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ragged_causal_conv1d.py:136`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L136)
- doc: Executor-driven wrapper for ragged causal depthwise conv1d.
- signature: `class RaggedCausalConv1D(Kernel[RaggedCausalConv1DConfig, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L151`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L151) — Initialize the ragged causal conv1d kernel.
  - `_wrapped(local_x, local_state, local_kernel, local_qsl, local_si, local_dist)` — [`L229`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L229) — Run ragged conv1d on a per-shard slice of the packed buffers.
  - `candidate_cfgs(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]])` — [`L334`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L334) — Return the default candidate config list (heuristic only).
  - `candidate_cfgs_gpu(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]])` — [`L345`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L345) — Return GPU autotuning candidates (heuristic XLA only).
  - `candidate_cfgs_tpu(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]])` — [`L356`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L356) — Return TPU autotuning candidates (heuristic XLA only).
  - `create_shard_map_wrapper(self, x: jnp.ndarray, conv_state: jnp.ndarray, kernel: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, d_conv: int | None = None, apply_silu: bool | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedCausalConv1DConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: tuple[PartitionSpec, ...] | None = None, check_vma: bool = False)` — [`L173`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L173) — Create a ``shard_map`` wrapper for executor-driven ragged conv1d.
  - `get_impl(self, cfg: RaggedCausalConv1DConfig)` — [`L159`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L159) — Resolve the backend implementation for a config.
  - `heuristic_cfg(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]])` — [`L317`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L317) — Return the default XLA config for an invocation.
  - `run(self, x: jnp.ndarray, conv_state: jnp.ndarray, kernel: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, d_conv: int | None = None, apply_silu: bool | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedCausalConv1DConfig, **_)` — [`L265`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L265) — Run the selected ragged conv1d backend implementation.
  - `candidate_cfgs_any` — [`L368`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L368)
  - `candidate_cfgs_shard_map` — [`L369`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L369)
  - `candidate_cfgs_shard_map_gpu` — [`L371`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L371)
  - `candidate_cfgs_shard_map_tpu` — [`L372`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L372)
  - `candidate_cfgs_shard_map_xla` — [`L370`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L370)
  - `candidate_cfgs_xla` — [`L367`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L367)
  - `version` — [`L149`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L149)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RaggedCausalConv1DConfig`](configs.md#RaggedCausalConv1DConfig), [`apply_silu`](configs.md#RaggedCausalConv1DConfig.apply_silu), [`d_conv`](configs.md#RaggedCausalConv1DConfig.d_conv), [`mesh_to_jax_mesh`](../base.md#mesh_to_jax_mesh)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ragged_causal_conv1d_head_sharded`](ragged_causal_conv1d.md#ragged_causal_conv1d_head_sharded), [`ragged_causal_conv1d`](ragged_causal_conv1d.md#ragged_causal_conv1d)

## Functions
- `_reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array, split_sizes: tuple[int, ...], n_shards: int, dim: int)` — [`L90`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L90) — Arrange fused feature groups into per-shard interleaved layout.
- `mesh_axis_size(mesh: object | None, axis_name: object | None)` — [`L55`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L55) — Return the size of ``axis_name`` in a JAX/SpectraX-like mesh.
- `ragged_causal_conv1d(x: jnp.ndarray, conv_state: jnp.ndarray, kernel: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, d_conv: int = 4, apply_silu: bool = True, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedCausalConv1DConfig | None = None)` — [`L389`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L389) — Execute ragged causal conv1d through the eJKernel operation stack.
- `ragged_causal_conv1d_head_sharded(x: jnp.ndarray, conv_state: jnp.ndarray, kernel: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, split_sizes: tuple[int, ...], mesh: object | None, head_axis: object | None, d_conv: int, apply_silu: bool = True, pre_sharded: bool = False, cfg: RaggedCausalConv1DConfig | None = None)` — [`L436`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L436) — Run ragged conv1d with optional feature-axis tensor parallelism.

## Module values
- `_executor` — [`L375`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L375)
- `ragged_causal_conv1d_op` — [`L528`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_causal_conv1d.py#L528)

