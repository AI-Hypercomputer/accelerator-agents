---
title: 'Module: ejkernel/modules/operations/ragged_gated_delta_rule_v2.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ragged_gated_delta_rule_v2.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ragged_gated_delta_rule_v2`/
symbols:
  ragged_gated_delta_rule_v2: ragged_gated_delta_rule_v2().
  _executor._executor: _executor._executor.
  RaggedGatedDeltaRuleV2.run: RaggedGatedDeltaRuleV2#run().
  RaggedGatedDeltaRuleV2.create_shard_map_wrapper: RaggedGatedDeltaRuleV2#create_shard_map_wrapper().
  RaggedGatedDeltaRuleV2.candidate_cfgs: RaggedGatedDeltaRuleV2#candidate_cfgs().
  RaggedGatedDeltaRuleV2.get_impl: RaggedGatedDeltaRuleV2#get_impl().
  RaggedGatedDeltaRuleV2.candidate_cfgs_gpu: RaggedGatedDeltaRuleV2#candidate_cfgs_gpu().
  RaggedGatedDeltaRuleV2.candidate_cfgs_tpu: RaggedGatedDeltaRuleV2#candidate_cfgs_tpu().
  RaggedGatedDeltaRuleV2.heuristic_cfg: RaggedGatedDeltaRuleV2#heuristic_cfg().
  RaggedGatedDeltaRuleV2._run_unsharded: RaggedGatedDeltaRuleV2#_run_unsharded().
  RaggedGatedDeltaRuleV2: RaggedGatedDeltaRuleV2#
  _select_head_shard_axis: _select_head_shard_axis().
  _is_tpu_pallas_request: _is_tpu_pallas_request().
  mesh_axis_size: mesh_axis_size().
  _mesh_axis_items: _mesh_axis_items().
  _supports_head_shard: _supports_head_shard().
  ragged_gated_delta_rule_v2_op: ragged_gated_delta_rule_v2_op.
  _has_nontrivial_mesh_axis: _has_nontrivial_mesh_axis().
  RaggedGatedDeltaRuleV2._wrapped_split: RaggedGatedDeltaRuleV2#_wrapped_split().
  RaggedGatedDeltaRuleV2._wrapped: RaggedGatedDeltaRuleV2#_wrapped().
  RaggedGatedDeltaRuleV2.__init__: RaggedGatedDeltaRuleV2#__init__().
  RaggedGatedDeltaRuleV2.candidate_cfgs_xla: RaggedGatedDeltaRuleV2#candidate_cfgs_xla.
  RaggedGatedDeltaRuleV2.candidate_cfgs_any: RaggedGatedDeltaRuleV2#candidate_cfgs_any.
  RaggedGatedDeltaRuleV2.candidate_cfgs_shard_map: RaggedGatedDeltaRuleV2#candidate_cfgs_shard_map.
  RaggedGatedDeltaRuleV2.candidate_cfgs_shard_map_xla: RaggedGatedDeltaRuleV2#candidate_cfgs_shard_map_xla.
  RaggedGatedDeltaRuleV2.candidate_cfgs_shard_map_gpu: RaggedGatedDeltaRuleV2#candidate_cfgs_shard_map_gpu.
  RaggedGatedDeltaRuleV2.candidate_cfgs_shard_map_tpu: RaggedGatedDeltaRuleV2#candidate_cfgs_shard_map_tpu.
  _normalize_runtime_dtype: _normalize_runtime_dtype().
  _reorder_concatenated_tensor_for_sharding: _reorder_concatenated_tensor_for_sharding().
  RaggedGatedDeltaRuleV2._flatten_output: RaggedGatedDeltaRuleV2#_flatten_output().
  RaggedGatedDeltaRuleV2.version: RaggedGatedDeltaRuleV2#version.
  __all__: __all__.
---
# Module: [`ejkernel/modules/operations/ragged_gated_delta_rule_v2.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py)

## Classes
### `RaggedGatedDeltaRuleV2`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ragged_gated_delta_rule_v2.py:259`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L259)
- doc: Executor-driven wrapper for packed mixed-QKV ragged GDN.
- signature: `class RaggedGatedDeltaRuleV2(Kernel[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L270`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L270) — Initialize the operation with its registry id ``ragged_gated_delta_rule_v2``.
  - `_flatten_output(outs, cfg)` — [`L501`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L501) — Flatten the per-head-sharded shard_map output back to the packed value layout.
  - `_run_unsharded(self, mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int, use_qk_norm_in_gdn: bool, apply_silu_in_gdr: bool, use_recurrent_scan_prefill: bool, mask_initial_state: bool, runtime_dtype: object | None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None, cfg: RaggedGatedDeltaRuleV2Config | None)` — [`L699`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L699) — Run ragged GDN v2 through the normal executor path.
  - `_wrapped(local_mixed_qkv, local_b, local_a, local_state, local_A_log, local_dt_bias, local_qsl, local_si, local_dist, local_has_initial_state)` — [`L536`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L536) — Run the GDN v2 backend on one tensor-parallel shard of the packed ``mixed_qkv`` buffer.
  - `_wrapped_split(local_query, local_key, local_value, local_b, local_a, local_state, local_A_log, local_dt_bias, local_qsl, local_si, local_dist, local_has_initial_state)` — [`L413`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L413) — Run the GDN v2 backend on one head-parallel shard of split Q/K/V projections.
  - `candidate_cfgs(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]])` — [`L808`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L808) — Enumerate autotuning candidates for the generic/XLA path.
  - `candidate_cfgs_gpu(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]])` — [`L820`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L820) — Enumerate autotuning candidates for GPU execution.
  - `candidate_cfgs_tpu(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]])` — [`L839`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L839) — Enumerate autotuning candidates for TPU execution.
  - `create_shard_map_wrapper(self, mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int | None = None, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, runtime_dtype: object | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, split_qkv_for_head_shard: bool = False, cfg: RaggedGatedDeltaRuleV2Config | None = None, mesh: object | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: tuple[PartitionSpec, ...] | None = None, check_vma: bool = False)` — [`L292`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L292) — Create a ``shard_map`` wrapper for tensor-parallel ragged GDN v2.
  - `get_impl(self, cfg: RaggedGatedDeltaRuleV2Config)` — [`L274`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L274) — Resolve the registered backend implementation for a configuration.
  - `heuristic_cfg(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]])` — [`L790`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L790) — Return the default configuration for an invocation.
  - `run(self, mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int | None = None, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, runtime_dtype: object | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedGatedDeltaRuleV2Config, **_)` — [`L605`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L605) — Run the selected ragged GDN v2 backend implementation.
  - `candidate_cfgs_any` — [`L862`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L862)
  - `candidate_cfgs_shard_map` — [`L863`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L863)
  - `candidate_cfgs_shard_map_gpu` — [`L865`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L865)
  - `candidate_cfgs_shard_map_tpu` — [`L866`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L866)
  - `candidate_cfgs_shard_map_xla` — [`L864`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L864)
  - `candidate_cfgs_xla` — [`L861`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L861)
  - `version` — [`L268`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L268)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`_executor`](ragged_gated_delta_rule_v2.md#_executor._executor), [`RaggedGatedDeltaRuleV2Config`](configs.md#RaggedGatedDeltaRuleV2Config), [`chunk_size`](configs.md#RaggedGatedDeltaRuleV2Config.chunk_size), [`mesh_to_jax_mesh`](../base.md#mesh_to_jax_mesh), [`kernel_tile_policy`](configs.md#RaggedGatedDeltaRuleV2Config.kernel_tile_policy), [`use_fused_gdn_decode`](configs.md#RaggedGatedDeltaRuleV2Config.use_fused_gdn_decode), [`_normalize_runtime_dtype`](ragged_gated_delta_rule_v2.md#_normalize_runtime_dtype)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ragged_gated_delta_rule_v2`](ragged_gated_delta_rule_v2.md#ragged_gated_delta_rule_v2)

## Functions
- `_has_nontrivial_mesh_axis(mesh: object | None)` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L133) — Whether ``mesh`` has at least one axis with more than one device.
- `_is_tpu_pallas_request(*, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None, cfg: RaggedGatedDeltaRuleV2Config | None)` — [`L194`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L194)
- `_mesh_axis_items(mesh: object | None)` — [`L117`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L117) — Return mesh axis names and sizes for JAX/SpectraX-like meshes.
- `_normalize_runtime_dtype(runtime_dtype: object | None)` — [`L64`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L64) — Coerce a runtime dtype specifier into a canonical ``jnp.dtype``.
- `_reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array, split_sizes: tuple[int, ...], n_shards: int, dim: int)` — [`L211`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L211) — Reorder a fused ``[A|B|C]`` feature axis into per-shard interleaved layout.
- `_select_head_shard_axis(mesh: object | None, head_axis: object | None, *, n_kq: int, n_v: int, d_k: int, d_v: int)` — [`L156`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L156) — Pick a real mesh axis for GDN head sharding.
- `_supports_head_shard(mesh: object | None, *, axis_name: object | None, n_kq: int, n_v: int, d_k: int, d_v: int)` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L138) — Whether ``axis_name`` can shard the GDN value-head work.
- `mesh_axis_size(mesh: object | None, axis_name: object | None)` — [`L81`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L81) — Return the size of ``axis_name`` in a JAX/SpectraX-like mesh.
- `ragged_gated_delta_rule_v2(mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, pre_sharded_mixed_qkv: bool = False, flat_tp_shard: bool = False, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, runtime_dtype: object | None = None, mesh: object | None = None, head_axis: object | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedGatedDeltaRuleV2Config | None = None)` — [`L883`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L883) — Execute packed-inference ragged GDN v2 through eJKernel.

## Module values
- `__all__` — [`L1131`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L1131)
- `_executor` — [`L869`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L869)
- `ragged_gated_delta_rule_v2_op` — [`L1129`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule_v2.py#L1129)

