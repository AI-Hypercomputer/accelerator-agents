---
title: 'Module: ejkernel/modules/operations/gated_delta_rule.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/gated_delta_rule.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.gated_delta_rule`/
symbols:
  GatedDeltaRule.run: GatedDeltaRule#run().
  _executor._executor: _executor._executor.
  GatedDeltaRule.candidate_cfgs_gpu: GatedDeltaRule#candidate_cfgs_gpu().
  GatedDeltaRule.candidate_cfgs_tpu: GatedDeltaRule#candidate_cfgs_tpu().
  GatedDeltaRule.candidate_cfgs: GatedDeltaRule#candidate_cfgs().
  GatedDeltaRule.get_impl: GatedDeltaRule#get_impl().
  GatedDeltaRule._wrapped_gdr: GatedDeltaRule#_wrapped_gdr().
  GatedDeltaRule.heuristic_cfg: GatedDeltaRule#heuristic_cfg().
  gated_delta_rule: gated_delta_rule().
  GatedDeltaRule.resolve_platform: GatedDeltaRule#resolve_platform().
  GatedDeltaRule: GatedDeltaRule#
  GatedDeltaRule.create_shard_map_wrapper: GatedDeltaRule#create_shard_map_wrapper().
  GatedDeltaRule.__init__: GatedDeltaRule#__init__().
  GatedDeltaRule.heuristic_cfg_shard_map: GatedDeltaRule#heuristic_cfg_shard_map.
  GatedDeltaRule.candidate_cfgs_shard_map: GatedDeltaRule#candidate_cfgs_shard_map.
  GatedDeltaRule.candidate_cfgs_shard_map_xla: GatedDeltaRule#candidate_cfgs_shard_map_xla.
  GatedDeltaRule.candidate_cfgs_shard_map_gpu: GatedDeltaRule#candidate_cfgs_shard_map_gpu.
  GatedDeltaRule.candidate_cfgs_shard_map_tpu: GatedDeltaRule#candidate_cfgs_shard_map_tpu.
  GatedDeltaRule.version: GatedDeltaRule#version.
---
# Module: [`ejkernel/modules/operations/gated_delta_rule.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py)

## Classes
### `GatedDeltaRule`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/gated_delta_rule.py:92`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L92)
- doc: Gated Delta Rule (GDR) operation.
- signature: `class GatedDeltaRule(Kernel[GatedDeltaRuleConfig, Array]):`
- members:
  - `__init__(self)` — [`L112`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L112) — Initialize GatedDeltaRule operation.
  - `_wrapped_gdr(query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_heads qk_head_dim"], value: Float[Array, "batch seq_len num_heads v_head_dim"], beta: Float[Array, "batch seq_len num_heads"], decay: Float[Array, "batch seq_len num_heads"] | None, initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None, seg_ids: Int[Array, "batch seq_len"] | None = None)` — [`L192`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L192) — Run GDR on a single shard by delegating to ``self.run``.
  - `candidate_cfgs(self, inv: Invocation[GatedDeltaRuleConfig, Array])` — [`L418`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L418) — Generate the platform-agnostic candidate configurations for chunk-size autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[GatedDeltaRuleConfig, Array])` — [`L438`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L438) — Generate GPU autotuning candidates for GDR across TileLang and XLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[GatedDeltaRuleConfig, Array])` — [`L472`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L472) — Generate TPU autotuning candidates for the Pallas and XLA GDR paths.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], key: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], value: Float[Array, "batch seq_len num_value_heads v_head_dim"], beta: Float[Array, "batch seq_len num_value_heads"], decay: Float[Array, "batch seq_len num_value_heads"] | None = None, initial_state: Float[Array, "batch num_value_heads qk_head_dim v_head_dim"] | None = None, *, use_qk_l2norm: bool = True, use_chunked: bool = True, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GatedDeltaRuleConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None, check_vma: bool = False, seg_ids: Int[Array, "batch seq_len"] | None = None, **_)` — [`L120`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L120) — Create a ``jax.shard_map`` wrapper for distributed GDR execution.
  - `get_impl(self, cfg: GatedDeltaRuleConfig)` — [`L266`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L266) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[GatedDeltaRuleConfig, Array])` — [`L401`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L401) — Provide the default (non-autotuned) configuration for GDR.
  - `resolve_platform(self, cfg: GatedDeltaRuleConfig)` — [`L281`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L281) — Resolve ``auto`` to the concrete registered GDR platform.
  - `run(self, query: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], key: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], value: Float[Array, "batch seq_len num_value_heads v_head_dim"], beta: Float[Array, "batch seq_len num_value_heads"], decay: Float[Array, "batch seq_len num_value_heads"] | None = None, initial_state: Float[Array, "batch num_value_heads qk_head_dim v_head_dim"] | None = None, seg_ids: Int[Array, "batch seq_len"] | None = None, *, use_qk_l2norm: bool = True, use_chunked: bool = True, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GatedDeltaRuleConfig, **_)` — [`L285`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L285) — Execute Gated Delta Rule operation.
  - `candidate_cfgs_shard_map` — [`L500`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L500)
  - `candidate_cfgs_shard_map_gpu` — [`L502`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L502)
  - `candidate_cfgs_shard_map_tpu` — [`L503`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L503)
  - `candidate_cfgs_shard_map_xla` — [`L501`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L501)
  - `heuristic_cfg_shard_map` — [`L499`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L499)
  - `version` — [`L110`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L110)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`PALLAS`](../../kernels/_registry.md#Platform.PALLAS), [`GatedDeltaRuleConfig`](configs.md#GatedDeltaRuleConfig), [`chunk_size`](configs.md#GatedDeltaRuleConfig.chunk_size), [`XLA`](../../kernels/_registry.md#Platform.XLA), [`use_input_dtype_phase1_outputs`](configs.md#GatedDeltaRuleConfig.use_input_dtype_phase1_outputs), [`use_chunked`](configs.md#GatedDeltaRuleConfig.use_chunked), [`use_input_dtype_state`](configs.md#GatedDeltaRuleConfig.use_input_dtype_state)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`gated_delta_rule`](gated_delta_rule.md#gated_delta_rule)

## Functions
- `gated_delta_rule(query: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], key: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], value: Float[Array, "batch seq_len num_value_heads v_head_dim"], beta: Float[Array, "batch seq_len num_value_heads"], decay: Float[Array, "batch seq_len num_value_heads"] | None = None, seg_ids: Int[Array, "batch seq_len"] | None = None, initial_state: Float[Array, "batch num_value_heads qk_head_dim v_head_dim"] | None = None,, *, autotune_chunk_candidates: tuple[int, ...] | list[int] | None = None, use_qk_l2norm: bool = True, use_chunked: bool = True, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GatedDeltaRuleConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None)` — [`L520`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L520) — Execute Gated Delta Rule (GDR) with automatic optimization.

## Module values
- `_executor` — [`L506`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule.py#L506)

