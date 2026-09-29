---
title: 'Module: ejkernel/modules/operations/gated_delta_rule_grouped_decode.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/gated_delta_rule_grouped_decode.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.gated_delta_rule_grouped_decode`/
symbols:
  _executor._executor: _executor._executor.
  GatedDeltaRuleGroupedDecode.get_impl: GatedDeltaRuleGroupedDecode#get_impl().
  GatedDeltaRuleGroupedDecode.run: GatedDeltaRuleGroupedDecode#run().
  GatedDeltaRuleGroupedDecode.heuristic_cfg: GatedDeltaRuleGroupedDecode#heuristic_cfg().
  gated_delta_rule_grouped_decode: gated_delta_rule_grouped_decode().
  GatedDeltaRuleGroupedDecode.candidate_cfgs: GatedDeltaRuleGroupedDecode#candidate_cfgs().
  GatedDeltaRuleGroupedDecode.candidate_cfgs_tpu: GatedDeltaRuleGroupedDecode#candidate_cfgs_tpu().
  GatedDeltaRuleGroupedDecode.create_shard_map_wrapper: GatedDeltaRuleGroupedDecode#create_shard_map_wrapper().
  GatedDeltaRuleGroupedDecode.candidate_cfgs_gpu: GatedDeltaRuleGroupedDecode#candidate_cfgs_gpu().
  GatedDeltaRuleGroupedDecode: GatedDeltaRuleGroupedDecode#
  gated_delta_rule_grouped_decode_op: gated_delta_rule_grouped_decode_op.
  GatedDeltaRuleGroupedDecode._wrapped: GatedDeltaRuleGroupedDecode#_wrapped().
  GatedDeltaRuleGroupedDecode.__init__: GatedDeltaRuleGroupedDecode#__init__().
  GatedDeltaRuleGroupedDecode.candidate_cfgs_xla: GatedDeltaRuleGroupedDecode#candidate_cfgs_xla.
  GatedDeltaRuleGroupedDecode.candidate_cfgs_any: GatedDeltaRuleGroupedDecode#candidate_cfgs_any.
  GatedDeltaRuleGroupedDecode.heuristic_cfg_shard_map: GatedDeltaRuleGroupedDecode#heuristic_cfg_shard_map.
  GatedDeltaRuleGroupedDecode.candidate_cfgs_shard_map: GatedDeltaRuleGroupedDecode#candidate_cfgs_shard_map.
  GatedDeltaRuleGroupedDecode.candidate_cfgs_shard_map_xla: GatedDeltaRuleGroupedDecode#candidate_cfgs_shard_map_xla.
  GatedDeltaRuleGroupedDecode.candidate_cfgs_shard_map_gpu: GatedDeltaRuleGroupedDecode#candidate_cfgs_shard_map_gpu.
  GatedDeltaRuleGroupedDecode.candidate_cfgs_shard_map_tpu: GatedDeltaRuleGroupedDecode#candidate_cfgs_shard_map_tpu.
  GatedDeltaRuleGroupedDecode.version: GatedDeltaRuleGroupedDecode#version.
---
# Module: [`ejkernel/modules/operations/gated_delta_rule_grouped_decode.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py)

## Classes
### `GatedDeltaRuleGroupedDecode`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/gated_delta_rule_grouped_decode.py:50`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L50)
- doc: Executor-driven wrapper for grouped GDR single-step decode.
- signature: `class GatedDeltaRuleGroupedDecode(Kernel[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L65`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L65) — Initialize the grouped GDR decode kernel.
  - `_wrapped(q, k, v, b, d, s)` — [`L152`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L152) — Run grouped GDR decode on a per-shard slice of the inputs.
  - `candidate_cfgs(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]])` — [`L260`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L260) — Return the default candidate config list (heuristic only).
  - `candidate_cfgs_gpu(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]])` — [`L271`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L271) — Return GPU autotuning candidates (heuristic XLA only).
  - `candidate_cfgs_tpu(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]])` — [`L282`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L282) — Return TPU autotuning candidates (XLA reference plus Pallas).
  - `create_shard_map_wrapper(self, query: Float[Array, "batch num_k_heads head_dim"], key: Float[Array, "batch num_k_heads head_dim"], value: Float[Array, "batch num_k_heads expand_ratio value_dim"], beta: Float[Array, "batch num_k_heads expand_ratio"], decay: Float[Array, "batch num_k_heads expand_ratio"] | None, recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"], *, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GatedDeltaRuleGroupedDecodeConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: tuple[PartitionSpec | None, ...] | None = None, check_vma: bool = False)` — [`L88`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L88) — Create a ``shard_map`` wrapper for TP head-sharded grouped decode.
  - `get_impl(self, cfg: GatedDeltaRuleGroupedDecodeConfig)` — [`L73`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L73) — Resolve the backend implementation for a config.
  - `heuristic_cfg(self, inv: Invocation[GatedDeltaRuleGroupedDecodeConfig, tuple[Array, Array]])` — [`L248`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L248) — Return the default XLA config for an invocation.
  - `run(self, query: Float[Array, "batch num_k_heads head_dim"], key: Float[Array, "batch num_k_heads head_dim"], value: Float[Array, "batch num_k_heads expand_ratio value_dim"], beta: Float[Array, "batch num_k_heads expand_ratio"], decay: Float[Array, "batch num_k_heads expand_ratio"] | None, recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"], *, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GatedDeltaRuleGroupedDecodeConfig, **_)` — [`L186`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L186) — Run the selected grouped GDR decode backend implementation.
  - `candidate_cfgs_any` — [`L298`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L298)
  - `candidate_cfgs_shard_map` — [`L300`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L300)
  - `candidate_cfgs_shard_map_gpu` — [`L302`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L302)
  - `candidate_cfgs_shard_map_tpu` — [`L303`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L303)
  - `candidate_cfgs_shard_map_xla` — [`L301`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L301)
  - `candidate_cfgs_xla` — [`L297`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L297)
  - `heuristic_cfg_shard_map` — [`L299`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L299)
  - `version` — [`L63`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L63)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`GatedDeltaRuleGroupedDecodeConfig`](configs.md#GatedDeltaRuleGroupedDecodeConfig), [`mesh_to_jax_mesh`](../base.md#mesh_to_jax_mesh)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`gated_delta_rule_grouped_decode`](gated_delta_rule_grouped_decode.md#gated_delta_rule_grouped_decode)

## Functions
- `gated_delta_rule_grouped_decode(query: Float[Array, "batch num_k_heads head_dim"], key: Float[Array, "batch num_k_heads head_dim"], value: Float[Array, "batch num_k_heads expand_ratio value_dim"], beta: Float[Array, "batch num_k_heads expand_ratio"], decay: Float[Array, "batch num_k_heads expand_ratio"] | None, recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"], *, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GatedDeltaRuleGroupedDecodeConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: tuple[PartitionSpec | None, ...] | None = None, check_vma: bool = False)` — [`L320`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L320) — Execute grouped GDR decode through the eJKernel operation stack.

## Module values
- `_executor` — [`L306`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L306)
- `gated_delta_rule_grouped_decode_op` — [`L383`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_delta_rule_grouped_decode.py#L383)

