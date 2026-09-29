---
title: 'Module: ejkernel/modules/operations/ragged_gated_delta_rule.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ragged_gated_delta_rule.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ragged_gated_delta_rule`/
symbols:
  _executor._executor: _executor._executor.
  RaggedGatedDeltaRule.create_shard_map_wrapper: RaggedGatedDeltaRule#create_shard_map_wrapper().
  RaggedGatedDeltaRule.candidate_cfgs_gpu: RaggedGatedDeltaRule#candidate_cfgs_gpu().
  RaggedGatedDeltaRule.get_impl: RaggedGatedDeltaRule#get_impl().
  ragged_gated_delta_rule: ragged_gated_delta_rule().
  RaggedGatedDeltaRule.run: RaggedGatedDeltaRule#run().
  RaggedGatedDeltaRule.candidate_cfgs: RaggedGatedDeltaRule#candidate_cfgs().
  RaggedGatedDeltaRule.candidate_cfgs_tpu: RaggedGatedDeltaRule#candidate_cfgs_tpu().
  RaggedGatedDeltaRule.heuristic_cfg: RaggedGatedDeltaRule#heuristic_cfg().
  RaggedGatedDeltaRule: RaggedGatedDeltaRule#
  RaggedGatedDeltaRule._dispatch: RaggedGatedDeltaRule#_dispatch().
  RaggedGatedDeltaRule._wrapped: RaggedGatedDeltaRule#_wrapped().
  RaggedGatedDeltaRule._decode_shard: RaggedGatedDeltaRule#_decode_shard().
  RaggedGatedDeltaRule._prefill_shard: RaggedGatedDeltaRule#_prefill_shard().
  RaggedGatedDeltaRule.__init__: RaggedGatedDeltaRule#__init__().
  RaggedGatedDeltaRule.candidate_cfgs_shard_map_gpu: RaggedGatedDeltaRule#candidate_cfgs_shard_map_gpu.
  RaggedGatedDeltaRule.candidate_cfgs_shard_map_tpu: RaggedGatedDeltaRule#candidate_cfgs_shard_map_tpu.
  RaggedGatedDeltaRule._run_decode: RaggedGatedDeltaRule#_run_decode().
  RaggedGatedDeltaRule._run_prefill: RaggedGatedDeltaRule#_run_prefill().
  RaggedGatedDeltaRule.version: RaggedGatedDeltaRule#version.
---
# Module: [`ejkernel/modules/operations/ragged_gated_delta_rule.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py)

## Classes
### `RaggedGatedDeltaRule`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ragged_gated_delta_rule.py:66`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L66)
- doc: Ragged Gated Delta Rule operation for packed continuous-batching inference.
- signature: `class RaggedGatedDeltaRule(Kernel[RaggedGatedDeltaRuleConfig, Array]):`
- members:
  - `__init__(self)` — [`L91`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L91) — Initialize the ragged GDR operation with registry identifier.
  - `_decode_shard(q, k, v, b, d, s, si)` — [`L258`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L258) — Run the per-shard decode (length-1) path via the TPU Pallas kernel.
  - `_dispatch(q, k, v, b, d, s, qsl, si)` — [`L340`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L340) — Select decode vs prefill at trace time and run the matching sharded path.
  - `_prefill_shard(q, k, v, b, d, s, qsl, si)` — [`L297`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L297) — Run the per-shard prefill (variable-length) path via the XLA chunked kernel.
  - `_run_decode(_)` — [`L377`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L377) — ``lax.cond`` true-branch: invoke the sharded decode function.
  - `_run_prefill(_)` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L388) — ``lax.cond`` false-branch: invoke the sharded prefill function.
  - `_wrapped(q, k, v, b, d, s, qsl, si)` — [`L407`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L407) — Run ragged GDR on a single shard via ``self.run`` (non-TPU-Pallas path).
  - `candidate_cfgs(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array])` — [`L466`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L466) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array])` — [`L480`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L480) — Generate GPU autotuning candidates for ragged GDR across TileLang and XLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array])` — [`L514`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L514) — Generate TPU autotuning candidates for ragged GDR across Pallas and XLA.
  - `create_shard_map_wrapper(self, query: Float[Array, "num_tokens num_heads qk_head_dim"], key: Float[Array, "num_tokens num_heads qk_head_dim"], value: Float[Array, "num_tokens num_heads v_head_dim"], beta: Float[Array, "num_tokens num_heads"], decay: Float[Array, "num_tokens num_heads"] | None, recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"], query_start_loc: Int[Array, num_requests_plus_1], state_indices: Int[Array, num_requests], *, use_qk_l2norm: bool = True, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedGatedDeltaRuleConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: tuple[PartitionSpec, ...] | None = None, check_vma: bool = False)` — [`L170`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L170) — Create a ``shard_map`` wrapper for distributed ragged GDR.
  - `get_impl(self, cfg: RaggedGatedDeltaRuleConfig)` — [`L95`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L95) — Retrieve the kernel implementation from the registry.
  - `heuristic_cfg(self, inv: Invocation[RaggedGatedDeltaRuleConfig, Array])` — [`L451`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L451) — Return the default configuration for ragged GDR.
  - `run(self, query: Float[Array, "num_tokens num_heads qk_head_dim"], key: Float[Array, "num_tokens num_heads qk_head_dim"], value: Float[Array, "num_tokens num_heads v_head_dim"], beta: Float[Array, "num_tokens num_heads"], decay: Float[Array, "num_tokens num_heads"] | None, recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"], query_start_loc: Int[Array, num_requests_plus_1], state_indices: Int[Array, num_requests], *, use_qk_l2norm: bool = True, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedGatedDeltaRuleConfig, **_)` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L113) — Execute ragged GDR.
  - `candidate_cfgs_shard_map_gpu` — [`L533`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L533)
  - `candidate_cfgs_shard_map_tpu` — [`L534`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L534)
  - `version` — [`L89`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L89)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RaggedGatedDeltaRuleConfig`](configs.md#RaggedGatedDeltaRuleConfig), [`_ragged_gdr_chunked_prefill`](../../kernels/_xla/ragged_gated_delta_rule/_xla_impl_fwd.md#_ragged_gdr_chunked_prefill), [`chunk_size`](configs.md#RaggedGatedDeltaRuleConfig.chunk_size), [`_decode_path`](../../kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.md#_decode_path)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ragged_gated_delta_rule`](ragged_gated_delta_rule.md#ragged_gated_delta_rule)

## Functions
- `ragged_gated_delta_rule(query: Float[Array, "num_tokens num_heads qk_head_dim"], key: Float[Array, "num_tokens num_heads qk_head_dim"], value: Float[Array, "num_tokens num_heads v_head_dim"], beta: Float[Array, "num_tokens num_heads"], decay: Float[Array, "num_tokens num_heads"] | None = None, recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"] | None = None, query_start_loc: Int[Array, num_requests_plus_1] | None = None, state_indices: Int[Array, num_requests] | None = None, *, chunk_size: int = 64, use_qk_l2norm: bool = True, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedGatedDeltaRuleConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None, check_vma: bool = False)` — [`L551`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L551) — Ragged Gated Delta Rule for packed continuous-batching inference.

## Module values
- `_executor` — [`L537`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_gated_delta_rule.py#L537)

