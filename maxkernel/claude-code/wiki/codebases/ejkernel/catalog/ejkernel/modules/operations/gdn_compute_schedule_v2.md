---
title: 'Module: ejkernel/modules/operations/gdn_compute_schedule_v2.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/gdn_compute_schedule_v2.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.gdn_compute_schedule_v2`/
symbols:
  _executor._executor: _executor._executor.
  GDNComputeScheduleV2.get_impl: GDNComputeScheduleV2#get_impl().
  compute_schedule_table_v2: compute_schedule_table_v2().
  GDNComputeScheduleV2.heuristic_cfg: GDNComputeScheduleV2#heuristic_cfg().
  GDNComputeScheduleV2.candidate_cfgs_gpu: GDNComputeScheduleV2#candidate_cfgs_gpu().
  GDNComputeScheduleV2.candidate_cfgs_tpu: GDNComputeScheduleV2#candidate_cfgs_tpu().
  GDNComputeScheduleV2.candidate_cfgs: GDNComputeScheduleV2#candidate_cfgs().
  GDNComputeScheduleV2: GDNComputeScheduleV2#
  GDNComputeScheduleV2.run: GDNComputeScheduleV2#run().
  gdn_compute_schedule_v2: gdn_compute_schedule_v2.
  GDNComputeScheduleV2.__init__: GDNComputeScheduleV2#__init__().
  GDNComputeScheduleV2.candidate_cfgs_xla: GDNComputeScheduleV2#candidate_cfgs_xla.
  GDNComputeScheduleV2.candidate_cfgs_any: GDNComputeScheduleV2#candidate_cfgs_any.
  GDNComputeScheduleV2.version: GDNComputeScheduleV2#version.
  __all__: __all__.
---
# Module: [`ejkernel/modules/operations/gdn_compute_schedule_v2.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py)

## Classes
### `GDNComputeScheduleV2`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/gdn_compute_schedule_v2.py:46`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L46)
- doc: Executor-driven schedule-table builder for packed GDN v2.
- signature: `class GDNComputeScheduleV2(Kernel[GDNComputeScheduleV2Config, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L60`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L60) — Initialize the GDN v2 schedule-table kernel.
  - `candidate_cfgs(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]])` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L133) — Return the default candidate config list (heuristic only).
  - `candidate_cfgs_gpu(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]])` — [`L144`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L144) — Return GPU autotuning candidates (XLA only).
  - `candidate_cfgs_tpu(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]])` — [`L158`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L158) — Return TPU autotuning candidates (XLA only).
  - `get_impl(self, cfg: GDNComputeScheduleV2Config)` — [`L68`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L68) — Resolve the backend implementation for a config.
  - `heuristic_cfg(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]])` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L121) — Return the default XLA config for an invocation.
  - `run(self, query_start_loc: Int[Array, num_requests_plus_one], decode_tokens: int | Int[Array, ""], num_valid_seqs: int | Int[Array, ""], max_tokens: int, chunk_size: int, BT: int | None = None, alignment: int = 8, *, cfg: GDNComputeScheduleV2Config)` — [`L82`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L82) — Build the schedule table with the selected backend.
  - `candidate_cfgs_any` — [`L173`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L173)
  - `candidate_cfgs_xla` — [`L172`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L172)
  - `version` — [`L58`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L58)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`GDNComputeScheduleV2Config`](configs.md#GDNComputeScheduleV2Config)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`compute_schedule_table_v2`](gdn_compute_schedule_v2.md#compute_schedule_table_v2)

## Functions
- `compute_schedule_table_v2(query_start_loc: jax.Array, decode_tokens: int | jax.Array, num_valid_seqs: int | jax.Array, max_tokens: int, chunk_size: int, BT: int | None = None, alignment: int = 8, *, cfg: GDNComputeScheduleV2Config | None = None)` — [`L190`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L190) — Build the GDN v2 schedule table through the eJKernel operation stack.

## Module values
- `__all__` — [`L257`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L257)
- `_executor` — [`L176`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L176)
- `gdn_compute_schedule_v2` — [`L255`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gdn_compute_schedule_v2.py#L255)

