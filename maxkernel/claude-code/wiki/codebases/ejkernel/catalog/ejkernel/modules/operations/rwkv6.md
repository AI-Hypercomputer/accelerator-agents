---
title: 'Module: ejkernel/modules/operations/rwkv6.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/rwkv6.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.rwkv6`/
symbols:
  _executor._executor: _executor._executor.
  RWKV6.get_impl: RWKV6#get_impl().
  RWKV6.run: RWKV6#run().
  RWKV6.candidate_cfgs_gpu: RWKV6#candidate_cfgs_gpu().
  RWKV6.candidate_cfgs: RWKV6#candidate_cfgs().
  RWKV6.heuristic_cfg: RWKV6#heuristic_cfg().
  RWKV6.candidate_cfgs_tpu: RWKV6#candidate_cfgs_tpu().
  rwkv6: rwkv6().
  RWKV6: RWKV6#
  RWKV6.__init__: RWKV6#__init__().
---
# Module: [`ejkernel/modules/operations/rwkv6.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py)

## Classes
### `RWKV6`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/rwkv6.py:82`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L82)
- doc: RWKV-6 recurrence kernel wrapper.
- signature: `class RWKV6(Kernel[RWKV6Config, Array]):`
- members:
  - `__init__(self)` — [`L126`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L126) — Initialize RWKV-6 kernel module.
  - `candidate_cfgs(self, inv: Invocation[RWKV6Config, Array])` — [`L236`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L236) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RWKV6Config, Array])` — [`L254`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L254) — Generate GPU platform candidates for RWKV-6.
  - `candidate_cfgs_tpu(self, inv: Invocation[RWKV6Config, Array])` — [`L267`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L267) — Return TPU candidates for the XLA RWKV-6 path.
  - `get_impl(self, cfg: RWKV6Config)` — [`L134`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L134) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RWKV6Config, Array])` — [`L225`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L225) — Provide default configuration.
  - `run(self, r: Float[Array, "batch seq_len num_heads qk_head_dim"], k: Float[Array, "batch seq_len num_heads qk_head_dim"], v: Float[Array, "batch seq_len num_heads v_head_dim"], w: Float[Array, "batch seq_len num_heads qk_head_dim"], u: Float[Array, "num_heads qk_head_dim"], *, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV6Config)` — [`L149`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L149) — Execute RWKV-6 linear attention recurrence.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RWKV6Config`](configs.md#RWKV6Config)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`rwkv6`](rwkv6.md#rwkv6)

## Functions
- `rwkv6(r: Float[Array, "batch seq_len num_heads qk_head_dim"], k: Float[Array, "batch seq_len num_heads qk_head_dim"], v: Float[Array, "batch seq_len num_heads v_head_dim"], w: Float[Array, "batch seq_len num_heads qk_head_dim"], u: Float[Array, "num_heads qk_head_dim"],, *, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV6Config | None = None)` — [`L286`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L286) — RWKV-6 recurrence with automatic backend selection.

## Module values
- `_executor` — [`L272`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv6.py#L272)

