---
title: 'Module: ejkernel/modules/operations/rwkv4.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/rwkv4.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.rwkv4`/
symbols:
  _executor._executor: _executor._executor.
  RWKV4.run: RWKV4#run().
  RWKV4.candidate_cfgs: RWKV4#candidate_cfgs().
  RWKV4.candidate_cfgs_gpu: RWKV4#candidate_cfgs_gpu().
  RWKV4.heuristic_cfg: RWKV4#heuristic_cfg().
  RWKV4.get_impl: RWKV4#get_impl().
  RWKV4._channels_from_inv: RWKV4#_channels_from_inv().
  RWKV4.candidate_cfgs_tpu: RWKV4#candidate_cfgs_tpu().
  rwkv4: rwkv4().
  RWKV4: RWKV4#
  RWKV4.__init__: RWKV4#__init__().
  RWKV4._heuristic_block_c: RWKV4#_heuristic_block_c().
---
# Module: [`ejkernel/modules/operations/rwkv4.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py)

## Classes
### `RWKV4`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/rwkv4.py:83`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L83)
- doc: RWKV-4 time-mix recurrence kernel wrapper.
- signature: `class RWKV4(Kernel[RWKV4Config, Array]):`
- members:
  - `__init__(self)` — [`L119`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L119) — Initialize RWKV-4 kernel module.
  - `_channels_from_inv(inv: Invocation[RWKV4Config, Array])` — [`L205`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L205) — Pull channel count ``C`` from the invocation's ``k`` tensor.
  - `_heuristic_block_c(channels: int)` — [`L216`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L216) — Operation-side tile heuristic — single source of truth.
  - `candidate_cfgs(self, inv: Invocation[RWKV4Config, Array])` — [`L233`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L233) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RWKV4Config, Array])` — [`L240`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L240) — Generate GPU platform + block_c candidates for RWKV-4.
  - `candidate_cfgs_tpu(self, inv: Invocation[RWKV4Config, Array])` — [`L274`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L274) — Return TPU candidates for the XLA RWKV-4 path.
  - `get_impl(self, cfg: RWKV4Config)` — [`L127`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L127) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RWKV4Config, Array])` — [`L225`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L225) — Cold-start configuration with shape-aware ``block_c``.
  - `run(self, w: Float[Array, chans], u: Float[Array, chans], k: Float[Array, "batch seq_len chans"], v: Float[Array, "batch seq_len chans"], state: Float[Array, "batch three chans"] | None = None, *, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV4Config)` — [`L142`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L142) — Execute RWKV-4 time-mix recurrence.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`args`](../../ops/core/kernel.md#Invocation.args), [`RWKV4Config`](configs.md#RWKV4Config), [`block_c`](configs.md#RWKV4Config.block_c)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`rwkv4`](rwkv4.md#rwkv4)

## Functions
- `rwkv4(w: Float[Array, chans], u: Float[Array, chans], k: Float[Array, "batch seq_len chans"], v: Float[Array, "batch seq_len chans"], state: Float[Array, "batch three chans"] | None = None,, *, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV4Config | None = None)` — [`L293`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L293) — RWKV-4 time-mix recurrence with automatic backend selection.

## Module values
- `_executor` — [`L279`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv4.py#L279)

