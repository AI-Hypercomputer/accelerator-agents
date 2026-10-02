---
title: 'Module: ejkernel/modules/operations/rwkv7.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/rwkv7.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.rwkv7`/
symbols:
  RWKV7.run: RWKV7#run().
  RWKV7Mul.run: RWKV7Mul#run().
  _executor_rwkv7._executor_rwkv7: _executor_rwkv7._executor_rwkv7.
  _executor_rwkv7_mul._executor_rwkv7_mul: _executor_rwkv7_mul._executor_rwkv7_mul.
  RWKV7.candidate_cfgs_gpu: RWKV7#candidate_cfgs_gpu().
  RWKV7Mul.candidate_cfgs_gpu: RWKV7Mul#candidate_cfgs_gpu().
  RWKV7.candidate_cfgs: RWKV7#candidate_cfgs().
  RWKV7Mul.candidate_cfgs: RWKV7Mul#candidate_cfgs().
  RWKV7.candidate_cfgs_tpu: RWKV7#candidate_cfgs_tpu().
  RWKV7Mul.candidate_cfgs_tpu: RWKV7Mul#candidate_cfgs_tpu().
  RWKV7.get_impl: RWKV7#get_impl().
  RWKV7Mul.get_impl: RWKV7Mul#get_impl().
  RWKV7.heuristic_cfg: RWKV7#heuristic_cfg().
  RWKV7Mul.heuristic_cfg: RWKV7Mul#heuristic_cfg().
  rwkv7_mul: rwkv7_mul().
  rwkv7: rwkv7().
  RWKV7: RWKV7#
  RWKV7Mul: RWKV7Mul#
  RWKV7.__init__: RWKV7#__init__().
  RWKV7Mul.__init__: RWKV7Mul#__init__().
---
# Module: [`ejkernel/modules/operations/rwkv7.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py)

## Classes
### `RWKV7`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/rwkv7.py:86`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L86)
- doc: RWKV-7 (a,b) DPLR recurrence wrapper.
- signature: `class RWKV7(Kernel[RWKV7Config, Array]):`
- members:
  - `__init__(self)` — [`L134`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L134) — Initialize RWKV-7 kernel module.
  - `candidate_cfgs(self, inv: Invocation[RWKV7Config, Array])` — [`L270`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L270) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RWKV7Config, Array])` — [`L289`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L289) — Generate GPU candidates for RWKV-7.
  - `candidate_cfgs_tpu(self, inv: Invocation[RWKV7Config, Array])` — [`L333`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L333) — Generate TPU candidates for the XLA RWKV-7 path.
  - `get_impl(self, cfg: RWKV7Config)` — [`L142`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L142) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RWKV7Config, Array])` — [`L259`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L259) — Provide default configuration.
  - `run(self, r: Float[Array, "batch seq_len num_heads qk_head_dim"], w: Float[Array, "batch seq_len num_heads qk_head_dim"], k: Float[Array, "batch seq_len num_heads qk_head_dim"], v: Float[Array, "batch seq_len num_heads v_head_dim"], a: Float[Array, "batch seq_len num_heads qk_head_dim"], b: Float[Array, "batch seq_len num_heads qk_head_dim"], *, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV7Config)` — [`L157`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L157) — Execute RWKV-7 DPLR recurrence with (a, b) parameterization.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RWKV7Config`](configs.md#RWKV7Config), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`block_v`](configs.md#RWKV7Config.block_v), [`num_stages`](configs.md#RWKV7Config.num_stages), [`num_warps`](configs.md#RWKV7Config.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`rwkv7`](rwkv7.md#rwkv7)

### `RWKV7Mul`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/rwkv7.py:338`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L338)
- doc: RWKV-7 multiplicative (kk, a) parameterization wrapper.
- signature: `class RWKV7Mul(Kernel[RWKV7MulConfig, Array]):`
- members:
  - `__init__(self)` — [`L380`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L380) — Initialize RWKV-7 Mul kernel module.
  - `candidate_cfgs(self, inv: Invocation[RWKV7MulConfig, Array])` — [`L517`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L517) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RWKV7MulConfig, Array])` — [`L536`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L536) — Generate GPU candidates for RWKV-7 Mul (same tile space as RWKV-7).
  - `candidate_cfgs_tpu(self, inv: Invocation[RWKV7MulConfig, Array])` — [`L569`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L569) — Generate TPU candidates for the XLA RWKV-7 Mul path.
  - `get_impl(self, cfg: RWKV7MulConfig)` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L388) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RWKV7MulConfig, Array])` — [`L506`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L506) — Provide default configuration.
  - `run(self, r: Float[Array, "batch seq_len num_heads qk_head_dim"], w: Float[Array, "batch seq_len num_heads qk_head_dim"], k: Float[Array, "batch seq_len num_heads qk_head_dim"], v: Float[Array, "batch seq_len num_heads v_head_dim"], kk: Float[Array, "batch seq_len num_heads qk_head_dim"], a: Float[Array, "batch seq_len num_heads qk_head_dim"], *, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV7MulConfig)` — [`L403`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L403) — Execute RWKV-7 DPLR recurrence with (kk, a) multiplicative parameterization.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RWKV7MulConfig`](configs.md#RWKV7MulConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`block_v`](configs.md#RWKV7MulConfig.block_v), [`num_stages`](configs.md#RWKV7MulConfig.num_stages), [`num_warps`](configs.md#RWKV7MulConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`rwkv7_mul`](rwkv7.md#rwkv7_mul)

## Functions
- `rwkv7(r: Float[Array, "batch seq_len num_heads qk_head_dim"], w: Float[Array, "batch seq_len num_heads qk_head_dim"], k: Float[Array, "batch seq_len num_heads qk_head_dim"], v: Float[Array, "batch seq_len num_heads v_head_dim"], a: Float[Array, "batch seq_len num_heads qk_head_dim"], b: Float[Array, "batch seq_len num_heads qk_head_dim"],, *, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV7Config | None = None)` — [`L601`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L601) — RWKV-7 DPLR recurrence (a,b) with automatic backend selection.
- `rwkv7_mul(r: Float[Array, "batch seq_len num_heads qk_head_dim"], w: Float[Array, "batch seq_len num_heads qk_head_dim"], k: Float[Array, "batch seq_len num_heads qk_head_dim"], v: Float[Array, "batch seq_len num_heads v_head_dim"], kk: Float[Array, "batch seq_len num_heads qk_head_dim"], a: Float[Array, "batch seq_len num_heads qk_head_dim"],, *, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RWKV7MulConfig | None = None)` — [`L667`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L667) — RWKV-7 recurrence (kk,a) multiplicative parameterization.

## Module values
- `_executor_rwkv7` — [`L574`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L574)
- `_executor_rwkv7_mul` — [`L587`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/rwkv7.py#L587)

