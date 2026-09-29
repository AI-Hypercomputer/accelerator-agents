---
title: 'Module: ejkernel/modules/operations/fused_conv_decode.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/fused_conv_decode.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.fused_conv_decode`/
symbols:
  _executor._executor: _executor._executor.
  FusedConvDecode.heuristic_cfg: FusedConvDecode#heuristic_cfg().
  FusedConvDecode.run: FusedConvDecode#run().
  FusedConvDecode.candidate_cfgs_tpu: FusedConvDecode#candidate_cfgs_tpu().
  fused_conv_decode: fused_conv_decode().
  FusedConvDecode.get_impl: FusedConvDecode#get_impl().
  FusedConvDecode.candidate_cfgs: FusedConvDecode#candidate_cfgs().
  FusedConvDecode: FusedConvDecode#
  FusedConvDecode.candidate_cfgs_gpu: FusedConvDecode#candidate_cfgs_gpu().
  fused_conv_decode_op: fused_conv_decode_op.
  FusedConvDecode.__init__: FusedConvDecode#__init__().
  FusedConvDecode.candidate_cfgs_xla: FusedConvDecode#candidate_cfgs_xla.
  FusedConvDecode.candidate_cfgs_any: FusedConvDecode#candidate_cfgs_any.
  FusedConvDecode.version: FusedConvDecode#version.
---
# Module: [`ejkernel/modules/operations/fused_conv_decode.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py)

## Classes
### `FusedConvDecode`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/fused_conv_decode.py:50`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L50)
- doc: Executor-driven wrapper for fused conv-state decode.
- signature: `class FusedConvDecode(Kernel[FusedConvDecodeConfig, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L66`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L66) — Initialize the fused conv decode kernel.
  - `candidate_cfgs(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]])` — [`L181`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L181) — Return the default candidate config list (heuristic only).
  - `candidate_cfgs_gpu(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]])` — [`L192`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L192) — Return GPU autotuning candidates (heuristic XLA only).
  - `candidate_cfgs_tpu(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]])` — [`L206`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L206) — Return TPU autotuning candidates (XLA reference plus Pallas).
  - `get_impl(self, cfg: FusedConvDecodeConfig)` — [`L74`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L74) — Resolve the backend implementation for a config.
  - `heuristic_cfg(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]])` — [`L164`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L164) — Return the default XLA config for an invocation.
  - `run(self, conv_state: jnp.ndarray, new_tokens: jnp.ndarray, kernel: jnp.ndarray, *, output_dtype: jnp.dtype, activation: Callable[[Array], Array] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: FusedConvDecodeConfig, **_)` — [`L88`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L88) — Run the selected fused conv decode backend implementation.
  - `candidate_cfgs_any` — [`L223`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L223)
  - `candidate_cfgs_xla` — [`L222`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L222)
  - `version` — [`L64`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L64)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`FusedConvDecodeConfig`](configs.md#FusedConvDecodeConfig), [`activation`](configs.md#FusedConvDecodeConfig.activation), [`d_conv`](configs.md#FusedConvDecodeConfig.d_conv)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`fused_conv_decode`](fused_conv_decode.md#fused_conv_decode)

## Functions
- `fused_conv_decode(conv_state: jnp.ndarray, new_tokens: jnp.ndarray, kernel: jnp.ndarray, *, output_dtype: jnp.dtype, activation: Callable[[Array], Array] | Literal["silu", "swish", "none"] | None = None, d_conv: int = 4, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: FusedConvDecodeConfig | None = None)` — [`L240`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L240) — Execute fused conv-state decode through the eJKernel operation stack.

## Module values
- `_executor` — [`L226`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L226)
- `fused_conv_decode_op` — [`L309`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_conv_decode.py#L309)

