---
title: 'Module: ejkernel/modules/operations/state_space_v1.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/state_space_v1.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.state_space_v1`/
symbols:
  _state_space_v1_executor._state_space_v1_executor: _state_space_v1_executor._state_space_v1_executor.
  StateSpaceV1.candidate_cfgs_gpu: StateSpaceV1#candidate_cfgs_gpu().
  StateSpaceV1.run: StateSpaceV1#run().
  StateSpaceV1.candidate_cfgs: StateSpaceV1#candidate_cfgs().
  StateSpaceV1.heuristic_cfg: StateSpaceV1#heuristic_cfg().
  StateSpaceV1.get_impl: StateSpaceV1#get_impl().
  StateSpaceV1.candidate_cfgs_tpu: StateSpaceV1#candidate_cfgs_tpu().
  StateSpaceV1._DN_from_inv: StateSpaceV1#_DN_from_inv().
  state_space_v1: state_space_v1().
  StateSpaceV1: StateSpaceV1#
  StateSpaceV1._heuristic_block_d: StateSpaceV1#_heuristic_block_d().
  StateSpaceV1.__init__: StateSpaceV1#__init__().
---
# Module: [`ejkernel/modules/operations/state_space_v1.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py)

## Classes
### `StateSpaceV1`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/state_space_v1.py:73`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L73)
- doc: SSM1 (Mamba1-style) Selective State Space operation.
- signature: `class StateSpaceV1(Kernel[StateSpaceV1Config, Array]):`
- members:
  - `_DN_from_inv(inv: Invocation[StateSpaceV1Config, Array])` — [`L222`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L222) — Pull ``(intermediate_size, ssm_state_size)`` from invocation.
  - `__init__(self)` — [`L115`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L115) — Initialize StateSpaceV1 module.
  - `_heuristic_block_d(D: int, N: int)` — [`L235`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L235) — Operation-side tile heuristic for the SSM-1 scan kernel.
  - `candidate_cfgs(self, inv: Invocation[StateSpaceV1Config, Array])` — [`L258`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L258) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[StateSpaceV1Config, Array])` — [`L266`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L266) — Generate GPU candidates for TileLang and XLA SSM1.
  - `candidate_cfgs_tpu(self, inv: Invocation[StateSpaceV1Config, Array])` — [`L283`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L283) — Generate TPU candidates for the XLA SSM1 path.
  - `get_impl(self, cfg: StateSpaceV1Config)` — [`L123`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L123) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[StateSpaceV1Config, Array])` — [`L248`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L248) — Cold-start configuration with shape-aware ``block_d``.
  - `run(self, hidden_states: Float[Array, "batch seq_len intermediate_size"], A: Float[Array, "intermediate_size ssm_state_size"], B: Float[Array, "batch seq_len ssm_state_size"], C: Float[Array, "batch seq_len ssm_state_size"], D: Float[Array, intermediate_size], dt: Float[Array, "batch seq_len intermediate_size"], gate: Float[Array, "batch seq_len intermediate_size"] | None = None, initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None, conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None, act_fn: Callable[[jax.Array], jax.Array] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: StateSpaceV1Config)` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L138) — Execute SSM1 selective state space operation.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`args`](../../ops/core/kernel.md#Invocation.args), [`StateSpaceV1Config`](configs.md#StateSpaceV1Config), [`block_d`](configs.md#StateSpaceV1Config.block_d), [`block_e`](configs.md#StateSpaceV1Config.block_e)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`state_space_v1`](state_space_v1.md#state_space_v1)

## Functions
- `state_space_v1(hidden_states: Float[Array, "batch seq_len intermediate_size"], A: Float[Array, "intermediate_size ssm_state_size"], B: Float[Array, "batch seq_len ssm_state_size"], C: Float[Array, "batch seq_len ssm_state_size"], D: Float[Array, intermediate_size], dt: Float[Array, "batch seq_len intermediate_size"],, gate: Float[Array, "batch seq_len intermediate_size"] | None = None, initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None, conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None, *, act_fn: Callable[[jax.Array], jax.Array] | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: StateSpaceV1Config | None = None)` — [`L302`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L302) — Execute SSM1 (Mamba1-style) selective state space with automatic optimization.

## Module values
- `_state_space_v1_executor` — [`L288`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v1.py#L288)

