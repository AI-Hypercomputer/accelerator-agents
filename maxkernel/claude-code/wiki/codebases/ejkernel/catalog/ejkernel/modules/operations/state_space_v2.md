---
title: 'Module: ejkernel/modules/operations/state_space_v2.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/state_space_v2.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.state_space_v2`/
symbols:
  StateSpaceV2.candidate_cfgs: StateSpaceV2#candidate_cfgs().
  _state_space_v2_executor._state_space_v2_executor: _state_space_v2_executor._state_space_v2_executor.
  StateSpaceV2.run: StateSpaceV2#run().
  StateSpaceV2.candidate_cfgs_gpu: StateSpaceV2#candidate_cfgs_gpu().
  StateSpaceV2.heuristic_cfg: StateSpaceV2#heuristic_cfg().
  StateSpaceV2.candidate_cfgs_tpu: StateSpaceV2#candidate_cfgs_tpu().
  StateSpaceV2.get_impl: StateSpaceV2#get_impl().
  state_space_v2: state_space_v2().
  StateSpaceV2: StateSpaceV2#
  StateSpaceV2.__init__: StateSpaceV2#__init__().
---
# Module: [`ejkernel/modules/operations/state_space_v2.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py)

## Classes
### `StateSpaceV2`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/state_space_v2.py:76`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L76)
- doc: SSM2 (Mamba2-style) Selective State Space operation.
- signature: `class StateSpaceV2(Kernel[StateSpaceV2Config, Array]):`
- members:
  - `__init__(self)` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L121) — Initialize StateSpaceV2 module.
  - `candidate_cfgs(self, inv: Invocation[StateSpaceV2Config, Array])` — [`L254`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L254) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[StateSpaceV2Config, Array])` — [`L278`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L278) — Generate GPU candidates for TileLang and XLA SSM2.
  - `candidate_cfgs_tpu(self, inv: Invocation[StateSpaceV2Config, Array])` — [`L317`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L317) — Generate TPU candidates for the XLA SSM2 path.
  - `get_impl(self, cfg: StateSpaceV2Config)` — [`L129`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L129) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[StateSpaceV2Config, Array])` — [`L236`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L236) — Provide default configuration.
  - `run(self, x: Float[Array, "batch seq_len num_heads head_dim"], A: Float[Array, num_heads], B: Float[Array, "batch seq_len n_groups ssm_state_size"], C: Float[Array, "batch seq_len n_groups ssm_state_size"], D: Float[Array, num_heads], dt: Float[Array, "batch seq_len num_heads"], gate: Float[Array, "batch seq_len intermediate_size"] | None = None, initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None, conv_state: Float[Array, "batch conv_dim d_conv"] | None = None, n_groups: int = 1, act_fn: Callable[[jax.Array], jax.Array] | None = None, use_gated_rmsnorm: bool = False, rmsnorm_eps: float = 0.00001, precision: lax.Precision | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: StateSpaceV2Config)` — [`L144`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L144) — Execute SSM2 selective state space operation.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`StateSpaceV2Config`](configs.md#StateSpaceV2Config), [`block_e`](configs.md#StateSpaceV2Config.block_e), [`n_groups`](configs.md#StateSpaceV2Config.n_groups), [`rmsnorm_eps`](configs.md#StateSpaceV2Config.rmsnorm_eps), [`use_gated_rmsnorm`](configs.md#StateSpaceV2Config.use_gated_rmsnorm)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`state_space_v2`](state_space_v2.md#state_space_v2)

## Functions
- `state_space_v2(x: Float[Array, "batch seq_len num_heads head_dim"], A: Float[Array, num_heads], B: Float[Array, "batch seq_len n_groups ssm_state_size"], C: Float[Array, "batch seq_len n_groups ssm_state_size"], D: Float[Array, num_heads], dt: Float[Array, "batch seq_len num_heads"],, gate: Float[Array, "batch seq_len intermediate_size"] | None = None, initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None, conv_state: Float[Array, "batch conv_dim d_conv"] | None = None, *, n_groups: int = 1, act_fn: Callable[[jax.Array], jax.Array] | None = None, use_gated_rmsnorm: bool = False, rmsnorm_eps: float = 0.00001, precision: lax.Precision | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: StateSpaceV2Config | None = None)` — [`L345`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L345) — Execute SSM2 (Mamba2-style) selective state space with automatic optimization.

## Module values
- `_state_space_v2_executor` — [`L331`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/state_space_v2.py#L331)

