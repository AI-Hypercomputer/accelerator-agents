---
title: 'Module: ejkernel/modules/operations/fused_kl_divergence.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/fused_kl_divergence.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.fused_kl_divergence`/
symbols:
  FusedKLDivergence.run: FusedKLDivergence#run().
  FusedKLDivergence.candidate_cfgs_gpu: FusedKLDivergence#candidate_cfgs_gpu().
  FusedKLDivergence.heuristic_cfg: FusedKLDivergence#heuristic_cfg().
  _executor._executor: _executor._executor.
  FusedKLDivergence.candidate_cfgs_tpu: FusedKLDivergence#candidate_cfgs_tpu().
  fused_kl_divergence: fused_kl_divergence().
  FusedKLDivergence.get_impl: FusedKLDivergence#get_impl().
  FusedKLDivergence.create_shard_map_wrapper: FusedKLDivergence#create_shard_map_wrapper().
  FusedKLDivergence._shape_from_inv: FusedKLDivergence#_shape_from_inv().
  FusedKLDivergence.candidate_cfgs: FusedKLDivergence#candidate_cfgs().
  FusedKLDivergence: FusedKLDivergence#
  _flatten_axes: _flatten_axes().
  _infer_leading_axes: _infer_leading_axes().
  PlatformName: PlatformName.
  KLDivergenceOutput: KLDivergenceOutput#
  FusedKLDivergence._heuristic_block_v: FusedKLDivergence#_heuristic_block_v().
  FusedKLDivergence.__init__: FusedKLDivergence#__init__().
  FusedKLDivergence.candidate_cfgs_shard_map_gpu: FusedKLDivergence#candidate_cfgs_shard_map_gpu.
  FusedKLDivergence.candidate_cfgs_shard_map_tpu: FusedKLDivergence#candidate_cfgs_shard_map_tpu.
  _infer_vocab_axis: _infer_vocab_axis().
  FusedKLDivergence._heuristic_block_m: FusedKLDivergence#_heuristic_block_m().
  KLDivergenceOutput.loss: KLDivergenceOutput#loss.
  KLDivergenceOutput.weight_sum: KLDivergenceOutput#weight_sum.
  KLDivergenceOutput.teacher_entropy: KLDivergenceOutput#teacher_entropy.
  FusedKLDivergence._per_device: FusedKLDivergence#_per_device().
---
# Module: [`ejkernel/modules/operations/fused_kl_divergence.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py)

## Classes
### `FusedKLDivergence`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/fused_kl_divergence.py:131`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L131)
- doc: Fused forward KL between two logit tensors with platform + sharding auto-dispatch.
- signature: `class FusedKLDivergence(Kernel[FusedKLDivergenceConfig, Array]):`
- members:
  - `__init__(self)` — [`L134`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L134) — Create the operation object bound to the registry op id.
  - `_heuristic_block_m(n: int)` — [`L352`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L352) — Pick the row-block size used before autotuning has a cache hit.
  - `_heuristic_block_v(v: int)` — [`L340`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L340) — Operation-side cold-start ``block_v``. The autotuner sweeps
  - `_per_device(*args)` — [`L274`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L274) — Run one shard-local KL call and merge scalar reductions globally.
  - `_shape_from_inv(inv: Invocation[FusedKLDivergenceConfig, Array])` — [`L325`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L325) — Extract ``(num_rows, vocab_size)`` from the invocation's student logits.
  - `candidate_cfgs(self, inv: Invocation[FusedKLDivergenceConfig, Array])` — [`L368`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L368) — Return autotune candidates for the default GPU-tuning path.
  - `candidate_cfgs_gpu(self, inv: Invocation[FusedKLDivergenceConfig, Array])` — [`L372`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L372) — GPU candidates for fused KL.
  - `candidate_cfgs_tpu(self, inv: Invocation[FusedKLDivergenceConfig, Array])` — [`L432`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L432) — Return TPU autotune candidates: XLA baseline + Pallas streaming variants.
  - `create_shard_map_wrapper(self, student_logits: Float[Array, "... vocab_size"], teacher_logits: Float[Array, "... vocab_size"], weights: Float[Array, ...] | None = None, *, attention_mask: Array | None = None, reduction: str = "mean", direction: str = "forward", temperature: float = 1, beta: float = 0.5, vocab_parallel_axis: str | None = None, platform: PlatformName | None = None, cfg: FusedKLDivergenceConfig, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = True)` — [`L205`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L205) — Wrap the KL call in ``shard_map`` with automatic collective insertion.
  - `get_impl(self, cfg: FusedKLDivergenceConfig)` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L138) — Resolve the concrete backend implementation for ``cfg``.
  - `heuristic_cfg(self, inv: Invocation[FusedKLDivergenceConfig, Array])` — [`L356`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L356) — Build the non-autotuned fallback config for this invocation.
  - `run(self, student_logits: Float[Array, "... vocab_size"], teacher_logits: Float[Array, "... vocab_size"], weights: Float[Array, ...] | None = None, *, attention_mask: Array | None = None, reduction: str = "mean", direction: str = "forward", temperature: float = 1, beta: float = 0.5, vocab_parallel_axis: str | None = None, platform: PlatformName | None = None, cfg: FusedKLDivergenceConfig)` — [`L143`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L143) — Run the registered KL backend with optional mask folding.
  - `candidate_cfgs_shard_map_gpu` — [`L461`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L461)
  - `candidate_cfgs_shard_map_tpu` — [`L462`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L462)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`args`](../../ops/core/kernel.md#Invocation.args), [`FusedKLDivergenceConfig`](configs.md#FusedKLDivergenceConfig), [`block_m`](configs.md#FusedKLDivergenceConfig.block_m), [`block_v`](configs.md#FusedKLDivergenceConfig.block_v), [`num_stages`](configs.md#FusedKLDivergenceConfig.num_stages), [`num_warps`](configs.md#FusedKLDivergenceConfig.num_warps), [`_infer_leading_axes`](fused_kl_divergence.md#_infer_leading_axes), [`PlatformName`](fused_kl_divergence.md#PlatformName), [`_infer_vocab_axis`](fused_kl_divergence.md#_infer_vocab_axis)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`fused_kl_divergence`](fused_kl_divergence.md#fused_kl_divergence)

### `KLDivergenceOutput`  ·  implements/extends NamedTuple
- def: [`ejkernel/modules/operations/fused_kl_divergence.py:68`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L68)
- doc: Per-call KL-divergence metrics returned by `fused_kl_divergence`.
- signature: `class KLDivergenceOutput(NamedTuple):`
- members:
  - `loss` — [`L85`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L85)
  - `teacher_entropy` — [`L87`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L87)
  - `weight_sum` — [`L86`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L86)
- used by: [`fused_kl_divergence`](fused_kl_divergence.md#fused_kl_divergence)

## Functions
- `_flatten_axes(spec_entry)` — [`L90`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L90) — Return the mesh-axis names referenced by a single PartitionSpec entry.
- `_infer_leading_axes(leading_spec: PartitionSpec | None)` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L121) — Return the flat list of mesh axes sharding the leading (batch/seq) dims.
- `_infer_vocab_axis(logits_spec: PartitionSpec | None)` — [`L104`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L104) — Pull the vocab-axis mesh name out of the logits partition spec.
- `fused_kl_divergence(student_logits: Float[Array, "... vocab_size"], teacher_logits: Float[Array, "... vocab_size"], weights: Float[Array, ...] | None = None,, *, attention_mask: Array | None = None, reduction: str = "mean", direction: str = "forward", temperature: float = 1, beta: float = 0.5, vocab_parallel_axis: str | None = None, return_teacher_entropy: bool = False, platform: PlatformName | None = None, cfg: FusedKLDivergenceConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L479`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L479) — Fused KL divergence (forward / reverse / JSD) with temperature softening.

## Module values
- `PlatformName` — [`L65`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L65)
- `_executor` — [`L465`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_kl_divergence.py#L465)

