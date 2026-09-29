---
title: 'Module: ejkernel/modules/operations/fused_cross_entropy.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/fused_cross_entropy.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.fused_cross_entropy`/
symbols:
  _fused_linear_cross_entropy_dispatch: _fused_linear_cross_entropy_dispatch().
  FusedCrossEntropy.run: FusedCrossEntropy#run().
  fused_cross_entropy: fused_cross_entropy().
  FusedCrossEntropy.candidate_cfgs_gpu: FusedCrossEntropy#candidate_cfgs_gpu().
  FusedCrossEntropy.heuristic_cfg: FusedCrossEntropy#heuristic_cfg().
  _executor._executor: _executor._executor.
  FusedCrossEntropy.candidate_cfgs_tpu: FusedCrossEntropy#candidate_cfgs_tpu().
  _chunked_cross_entropy_dispatch: _chunked_cross_entropy_dispatch().
  _fused_linear_cross_entropy_vp_dispatch: _fused_linear_cross_entropy_vp_dispatch().
  FusedCrossEntropy.get_impl: FusedCrossEntropy#get_impl().
  FusedCrossEntropy.create_shard_map_wrapper: FusedCrossEntropy#create_shard_map_wrapper().
  FusedCrossEntropy._shape_from_inv: FusedCrossEntropy#_shape_from_inv().
  FusedCrossEntropy.candidate_cfgs: FusedCrossEntropy#candidate_cfgs().
  FusedCrossEntropy: FusedCrossEntropy#
  CrossEntropyOutput: CrossEntropyOutput#
  _flatten_axes: _flatten_axes().
  _infer_leading_axes: _infer_leading_axes().
  PlatformName: PlatformName.
  CrossEntropyOutput.loss: CrossEntropyOutput#loss.
  CrossEntropyOutput.z_loss: CrossEntropyOutput#z_loss.
  CrossEntropyOutput.weight_sum: CrossEntropyOutput#weight_sum.
  CrossEntropyOutput.accuracy: CrossEntropyOutput#accuracy.
  _infer_vocab_axis: _infer_vocab_axis().
  _fused_linear_cross_entropy_vp_dispatch._per_device: _fused_linear_cross_entropy_vp_dispatch()._per_device().
  FusedCrossEntropy._heuristic_block_v: FusedCrossEntropy#_heuristic_block_v().
  _combine_weights: _combine_weights().
  FusedCrossEntropy.__init__: FusedCrossEntropy#__init__().
  FusedCrossEntropy.candidate_cfgs_shard_map_gpu: FusedCrossEntropy#candidate_cfgs_shard_map_gpu.
  FusedCrossEntropy.candidate_cfgs_shard_map_tpu: FusedCrossEntropy#candidate_cfgs_shard_map_tpu.
  FusedCrossEntropy._heuristic_block_m: FusedCrossEntropy#_heuristic_block_m().
  ChunkStrategy: ChunkStrategy.
  FusedCrossEntropy._per_device: FusedCrossEntropy#_per_device().
---
# Module: [`ejkernel/modules/operations/fused_cross_entropy.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py)

## Classes
### `CrossEntropyOutput`  ·  implements/extends NamedTuple
- def: [`ejkernel/modules/operations/fused_cross_entropy.py:67`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L67)
- doc: Per-call cross-entropy metrics returned by `fused_cross_entropy`.
- signature: `class CrossEntropyOutput(NamedTuple):`
- members:
  - `accuracy` — [`L89`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L89)
  - `loss` — [`L86`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L86)
  - `weight_sum` — [`L88`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L88)
  - `z_loss` — [`L87`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L87)
- used by: [`_fused_linear_cross_entropy_dispatch`](fused_cross_entropy.md#_fused_linear_cross_entropy_dispatch), [`fused_cross_entropy`](fused_cross_entropy.md#fused_cross_entropy), [`_chunked_cross_entropy_dispatch`](fused_cross_entropy.md#_chunked_cross_entropy_dispatch), [`_fused_linear_cross_entropy_vp_dispatch`](fused_cross_entropy.md#_fused_linear_cross_entropy_vp_dispatch)

### `FusedCrossEntropy`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/fused_cross_entropy.py:131`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L131)
- doc: Fused cross-entropy with platform + sharding auto-dispatch.
- signature: `class FusedCrossEntropy(Kernel[FusedCrossEntropyConfig, Array]):`
- members:
  - `__init__(self)` — [`L139`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L139) — Create the operation object bound to the registry op id.
  - `_heuristic_block_m(n: int)` — [`L390`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L390) — Pick the row-block size used before autotuning has a cache hit.
  - `_heuristic_block_v(v: int)` — [`L375`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L375) — Operation-side ``block_v`` heuristic (mirrors the kernel-side fallback).
  - `_per_device(*args)` — [`L305`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L305) — Run one shard-local loss and merge scalar reductions globally.
  - `_shape_from_inv(inv: Invocation[FusedCrossEntropyConfig, Array])` — [`L360`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L360) — Extract ``(num_rows, vocab_size)`` from the invocation's logits arg.
  - `candidate_cfgs(self, inv: Invocation[FusedCrossEntropyConfig, Array])` — [`L406`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L406) — Return autotune candidates for the default GPU-tuning path.
  - `candidate_cfgs_gpu(self, inv: Invocation[FusedCrossEntropyConfig, Array])` — [`L410`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L410) — GPU candidates: enumerate (block_v, block_m, num_warps) for
  - `candidate_cfgs_tpu(self, inv: Invocation[FusedCrossEntropyConfig, Array])` — [`L478`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L478) — Return TPU autotune candidates: XLA baseline + Pallas streaming variants.
  - `create_shard_map_wrapper(self, logits: Float[Array, "... vocab_size"], targets: Int[Array, ...] | None = None, weights: Float[Array, ...] | None = None, *, attention_mask: Array | None = None, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, soft_targets: Float[Array, "... vocab_size"] | None = None, reduction: str = "mean", vocab_parallel_axis: str | None = None, platform: PlatformName | None = None, cfg: FusedCrossEntropyConfig, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = True)` — [`L218`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L218) — Wrap the loss call in ``shard_map`` with automatic collective insertion.
  - `get_impl(self, cfg: FusedCrossEntropyConfig)` — [`L143`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L143) — Resolve the concrete backend implementation for ``cfg``.
  - `heuristic_cfg(self, inv: Invocation[FusedCrossEntropyConfig, Array])` — [`L394`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L394) — Build the non-autotuned fallback config for this invocation.
  - `run(self, logits: Float[Array, "... vocab_size"], targets: Int[Array, ...] | None = None, weights: Float[Array, ...] | None = None, *, attention_mask: Array | None = None, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, soft_targets: Float[Array, "... vocab_size"] | None = None, reduction: str = "mean", vocab_parallel_axis: str | None = None, platform: PlatformName | None = None, cfg: FusedCrossEntropyConfig)` — [`L148`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L148) — Run the registered backend, returning ``(loss, per_row_correct)``.
  - `candidate_cfgs_shard_map_gpu` — [`L512`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L512)
  - `candidate_cfgs_shard_map_tpu` — [`L513`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L513)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`args`](../../ops/core/kernel.md#Invocation.args), [`FusedCrossEntropyConfig`](configs.md#FusedCrossEntropyConfig), [`block_m`](configs.md#FusedCrossEntropyConfig.block_m), [`block_v`](configs.md#FusedCrossEntropyConfig.block_v), [`num_stages`](configs.md#FusedCrossEntropyConfig.num_stages), [`num_warps`](configs.md#FusedCrossEntropyConfig.num_warps), [`_infer_leading_axes`](fused_cross_entropy.md#_infer_leading_axes), [`PlatformName`](fused_cross_entropy.md#PlatformName), [`_infer_vocab_axis`](fused_cross_entropy.md#_infer_vocab_axis)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`fused_cross_entropy`](fused_cross_entropy.md#fused_cross_entropy)

## Functions
- `_chunked_cross_entropy_dispatch(*, logits, targets, weights, attention_mask, ignore_index, label_smoothing, z_loss, reduction, chunk_size, chunk_strategy, compute_dtype, checkpoint=True)` — [`L546`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L546) — Route a chunked-logits CE call to the matching XLA streaming kernel.
- `_combine_weights(targets, weights, attention_mask, ignore_index, compute_dtype)` — [`L530`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L530) — Build effective per-token float weights: ``(weights or valid) * mask``.
- `_flatten_axes(spec_entry)` — [`L92`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L92) — Return the mesh-axis names referenced by a single PartitionSpec entry.
- `_fused_linear_cross_entropy_dispatch(*, hidden, targets, weights, lm_head_weight, lm_head_bias, lm_head_fn, logit_softcap, attention_mask, ignore_index, label_smoothing, z_loss, reduction, token_chunk_size, compute_dtype, checkpoint=True, platform: PlatformName | None = None, cfg: FusedCrossEntropyConfig | None = None, sparse_skip=False)` — [`L604`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L604) — Route an FLCE call to the token-chunked fused-linear kernel.
- `_fused_linear_cross_entropy_vp_dispatch(*, hidden, targets, weights, lm_head_weight, lm_head_bias, lm_head_fn, logit_softcap, attention_mask, ignore_index, reduction, token_chunk_size, compute_dtype, checkpoint, vocab_parallel_axis, mesh, in_specs, out_specs, sparse_skip=False)` — [`L699`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L699) — Vocab-parallel fused-linear cross-entropy (FLCE) wrapped in ``shard_map``.
- `_infer_leading_axes(leading_spec: PartitionSpec | None)` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L121) — Return the flat list of mesh axes sharding the leading (batch/seq) dims.
- `_infer_vocab_axis(logits_spec: PartitionSpec | None)` — [`L106`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L106) — Pull the vocab-axis mesh name out of the logits partition spec.
- `_per_device(*args)` — [`L760`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L760)
- `fused_cross_entropy(logits: Float[Array, "... vocab_size"] | None = None, targets: Int[Array, ...] | None = None, weights: Float[Array, ...] | None = None, *, hidden: Float[Array, "... hidden_size"] | None = None, lm_head_weight: Float[Array, "hidden_size vocab_size"] | None = None, lm_head_bias: Float[Array, vocab_size] | None = None, lm_head_fn: Callable[[Array], Array] | None = None, logit_softcap: float | None = None, chunk_size: int = 0, chunk_strategy: ChunkStrategy = "vocab", attention_mask: Array | None = None, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, soft_targets: Float[Array, "... vocab_size"] | None = None, reduction: str = "mean", vocab_parallel_axis: str | None = None, compute_dtype: jnp.dtype | None = None, checkpoint: bool = True, sparse_skip: bool = False, platform: PlatformName | None = None, cfg: FusedCrossEntropyConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L809`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L809) — Fused cross-entropy with automatic platform + sharding dispatch.

## Module values
- `ChunkStrategy` — [`L64`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L64)
- `PlatformName` — [`L63`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L63)
- `_executor` — [`L516`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/fused_cross_entropy.py#L516)

