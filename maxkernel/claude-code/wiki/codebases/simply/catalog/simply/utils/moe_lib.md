---
title: 'Module: simply/utils/moe_lib.py'
type: catalog
provenance: extracted
module: simply/utils/moe_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.moe_lib`/
symbols:
  _overlap_fn: _overlap_fn().
  _create_pipelined_ra2a_moe.unload_fn.prepare_fn: _create_pipelined_ra2a_moe().unload_fn().prepare_fn().
  _create_pipelined_ag_moe.compute_meta: _create_pipelined_ag_moe().compute_meta().
  _create_pipelined_ra2a_moe: _create_pipelined_ra2a_moe().
  _create_pipelined_ra2a_moe.compute_meta: _create_pipelined_ra2a_moe().compute_meta().
  _create_pipelined_ra2a_moe.load_fn.prepare_fn: _create_pipelined_ra2a_moe().load_fn().prepare_fn().
  _compute_chunked_ra2a: _compute_chunked_ra2a().
  run_moe_pipelined_shard_map: run_moe_pipelined_shard_map().
  _create_pipelined_ag_moe: _create_pipelined_ag_moe().
  _create_pipelined_ra2a_moe.unload_fn.finalize_fn: _create_pipelined_ra2a_moe().unload_fn().finalize_fn().
  _create_pipelined_ra2a_moe.load_fn.finalize_fn: _create_pipelined_ra2a_moe().load_fn().finalize_fn().
  surround_compute_with_ra2a: surround_compute_with_ra2a().
  run_moe_pipelined_shard_map.make_pipeline.run_pipeline: run_moe_pipelined_shard_map().make_pipeline().run_pipeline().
  _create_pipelined_ag_moe.load_fn.finalize_fn: _create_pipelined_ag_moe().load_fn().finalize_fn().
  _create_pipelined_ag_moe.compute_fn: _create_pipelined_ag_moe().compute_fn().
  _ra2a_3d_kernel_async: _ra2a_3d_kernel_async().
  _create_pipelined_ag_moe.load_fn.prepare_fn: _create_pipelined_ag_moe().load_fn().prepare_fn().
  _create_pipelined_ag_moe.unload_fn.prepare_fn: _create_pipelined_ag_moe().unload_fn().prepare_fn().
  PipelinedMoEConfig: PipelinedMoEConfig#
  MoEMetaRA2A.local_permute: MoEMetaRA2A#local_permute.
  make_ra2a_3d: make_ra2a_3d().
  _create_pipelined_ra2a_moe.compute_fn: _create_pipelined_ra2a_moe().compute_fn().
  PipelinedMoEConfig.gathers: PipelinedMoEConfig#gathers.
  PipelinedMoEConfig.ep_method: PipelinedMoEConfig#ep_method.
  MoEMetaRA2A: MoEMetaRA2A#
  MoEMetaAG: MoEMetaAG#
  MoEMetaRA2A.info: MoEMetaRA2A#info.
  MoEMetaRA2A.preamble: MoEMetaRA2A#preamble.
  _MoEMethods.compute_meta: _MoEMethods#compute_meta.
  unique_gather: unique_gather().
  surround_compute_with_ra2a._ra2a_split: surround_compute_with_ra2a()._ra2a_split().
  run_moe_pipelined_shard_map.make_pipeline: run_moe_pipelined_shard_map().make_pipeline().
  _RA2AMeta: _RA2AMeta#
  _MoEMethods.compute_fn: _MoEMethods#compute_fn.
  make_ra2a_3d.start: make_ra2a_3d().start().
  surround_compute_with_ra2a.ra2a_split_bwd: surround_compute_with_ra2a().ra2a_split_bwd().
  _create_pipelined_ra2a_moe.load_fn: _create_pipelined_ra2a_moe().load_fn().
  _create_pipelined_ra2a_moe.unload_fn: _create_pipelined_ra2a_moe().unload_fn().
  AG_PIPELINE_COLLECTIVES: AG_PIPELINE_COLLECTIVES.
  _create_pipelined_ag_moe.load_fn: _create_pipelined_ag_moe().load_fn().
  _create_pipelined_ag_moe.unload_fn: _create_pipelined_ag_moe().unload_fn().
  PipelinedMoEConfig.ra2a: PipelinedMoEConfig#ra2a.
  MoEMetaAG.info: MoEMetaAG#info.
  PipelinedMoEConfig.fine_grained_ra2a: PipelinedMoEConfig#fine_grained_ra2a.
  PipelinedMoEConfig.__post_init__: PipelinedMoEConfig#__post_init__().
  MoEMetaAG.gather_idx: MoEMetaAG#gather_idx.
  _MoEMethods: _MoEMethods#
  _ra2a_3d_kernel_async.make_rdma: _ra2a_3d_kernel_async().make_rdma().
  surround_compute_with_ra2a.ra2a_split: surround_compute_with_ra2a().ra2a_split().
  PipelinedMoEConfig.use_pipelined_ra2a_barriers: PipelinedMoEConfig#use_pipelined_ra2a_barriers.
  MoEInfo: MoEInfo#
  MoEInfo.buffer_size: MoEInfo#buffer_size.
  MoEMetaRA2A.epilogue: MoEMetaRA2A#epilogue.
  compute_local_permute_metadata: compute_local_permute_metadata().
  unique_gather_fwd: unique_gather_fwd().
  _ra2a_3d_kernel_async.make_dma: _ra2a_3d_kernel_async().make_dma().
  make_ra2a_3d.start.ra2a_kernel_start: make_ra2a_3d().start().ra2a_kernel_start().
  make_ra2a_3d.wait: make_ra2a_3d().wait().
  make_ra2a_3d.wait.ra2a_kernel_wait: make_ra2a_3d().wait().ra2a_kernel_wait().
  surround_compute_with_ag: surround_compute_with_ag().
  surround_compute_with_ra2a.ra2a_split_fwd: surround_compute_with_ra2a().ra2a_split_fwd().
  ra2a_with_gather: ra2a_with_gather().
  _create_pipelined_ag_moe.unload_fn.finalize_fn: _create_pipelined_ag_moe().unload_fn().finalize_fn().
  SENTINEL_VALUE: SENTINEL_VALUE.
  ComputeBlockCallable: ComputeBlockCallable#
  PipelinedMoEConfig.safety_factor: PipelinedMoEConfig#safety_factor.
  PipelinedMoEConfig.pad_buffers_to_multiple: PipelinedMoEConfig#pad_buffers_to_multiple.
  _RA2AMeta.recv_sizes: _RA2AMeta#recv_sizes.
  MoEInfo.batch_size: MoEInfo#batch_size.
  MoEInfo.experts_per_tok: MoEInfo#experts_per_tok.
  LocalPermuteMetadata.isort_idx: LocalPermuteMetadata#isort_idx.
  MoEMetaRA2A.local_ra2a_sort: MoEMetaRA2A#local_ra2a_sort.
  MoEMetaRA2A.local_ra2a_isort: MoEMetaRA2A#local_ra2a_isort.
  MoEMetaAG.group_counts: MoEMetaAG#group_counts.
  MoEMetaAG.output_mask: MoEMetaAG#output_mask.
  MoEMetaAG.scales: MoEMetaAG#scales.
  _RA2A_COLLECTIVE_ID: _RA2A_COLLECTIVE_ID.
  _RDMACopy.copy: _RDMACopy#copy.
  RaggedAllToAllCallable: RaggedAllToAllCallable#
  PipelinedMoEConfig.use_scheduling_groups: PipelinedMoEConfig#use_scheduling_groups.
  LocalPermuteMetadata: LocalPermuteMetadata#
  LocalPermuteMetadata.sort_idx: LocalPermuteMetadata#sort_idx.
  MoEMetaRA2A.scales: MoEMetaRA2A#scales.
  _RDMACopy: _RDMACopy#
  maybe_pad_size: maybe_pad_size().
  surround_compute_with_ra2a.start_fn: surround_compute_with_ra2a().start_fn().
  surround_compute_with_ra2a.wait_fn: surround_compute_with_ra2a().wait_fn().
  SPARSECORE_PAD_SIZE: SPARSECORE_PAD_SIZE.
  AsyncCopyDescriptor: AsyncCopyDescriptor.
  PipelinedMoEConfig.dropless_fallback: PipelinedMoEConfig#dropless_fallback.
  _RA2AMeta.send_sizes: _RA2AMeta#send_sizes.
  MoEInfo.num_experts: MoEInfo#num_experts.
  LocalPermuteMetadata.group_counts: LocalPermuteMetadata#group_counts.
  MoEMetaRA2A.buffer_overflow: MoEMetaRA2A#buffer_overflow.
  _MoEMethods.load_fn: _MoEMethods#load_fn.
  _MoEMethods.unload_fn: _MoEMethods#unload_fn.
  _RDMACopy.start: _RDMACopy#start.
  _RDMACopy.wait: _RDMACopy#wait.
  unique_gather_bwd: unique_gather_bwd().
  surround_compute_with_ag.fn: surround_compute_with_ag().fn().
  surround_compute_with_ra2a.fn: surround_compute_with_ra2a().fn().
  _compute_chunked_ra2a._update_fn: _compute_chunked_ra2a()._update_fn().
  _ag_tokens: _ag_tokens().
  _rs_tokens: _rs_tokens().
  _SCHEDULING_GROUP_ID: _SCHEDULING_GROUP_ID.
  RaggedAllToAllCallable.__call__: RaggedAllToAllCallable#__call__().
  ComputeBlockCallable.__call__: ComputeBlockCallable#__call__().
  _RA2AMeta.input_offsets: _RA2AMeta#input_offsets.
  _RA2AMeta.output_offsets: _RA2AMeta#output_offsets.
  LocalPermuteMetadata.group_idx: LocalPermuteMetadata#group_idx.
  maybe_tpu_sublane_size: maybe_tpu_sublane_size().
  _ra2a_3d_kernel_async._final_barrier: _ra2a_3d_kernel_async()._final_barrier().
---
# Module: [`simply/utils/moe_lib.py`](../../../../../../raw/code/simply/simply/utils/moe_lib.py)

## Classes
### `ComputeBlockCallable`  ·  implements/extends Protocol
- def: [`simply/utils/moe_lib.py:34`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L34)
- signature: `class ComputeBlockCallable(Protocol):`
- protocol/private: `__call__`[`L35`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L35)
- used by: [`_create_pipelined_ra2a_moe`](moe_lib.md#_create_pipelined_ra2a_moe), [`run_moe_pipelined_shard_map`](moe_lib.md#run_moe_pipelined_shard_map), [`_create_pipelined_ag_moe`](moe_lib.md#_create_pipelined_ag_moe)

### `LocalPermuteMetadata`
- def: [`simply/utils/moe_lib.py:80`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L80)
- signature: `class LocalPermuteMetadata:`
- members:
  - `group_counts` — [`L82`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L82)
  - `group_idx` — [`L81`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L81)
  - `isort_idx` — [`L84`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L84) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
  - `sort_idx` — [`L83`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L83)
- used by: [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.prepare_fn), [`finalize_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.finalize_fn), [`compute_fn`](moe_lib.md#_create_pipelined_ra2a_moe.compute_fn), [`local_permute`](moe_lib.md#MoEMetaRA2A.local_permute), [`compute_local_permute_metadata`](moe_lib.md#compute_local_permute_metadata)

### `MoEInfo`
- def: [`simply/utils/moe_lib.py:71`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L71) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- signature: `class MoEInfo:`
- members:
  - `batch_size` — [`L72`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L72)
  - `buffer_size` — [`L75`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L75)
  - `experts_per_tok` — [`L73`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L73)
  - `num_experts` — [`L74`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L74)
- used by: [`compute_meta`](moe_lib.md#_create_pipelined_ag_moe.compute_meta), [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.prepare_fn), [`compute_meta`](moe_lib.md#_create_pipelined_ra2a_moe.compute_meta), [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.prepare_fn), [`prepare_fn`](moe_lib.md#_create_pipelined_ag_moe.load_fn.prepare_fn), [`prepare_fn`](moe_lib.md#_create_pipelined_ag_moe.unload_fn.prepare_fn), [`info`](moe_lib.md#MoEMetaRA2A.info), [`info`](moe_lib.md#MoEMetaAG.info)

### `MoEMetaAG`
- def: [`simply/utils/moe_lib.py:102`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L102) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- signature: `class MoEMetaAG:`
- members:
  - `gather_idx` — [`L104`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L104)
  - `group_counts` — [`L105`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L105)
  - `info` — [`L103`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L103)
  - `output_mask` — [`L106`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L106)
  - `scales` — [`L107`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L107) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- uses (calls/refs, reference-scoped): [`MoEInfo`](moe_lib.md#MoEInfo)
- used by: [`compute_meta`](moe_lib.md#_create_pipelined_ag_moe.compute_meta), [`compute_fn`](moe_lib.md#_create_pipelined_ag_moe.compute_fn), [`finalize_fn`](moe_lib.md#_create_pipelined_ag_moe.load_fn.finalize_fn), [`prepare_fn`](moe_lib.md#_create_pipelined_ag_moe.load_fn.prepare_fn), [`prepare_fn`](moe_lib.md#_create_pipelined_ag_moe.unload_fn.prepare_fn), [`compute_meta`](moe_lib.md#_MoEMethods.compute_meta), [`compute_fn`](moe_lib.md#_MoEMethods.compute_fn), [`finalize_fn`](moe_lib.md#_create_pipelined_ag_moe.unload_fn.finalize_fn)

### `MoEMetaRA2A`
- def: [`simply/utils/moe_lib.py:89`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L89) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- signature: `class MoEMetaRA2A:`
- members:
  - `buffer_overflow` — [`L96`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L96)
  - `epilogue` — [`L94`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L94)
  - `info` — [`L90`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L90)
  - `local_permute` — [`L95`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L95)
  - `local_ra2a_isort` — [`L92`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L92)
  - `local_ra2a_sort` — [`L91`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L91)
  - `preamble` — [`L93`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L93)
  - `scales` — [`L97`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L97)
- uses (calls/refs, reference-scoped): [`_RA2AMeta`](moe_lib.md#_RA2AMeta), [`MoEInfo`](moe_lib.md#MoEInfo), [`LocalPermuteMetadata`](moe_lib.md#LocalPermuteMetadata)
- used by: [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.prepare_fn), [`compute_meta`](moe_lib.md#_create_pipelined_ra2a_moe.compute_meta), [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.prepare_fn), [`run_moe_pipelined_shard_map`](moe_lib.md#run_moe_pipelined_shard_map), [`finalize_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.finalize_fn), [`finalize_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.finalize_fn), [`compute_fn`](moe_lib.md#_create_pipelined_ra2a_moe.compute_fn), [`compute_meta`](moe_lib.md#_MoEMethods.compute_meta), [`compute_fn`](moe_lib.md#_MoEMethods.compute_fn)

### `PipelinedMoEConfig`
- def: [`simply/utils/moe_lib.py:42`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L42) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- doc: A performance hyperparameters config for the MoE routines.
- signature: `class PipelinedMoEConfig:`
- members:
  - `dropless_fallback` — [`L52`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L52)
  - `ep_method` — [`L50`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L50)
  - `fine_grained_ra2a` — [`L51`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L51)
  - `gathers` — [`L44`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L44)
  - `pad_buffers_to_multiple` — [`L47`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L47) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
  - `ra2a` — [`L46`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L46)
  - `safety_factor` — [`L45`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L45) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
  - `use_pipelined_ra2a_barriers` — [`L49`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L49)
  - `use_scheduling_groups` — [`L48`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L48)
- protocol/private: `__post_init__`[`L54`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L54)
- uses (calls/refs, reference-scoped): [`RaggedAllToAllCallable`](moe_lib.md#RaggedAllToAllCallable), [`SPARSECORE_PAD_SIZE`](moe_lib.md#SPARSECORE_PAD_SIZE)
- used by: [`moe_ffn`](../model_lib.md#MoEFeedForward.moe_ffn), [`_overlap_fn`](moe_lib.md#_overlap_fn), [`compute_meta`](moe_lib.md#_create_pipelined_ag_moe.compute_meta), [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.prepare_fn), [`_create_pipelined_ra2a_moe`](moe_lib.md#_create_pipelined_ra2a_moe), [`compute_meta`](moe_lib.md#_create_pipelined_ra2a_moe.compute_meta), [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.prepare_fn), [`_compute_chunked_ra2a`](moe_lib.md#_compute_chunked_ra2a), [`run_moe_pipelined_shard_map`](moe_lib.md#run_moe_pipelined_shard_map), [`_create_pipelined_ag_moe`](moe_lib.md#_create_pipelined_ag_moe), [`finalize_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.finalize_fn), [`finalize_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.finalize_fn), [`run_pipeline`](moe_lib.md#run_moe_pipelined_shard_map.make_pipeline.run_pipeline), [`finalize_fn`](moe_lib.md#_create_pipelined_ag_moe.load_fn.finalize_fn), [`make_pipeline`](moe_lib.md#run_moe_pipelined_shard_map.make_pipeline)

### `RaggedAllToAllCallable`  ·  implements/extends Protocol
- def: [`simply/utils/moe_lib.py:26`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L26)
- signature: `class RaggedAllToAllCallable(Protocol):`
- protocol/private: `__call__`[`L27`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L27)
- used by: [`surround_compute_with_ra2a`](moe_lib.md#surround_compute_with_ra2a), [`ra2a`](moe_lib.md#PipelinedMoEConfig.ra2a)

### `_MoEMethods`
- def: [`simply/utils/moe_lib.py:114`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L114) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- signature: `class _MoEMethods:`
- members:
  - `compute_fn` — [`L120`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L120)
  - `compute_meta` — [`L115`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L115)
  - `load_fn` — [`L119`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L119)
  - `unload_fn` — [`L124`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L124)
- uses (calls/refs, reference-scoped): [`MoEMetaAG`](moe_lib.md#MoEMetaAG), [`MoEMetaRA2A`](moe_lib.md#MoEMetaRA2A)
- used by: [`_overlap_fn`](moe_lib.md#_overlap_fn), [`_create_pipelined_ra2a_moe`](moe_lib.md#_create_pipelined_ra2a_moe), [`run_moe_pipelined_shard_map`](moe_lib.md#run_moe_pipelined_shard_map), [`_create_pipelined_ag_moe`](moe_lib.md#_create_pipelined_ag_moe), [`run_pipeline`](moe_lib.md#run_moe_pipelined_shard_map.make_pipeline.run_pipeline)

### `_RA2AMeta`
- def: [`simply/utils/moe_lib.py:61`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L61) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- doc: Holds sizes and offsets for ragged all-to-all communication.
- signature: `class _RA2AMeta:`
- members:
  - `input_offsets` — [`L63`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L63)
  - `output_offsets` — [`L65`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L65)
  - `recv_sizes` — [`L66`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L66)
  - `send_sizes` — [`L64`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L64)
- used by: [`compute_meta`](moe_lib.md#_create_pipelined_ra2a_moe.compute_meta), [`prepare_fn`](moe_lib.md#_create_pipelined_ra2a_moe.load_fn.prepare_fn), [`_compute_chunked_ra2a`](moe_lib.md#_compute_chunked_ra2a), [`finalize_fn`](moe_lib.md#_create_pipelined_ra2a_moe.unload_fn.finalize_fn), [`preamble`](moe_lib.md#MoEMetaRA2A.preamble), [`epilogue`](moe_lib.md#MoEMetaRA2A.epilogue), [`ra2a_with_gather`](moe_lib.md#ra2a_with_gather)

### `_RDMACopy`
- def: [`simply/utils/moe_lib.py:128`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L128)
- signature: `class _RDMACopy:`
- members:
  - `copy` — [`L129`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L129)
  - `start` — [`L130`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L130)
  - `wait` — [`L131`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L131)
- uses (calls/refs, reference-scoped): [`AsyncCopyDescriptor`](moe_lib.md#AsyncCopyDescriptor)
- used by: [`_ra2a_3d_kernel_async`](moe_lib.md#_ra2a_3d_kernel_async), [`make_rdma`](moe_lib.md#_ra2a_3d_kernel_async.make_rdma), [`make_dma`](moe_lib.md#_ra2a_3d_kernel_async.make_dma)

## Functions
- `_ag_tokens(x, *, axis_name: str, axis: int, tiled: bool)` — [`L781`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L781)
- `_compute_chunked_ra2a(all_sizes: jax.Array, shard_idxs: jax.Array, *, num_shards: int, shard_idx: int | jax.Array, axis_name: str, num_experts: int, config: PipelinedMoEConfig)` — [`L507`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L507) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `_create_pipelined_ag_moe(compute_block: ComputeBlockCallable | None = None, *, axis_name: str, experts_per_tok: int, num_experts: int, config: PipelinedMoEConfig = PipelinedMoEConfig())` — [`L797`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L797) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `_create_pipelined_ra2a_moe(compute_block: ComputeBlockCallable | None = None, *, axis_name: str, experts_per_tok: int, num_experts: int, config: PipelinedMoEConfig = PipelinedMoEConfig())` — [`L614`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L614) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `_final_barrier(second_barrier)` — [`L261`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L261)
- `_overlap_fn(y1, y2, meta1, meta2, meta3, x_next, *extra_args, axis_name: str, moe_methods: _MoEMethods | Any, i: int, splits: int, config: PipelinedMoEConfig, metrics: dict[str, Any] | None = None)` — [`L909`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L909) — Overlap stages 1 & 2 communication, with stage 2 computation. — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `_ra2a_3d_kernel_async(src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems, *, axis_name: str, start: bool = True, multiple: int = 1)` — [`L202`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L202)
- `_ra2a_split(payloads, args)` — [`L407`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L407)
- `_rs_tokens(x, *, axis_name: str, scatter_dimension: int, tiled: bool)` — [`L786`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L786)
- `_update_fn(i, local_expert_idxs)` — [`L549`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L549)
- `compute_fn(y: jax.Array, meta: MoEMetaRA2A, *args)` — [`L711`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L711) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `compute_fn(y: jax.Array, meta: MoEMetaAG, *args)` — [`L871`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L871)
- `compute_local_permute_metadata(group_idx: jax.Array, groups: int, group_counts: jax.Array | None = None)` — [`L152`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L152) — Compute metadata for sorting tokens according to group_idx.
- `compute_meta(shard_idxs: jax.Array, scales: jax.Array | None = None)` — [`L628`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L628)
- `compute_meta(shard_idxs: jax.Array, scales: jax.Array | None = None)` — [`L807`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L807)
- `finalize_fn(y, meta: MoEMetaRA2A)` — [`L691`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L691)
- `finalize_fn(x_sort, meta: MoEMetaRA2A)` — [`L749`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L749)
- `finalize_fn(all_x, meta: MoEMetaAG)` — [`L856`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L856)
- `finalize_fn(y, meta: MoEMetaAG)` — [`L894`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L894)
- `fn(payloads, args)` — [`L352`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L352)
- `fn(payloads, args)` — [`L370`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L370)
- `load_fn(metrics: dict[str, Any] | None = None)` — [`L658`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L658)
- `load_fn(metrics: dict[str, Any] | None = None)` — [`L845`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L845)
- `make_dma()` — [`L214`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L214)
- `make_pipeline(config: PipelinedMoEConfig)` — [`L990`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L990)
- `make_ra2a_3d(axis_name: str = "x", multiple: int = 1)` — [`L274`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L274) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `make_rdma(other_id, send: bool = True)` — [`L222`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L222)
- `maybe_pad_size(size: int, pad_to_multiple: int | None)` — [`L146`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L146)
- `maybe_tpu_sublane_size()` — [`L139`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L139)
- `prepare_fn(x, meta: MoEMetaRA2A)` — [`L659`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L659)
- `prepare_fn(y, meta: MoEMetaRA2A)` — [`L720`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L720)
- `prepare_fn(x, meta: MoEMetaAG)` — [`L846`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L846)
- `prepare_fn(y, meta: MoEMetaAG)` — [`L884`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L884)
- `ra2a_kernel_start(src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems)` — [`L283`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L283)
- `ra2a_kernel_wait(src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, sems, dst_ref)` — [`L320`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L320)
- `ra2a_split(payloads, args)` — [`L440`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L440)
- `ra2a_split_bwd(res, g)` — [`L450`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L450) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `ra2a_split_fwd(payloads, args)` — [`L443`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L443) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `ra2a_with_gather(all_sizes: jax.Array, num_shards: int, shard_idx: jax.Array)` — [`L588`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L588)
- `run_moe_pipelined_shard_map(shard_idxs: jax.Array, x: jax.Array, *extra_args, scales: jax.Array | None = None, compute_block: ComputeBlockCallable | None = None, axis_name: str, experts_per_tok: int, num_experts: int, splits: int = 1, config: PipelinedMoEConfig = PipelinedMoEConfig(), metrics: dict[str, Any] | None = None)` — [`L955`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L955) — Execute a piplined MoE layer in full assuming expert axis is manual. — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `run_pipeline(x_, scales_, shard_idxs_, ra2a_metas=None)` — [`L991`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L991)
- `start(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, collective_id=_RA2A_COLLECTIVE_ID)` — [`L277`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L277)
- `start_fn(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, collective_id)` — [`L391`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L391)
- `surround_compute_with_ag(axis_name: str, compute_fn, use_barriers: bool = False, *, collective_methods: list[Callable])` — [`L347`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L347)
- `surround_compute_with_ra2a(axis_name: str, compute_fn, multiple: int = 1, ra2a: RaggedAllToAllCallable | None = None, use_barriers: bool = False)` — [`L365`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L365) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `unique_gather(x: jax.Array, idx: jax.Array, inv_idx: jax.Array | None = None, empty_scatter: bool = True)` — [`L167`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L167) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `unique_gather_bwd(empty_scatter: bool, res: tuple[Any, jax.Array, jax.Array], g: jax.Array)` — [`L180`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L180)
- `unique_gather_fwd(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, empty_scatter: bool)` — [`L174`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L174)
- `unload_fn()` — [`L719`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L719)
- `unload_fn()` — [`L883`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L883)
- `wait(future, collective_id=0)` — [`L315`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L315)
- `wait_fn(future, collective_id)` — [`L401`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L401)

## Module values
- `AG_PIPELINE_COLLECTIVES` — [`L791`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L791)
- `AsyncCopyDescriptor` — [`L23`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L23)
- `SENTINEL_VALUE` — [`L21`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L21) — documented in [simply-utils-moe_lib](../../../concepts/simply-utils-moe_lib.md)
- `SPARSECORE_PAD_SIZE` — [`L22`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L22)
- `_RA2A_COLLECTIVE_ID` — [`L270`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L270)
- `_SCHEDULING_GROUP_ID` — [`L906`](../../../../../../raw/code/simply/simply/utils/moe_lib.py#L906)

