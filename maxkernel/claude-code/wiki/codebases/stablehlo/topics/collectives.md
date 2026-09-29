---
title: "collectives — sharding communication ops"
type: concept
tags: [stablehlo, collectives, sharding, all_reduce, all_gather, reduce_scatter, reference]
created: 2026-07-04
updated: 2026-07-04
---

The **collective ops** are StableHLO's cross-device communication primitives — the ops that implement sharding (data / tensor / pipeline parallelism). In an `xla_dump` or xprof trace these are the `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `collective-permute`, `collective-broadcast` entries; their cost (bytes moved × topology) is what you weigh against compute when hunting overlap gaps. This page decodes each op and the shared **process-group** model that sets who talks to whom.

## The six collectives

There are exactly six ([collective-ops](src:docs/spec.md#collective-ops)); all split the process grid into **process groups** and compute jointly within each group, independently across groups.

| Op | Semantics | Shape effect | TPU-perf read |
|---|---|---|---|
| [all_gather](src:docs/spec.md#all_gather) | Concatenate each process's operand along `all_gather_dim` | grows `all_gather_dim` × group size | Materializes a full replica of a sharded tensor; FSDP weight/param gather. |
| [all_reduce](src:docs/spec.md#all_reduce) | Apply reduction `computation` across processes | shape unchanged | Gradient sync (data parallel); the classic overlap-with-backprop target. |
| [reduce_scatter](src:docs/spec.md#reduce_scatter) | `all_reduce` then split & scatter along `scatter_dimension` | shrinks `scatter_dimension` ÷ group size | Fused reduce+shard — cheaper than all_reduce+dynamic_slice. |
| [all_to_all](src:docs/spec.md#all_to_all) | Split along `split_dimension`, scatter parts, concat along `concat_dimension` | moves size between the two dims | MoE dispatch/combine; resharding across mesh axes. |
| [collective_permute](src:docs/spec.md#collective_permute) | Point-to-point send along `source_target_pairs` | shape unchanged | Pipeline-parallel ring shifts; halo exchange. |
| [collective_broadcast](src:docs/spec.md#collective_broadcast) | Send source process's value to the group | shape unchanged | One-to-many replication. |

## The process-group model (shared cost driver)

All collectives select their process groups from the StableHLO process grid via `replica_groups` (or a `ReplicaGroupMeshAxes`) plus `channel_id` / `use_global_device_ids` ([replica-groups](src:docs/spec.md#replica-groups)). Four grouping strategies decide the Cartesian product of replicas × partitions:

- **[cross_replica](src:docs/spec.md#cross_replica)** — `channel_id <= 0`; groups replicas, product over partitions.
- **[cross_partition](src:docs/spec.md#cross_partition)** — `channel_id > 0` (all_to_all / broadcast / permute); groups partitions, product over replicas.
- **[cross_replica_and_partition](src:docs/spec.md#cross_replica_and_partition)** — `channel_id > 0`, `use_global_device_ids=false`; groups replicas but spans partitions too.
- **[flattened_ids](src:docs/spec.md#flattened_ids)** — `channel_id > 0`, `use_global_device_ids=true`; explicit `replica_id * num_partitions + partition_id` groups.

Cross-partition communication needs a channel (positive `channel_id`); cross-replica does not ([collective-ops](src:docs/spec.md#collective-ops)). Collectives may introduce a synchronization barrier within each group.

> [!inferred]
> Reading a dump for perf: the `replica_groups={{...}}` attribute on a collective tells you the group *size* — that size, times the per-process bytes, times the topology bandwidth, is the first-order latency. A collective whose group spans a slow mesh axis (e.g. inter-slice) is the usual overlap bottleneck. `reduce_scatter` and `all_gather` over the same groups are the two halves of a sharded matmul; seeing both back-to-back around a `dot` is normal FSDP/tensor-parallel structure. `channel_id > 0` marks the cross-partition (SPMD-partitioned) collectives.

## See also

- [dot_general](dot_general.md) — sharded matmuls emit all_gather / reduce_scatter around the dot
- [scatter-gather](scatter-gather.md) — `all_to_all` underlies MoE dispatch, distinct from the local `gather`/`scatter` ops
- Source: [spec.md](../sources/spec.md)

## Sources

- `raw/code/stablehlo/docs/spec.md`
