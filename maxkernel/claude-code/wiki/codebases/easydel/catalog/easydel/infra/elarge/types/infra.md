---
title: 'Module: easydel/infra/elarge/types/infra.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/infra.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.infra`/
symbols:
  PlatformCfg: PlatformCfg#
  ShardingCfg: ShardingCfg#
  ShardingCfg.partition_rules: ShardingCfg#partition_rules.
  PlatformCfg.backend: PlatformCfg#backend.
  ShardingCfg.axis_dims: ShardingCfg#axis_dims.
  ShardingCfg.dcn_axis_dims: ShardingCfg#dcn_axis_dims.
  ShardingCfg.axis_names: ShardingCfg#axis_names.
  ShardingCfg.partition_axis: ShardingCfg#partition_axis.
  ShardingCfg.shard_fns: ShardingCfg#shard_fns.
  ShardingCfg.auto_shard_model: ShardingCfg#auto_shard_model.
  ShardingCfg.use_ring_of_experts: ShardingCfg#use_ring_of_experts.
  ShardingCfg.fsdp_is_ep_bound: ShardingCfg#fsdp_is_ep_bound.
  ShardingCfg.sp_is_ep_bound: ShardingCfg#sp_is_ep_bound.
  PlatformCfg.platform: PlatformCfg#platform.
---
# Module: [`easydel/infra/elarge/types/infra.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py)

## Classes
### `PlatformCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/infra.py:67`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L67)
- doc: Platform and backend configuration for hardware selection.
- signature: `class PlatformCfg(TypedDict, total=False):`
- members:
  - `backend` — [`L76`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L76)
  - `platform` — [`L77`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L77)
- uses (calls/refs, reference-scoped): [`EasyDeLBackends`](../../etils.md#EasyDeLBackends), [`EasyDeLPlatforms`](../../etils.md#EasyDeLPlatforms)
- used by: [`platform`](root.md#eLMConfig.platform)

### `ShardingCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/infra.py:34`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L34)
- doc: Model sharding configuration for distributed training and inference.
- signature: `class ShardingCfg(TypedDict, total=False):`
- members:
  - `auto_shard_model` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L60)
  - `axis_dims` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L55)
  - `axis_names` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L57)
  - `dcn_axis_dims` — [`L56`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L56)
  - `fsdp_is_ep_bound` — [`L63`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L63)
  - `partition_axis` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L58)
  - `partition_rules` — [`L61`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L61)
  - `shard_fns` — [`L59`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L59)
  - `sp_is_ep_bound` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L64)
  - `use_ring_of_experts` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/infra.py#L62)
- uses (calls/refs, reference-scoped): [`PartitionRules`](aliases.md#PartitionRules)
- used by: [`sharding`](root.md#eLMConfig.sharding)

