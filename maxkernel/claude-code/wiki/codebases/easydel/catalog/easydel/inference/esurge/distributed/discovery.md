---
title: 'Module: easydel/inference/esurge/distributed/discovery.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/distributed/discovery.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.distributed.discovery`/
symbols:
  resolve_service_hosts: resolve_service_hosts().
  DiscoveryResult: DiscoveryResult#
  DiscoveryResult.hosts: DiscoveryResult#hosts.
  DiscoveryResult.world_size: DiscoveryResult#world_size().
  DiscoveryResult.rank_to_host: DiscoveryResult#rank_to_host().
  _host_sort_key: _host_sort_key().
---
# Module: [`easydel/inference/esurge/distributed/discovery.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py)

## Classes
### `DiscoveryResult`
- def: [`easydel/inference/esurge/distributed/discovery.py:47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py#L47)
- doc: Immutable result of DNS service discovery containing the resolved host list.
- signature: `class DiscoveryResult:`
- members:
  - `rank_to_host(self)` — [`L66`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py#L66) — Return a mapping from rank index to host address.
  - `world_size(self)` — [`L61`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py#L61) — Return the total number of hosts in the cluster.
  - `hosts` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py#L58)
- used by: [`start`](controller.md#DistributedController.start), [`resolve_service_hosts`](discovery.md#resolve_service_hosts), [`_discovery`](controller.md#DistributedController._discovery)

## Functions
- `_host_sort_key(host: str)` — [`L71`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py#L71) — Return a sort key that orders IP addresses before hostnames.
- `resolve_service_hosts(service_name: str, world_size: int | None = None)` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/discovery.py#L91) — Resolve a DNS service name into a sorted, de-duplicated host list.

