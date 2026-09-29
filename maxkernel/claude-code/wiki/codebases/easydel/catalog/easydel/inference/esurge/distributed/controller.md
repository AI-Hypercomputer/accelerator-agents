---
title: 'Module: easydel/inference/esurge/distributed/controller.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/distributed/controller.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.distributed.controller`/
symbols:
  DistributedController.start: DistributedController#start().
  DistributedController.dispatch_step: DistributedController#dispatch_step().
  DistributedController.shutdown: DistributedController#shutdown().
  DistributedController.verify_step: DistributedController#verify_step().
  DistributedController._worker_clients: DistributedController#_worker_clients.
  DistributedController._worker_server: DistributedController#_worker_server.
  DistributedController.rank: DistributedController#rank.
  StepDispatch.step_id: StepDispatch#step_id.
  DistributedController.is_leader: DistributedController#is_leader().
  DistributedController.is_worker: DistributedController#is_worker().
  StepDispatch: StepDispatch#
  DistributedController: DistributedController#
  DistributedController.world_size: DistributedController#world_size.
  DistributedController._discovery: DistributedController#_discovery.
  resolve_distributed_role: resolve_distributed_role().
  DistributedController.enabled: DistributedController#enabled.
  DistributedController.config_fingerprint: DistributedController#config_fingerprint.
  DistributedController._started: DistributedController#_started.
  DistributedController.has_remote_workers: DistributedController#has_remote_workers().
  DistributedController.role: DistributedController#role.
  DistributedController.service_name: DistributedController#service_name.
  DistributedController.control_port: DistributedController#control_port.
  DistributedController.advertise_addr: DistributedController#advertise_addr.
  DistributedController.auth_token: DistributedController#auth_token.
  DistributedController._execute_step: DistributedController#_execute_step.
  DistributedController._step_counter: DistributedController#_step_counter.
  DistributedController.control_bind_host: DistributedController#control_bind_host.
  DistributedController.step_timeout_s: DistributedController#step_timeout_s.
  DistributedController.connect_timeout_s: DistributedController#connect_timeout_s.
  DistributedController.verify_sampling_digest: DistributedController#verify_sampling_digest.
  DistributedController.__init__: DistributedController#__init__().
---
# Module: [`easydel/inference/esurge/distributed/controller.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py)

## Classes
### `DistributedController`
- def: [`easydel/inference/esurge/distributed/controller.py:99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L99)
- doc: Coordinates leader/worker lockstep execution for distributed eSurge serving.
- signature: `class DistributedController:`
- members:
  - `__init__(self, *, enabled: bool, role: str, rank: int, world_size: int, service_name: str | None, control_port: int, control_bind_host: str, advertise_addr: str | None, auth_token: str, step_timeout_s: float, connect_timeout_s: float, verify_sampling_digest: bool, config_fingerprint: str, execute_step: Any | None)` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L137) — Initialize the distributed controller with cluster configuration.
  - `dispatch_step(self, scheduler_output: Any)` — [`L316`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L316) — Fan-out a scheduler output to all connected workers.
  - `has_remote_workers(self)` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L208) — Whether the leader has established connections to remote workers.
  - `is_leader(self)` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L198) — Whether this controller is the cluster leader.
  - `is_worker(self)` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L203) — Whether this controller is a cluster worker.
  - `shutdown(self)` — [`L443`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L443) — Tear down the distributed control-plane.
  - `start(self)` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L212) — Initialize the distributed cluster.
  - `verify_step(self, dispatch: StepDispatch | None, model_output: Any)` — [`L379`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L379) — Collect and validate worker results for a previously dispatched step.
  - `advertise_addr` — [`L183`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L183)
  - `auth_token` — [`L184`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L184)
  - `config_fingerprint` — [`L188`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L188)
  - `connect_timeout_s` — [`L186`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L186)
  - `control_bind_host` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L182)
  - `control_port` — [`L181`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L181)
  - `enabled` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L176)
  - `rank` — [`L178`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L178)
  - `role` — [`L177`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L177)
  - `service_name` — [`L180`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L180)
  - `step_timeout_s` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L185)
  - `verify_sampling_digest` — [`L187`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L187)
  - `world_size` — [`L179`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L179)
- protocol/private: `_discovery`[`L193`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L193), `_execute_step`[`L189`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L189), `_started`[`L191`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L191), `_step_counter`[`L192`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L192), `_worker_clients`[`L195`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L195), `_worker_server`[`L194`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L194)
- uses (calls/refs, reference-scoped): [`logger`](../logger.md#logger), [`start`](worker_server.md#WorkerControlServer.start), [`begin_step`](leader_client.md#WorkerRpcClient.begin_step), [`stop`](worker_server.md#WorkerControlServer.stop), [`resolve_service_hosts`](discovery.md#resolve_service_hosts), [`finish_step`](leader_client.md#WorkerRpcClient.finish_step), [`reset_connection`](leader_client.md#WorkerRpcClient.reset_connection), [`endpoint`](worker_server.md#WorkerControlServer.endpoint), [`compute_sampled_digest`](protocol.md#compute_sampled_digest), [`hello`](leader_client.md#WorkerRpcClient.hello), [`shutdown`](leader_client.md#WorkerRpcClient.shutdown), [`STATUS_OK`](protocol.md#STATUS_OK), [`close`](leader_client.md#WorkerRpcClient.close), [`step_id`](controller.md#StepDispatch.step_id), [`DiscoveryResult`](discovery.md#DiscoveryResult), [`StepDispatch`](controller.md#StepDispatch), [`WorkerRpcClient`](leader_client.md#WorkerRpcClient), [`hosts`](discovery.md#DiscoveryResult.hosts), [`WorkerControlServer`](worker_server.md#WorkerControlServer)
- used by: [`_distributed_config_fingerprint`](../esurge_engine.md#eSurge._distributed_config_fingerprint), [`__del__`](../esurge_engine.md#eSurge.__del__), [`_distributed_controller`](../esurge_engine.md#eSurge._distributed_controller)

### `StepDispatch`
- def: [`easydel/inference/esurge/distributed/controller.py:55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L55)
- doc: Lightweight token representing one distributed step dispatched to workers.
- signature: `class StepDispatch:`
- members:
  - `step_id` — [`L66`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L66)
- used by: [`dispatch_step`](controller.md#DistributedController.dispatch_step), [`verify_step`](controller.md#DistributedController.verify_step)

## Functions
- `resolve_distributed_role(role: str, rank: int)` — [`L69`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/controller.py#L69) — Resolve and validate the distributed role for a given rank.

