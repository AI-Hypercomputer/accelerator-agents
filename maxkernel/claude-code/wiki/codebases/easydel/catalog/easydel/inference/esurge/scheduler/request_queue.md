---
title: 'Module: easydel/inference/esurge/scheduler/request_queue.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/scheduler/request_queue.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.scheduler.request_queue`/
symbols:
  create_request_queue: create_request_queue().
  PriorityRequestQueue._heap: PriorityRequestQueue#_heap.
  RequestQueue: RequestQueue#
  RequestQueue.pop_request: RequestQueue#pop_request().
  RequestQueue.prepend_request: RequestQueue#prepend_request().
  PriorityRequestQueue.add_request: PriorityRequestQueue#add_request().
  RequestQueue.remove_requests: RequestQueue#remove_requests().
  RequestQueue.add_request: RequestQueue#add_request().
  RequestQueue.peek_request: RequestQueue#peek_request().
  RequestQueue.prepend_requests: RequestQueue#prepend_requests().
  FCFSRequestQueue: FCFSRequestQueue#
  RequestQueue.remove_request: RequestQueue#remove_request().
  SchedulingPolicy: SchedulingPolicy#
  PriorityRequestQueue: PriorityRequestQueue#
  PriorityRequestQueue.pop_request: PriorityRequestQueue#pop_request().
  PriorityRequestQueue.peek_request: PriorityRequestQueue#peek_request().
  PriorityRequestQueue.prepend_request: PriorityRequestQueue#prepend_request().
  PriorityRequestQueue.prepend_requests: PriorityRequestQueue#prepend_requests().
  PriorityRequestQueue.remove_request: PriorityRequestQueue#remove_request().
  PriorityRequestQueue.remove_requests: PriorityRequestQueue#remove_requests().
  PriorityRequestQueue.__iter__: PriorityRequestQueue#__iter__().
  SchedulingPolicy.PRIORITY: SchedulingPolicy#PRIORITY.
  RequestQueue.__iter__: RequestQueue#__iter__().
  RequestQueue.__reversed__: RequestQueue#__reversed__().
  FCFSRequestQueue.add_request: FCFSRequestQueue#add_request().
  FCFSRequestQueue.pop_request: FCFSRequestQueue#pop_request().
  FCFSRequestQueue.peek_request: FCFSRequestQueue#peek_request().
  FCFSRequestQueue.prepend_request: FCFSRequestQueue#prepend_request().
  FCFSRequestQueue.prepend_requests: FCFSRequestQueue#prepend_requests().
  FCFSRequestQueue.remove_request: FCFSRequestQueue#remove_request().
  FCFSRequestQueue.remove_requests: FCFSRequestQueue#remove_requests().
  FCFSRequestQueue.__iter__: FCFSRequestQueue#__iter__().
  FCFSRequestQueue.__reversed__: FCFSRequestQueue#__reversed__().
  PriorityRequestQueue.__bool__: PriorityRequestQueue#__bool__().
  PriorityRequestQueue.__len__: PriorityRequestQueue#__len__().
  PriorityRequestQueue.__reversed__: PriorityRequestQueue#__reversed__().
  SchedulingPolicy.FCFS: SchedulingPolicy#FCFS.
  RequestQueue.__bool__: RequestQueue#__bool__().
  RequestQueue.__len__: RequestQueue#__len__().
  FCFSRequestQueue.__bool__: FCFSRequestQueue#__bool__().
  FCFSRequestQueue.__len__: FCFSRequestQueue#__len__().
  PriorityRequestQueue.__init__: PriorityRequestQueue#__init__().
---
# Module: [`easydel/inference/esurge/scheduler/request_queue.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py)

## Classes
### `FCFSRequestQueue`  ·  implements/extends RequestQueue, deque
- def: [`easydel/inference/esurge/scheduler/request_queue.py:296`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L296)
- doc: A first-come-first-served queue that supports deque operations.
- signature: `class FCFSRequestQueue(deque[EngineRequest], RequestQueue):`
- members:
  - `__bool__(self)` — [`L428`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L428) — Check if queue has any requests.
  - `__iter__(self)` — [`L444`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L444) — Iterate over the queue according to FCFS policy.
  - `__len__(self)` — [`L436`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L436) — Get number of requests in queue.
  - `__reversed__(self)` — [`L454`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L454) — Iterate over the queue in reverse order.
  - `add_request(self, request: EngineRequest)` — [`L315`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L315) — Add a request to the queue according to FCFS policy.
  - `peek_request(self)` — [`L345`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L345) — Peek at the next request in the queue without removing it.
  - `pop_request(self)` — [`L328`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L328) — Pop a request from the queue according to FCFS policy.
  - `prepend_request(self, request: EngineRequest)` — [`L363`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L363) — Prepend a request to the front of the queue.
  - `prepend_requests(self, requests: RequestQueue)` — [`L378`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L378) — Prepend all requests from another queue to the front of this queue.
  - `remove_request(self, request: EngineRequest)` — [`L392`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L392) — Remove a specific request from the queue.
  - `remove_requests(self, requests: Iterable[EngineRequest])` — [`L408`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L408) — Remove multiple specific requests from the queue.
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`RequestQueue`](request_queue.md#RequestQueue)
- used by: [`create_request_queue`](request_queue.md#create_request_queue), [`RequestQueue`](request_queue.md#RequestQueue), [`pop_request`](request_queue.md#RequestQueue.pop_request), [`prepend_request`](request_queue.md#RequestQueue.prepend_request), [`remove_requests`](request_queue.md#RequestQueue.remove_requests), [`add_request`](request_queue.md#RequestQueue.add_request), [`peek_request`](request_queue.md#RequestQueue.peek_request), [`prepend_requests`](request_queue.md#RequestQueue.prepend_requests), [`remove_request`](request_queue.md#RequestQueue.remove_request)

### `PriorityRequestQueue`  ·  implements/extends RequestQueue
- def: [`easydel/inference/esurge/scheduler/request_queue.py:465`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L465)
- doc: A priority queue that supports heap operations.
- signature: `class PriorityRequestQueue(RequestQueue):`
- members:
  - `__bool__(self)` — [`L618`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L618) — Check if queue has any requests.
  - `__init__(self)` — [`L486`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L486) — Initialize an empty priority queue.
  - `__iter__(self)` — [`L634`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L634) — Iterate over the queue according to priority policy.
  - `__len__(self)` — [`L626`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L626) — Get number of requests in queue.
  - `__reversed__(self)` — [`L653`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L653) — Iterate over the queue in reverse priority order.
  - `add_request(self, request: EngineRequest)` — [`L493`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L493) — Add a request to the queue according to priority policy.
  - `peek_request(self)` — [`L528`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L528) — Peek at the next request in the queue without removing it.
  - `pop_request(self)` — [`L508`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L508) — Pop a request from the queue according to priority policy.
  - `prepend_request(self, request: EngineRequest)` — [`L547`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L547) — Add a request to the queue according to priority policy.
  - `prepend_requests(self, requests: RequestQueue)` — [`L567`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L567) — Add all requests from another queue according to priority policy.
  - `remove_request(self, request: EngineRequest)` — [`L587`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L587) — Remove a specific request from the queue.
  - `remove_requests(self, requests: Iterable[EngineRequest])` — [`L602`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L602) — Remove multiple specific requests from the queue.
- protocol/private: `_heap`[`L491`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L491)
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`RequestQueue`](request_queue.md#RequestQueue), [`arrival_time`](../request.md#EngineRequest.arrival_time), [`priority`](../request.md#EngineRequest.priority)
- used by: [`create_request_queue`](request_queue.md#create_request_queue), [`RequestQueue`](request_queue.md#RequestQueue), [`pop_request`](request_queue.md#RequestQueue.pop_request), [`prepend_request`](request_queue.md#RequestQueue.prepend_request), [`remove_requests`](request_queue.md#RequestQueue.remove_requests), [`add_request`](request_queue.md#RequestQueue.add_request), [`peek_request`](request_queue.md#RequestQueue.peek_request), [`prepend_requests`](request_queue.md#RequestQueue.prepend_requests), [`remove_request`](request_queue.md#RequestQueue.remove_request)

### `RequestQueue`  ·  implements/extends ABC
- def: [`easydel/inference/esurge/scheduler/request_queue.py:92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L92)
- doc: Abstract base class for request queues.
- signature: `class RequestQueue(ABC):`
- members:
  - `__bool__(self)` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L237) — Check if queue has any requests.
  - `__iter__(self)` — [`L263`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L263) — Iterate over the queue according to the policy.
  - `__len__(self)` — [`L251`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L251) — Get number of requests in queue.
  - `__reversed__(self)` — [`L280`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L280) — Iterate over the queue in reverse order.
  - `add_request(self, request: EngineRequest)` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L116) — Add a request to the queue according to the policy.
  - `peek_request(self)` — [`L150`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L150) — Peek at the request at the front of the queue without removing it.
  - `pop_request(self)` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L132) — Pop a request from the queue according to the policy.
  - `prepend_request(self, request: EngineRequest)` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L169) — Prepend a request to the front of the queue.
  - `prepend_requests(self, requests: RequestQueue)` — [`L188`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L188) — Prepend all requests from another queue to the front of this queue.
  - `remove_request(self, request: EngineRequest)` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L204) — Remove a specific request from the queue.
  - `remove_requests(self, requests: Iterable[EngineRequest])` — [`L222`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L222) — Remove multiple specific requests from the queue.
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`add_request`](request_queue.md#PriorityRequestQueue.add_request), [`FCFSRequestQueue`](request_queue.md#FCFSRequestQueue), [`PriorityRequestQueue`](request_queue.md#PriorityRequestQueue), [`peek_request`](request_queue.md#PriorityRequestQueue.peek_request), [`pop_request`](request_queue.md#PriorityRequestQueue.pop_request), [`prepend_request`](request_queue.md#PriorityRequestQueue.prepend_request), [`prepend_requests`](request_queue.md#PriorityRequestQueue.prepend_requests), [`remove_request`](request_queue.md#PriorityRequestQueue.remove_request), [`remove_requests`](request_queue.md#PriorityRequestQueue.remove_requests), [`add_request`](request_queue.md#FCFSRequestQueue.add_request), [`peek_request`](request_queue.md#FCFSRequestQueue.peek_request), [`pop_request`](request_queue.md#FCFSRequestQueue.pop_request), [`prepend_request`](request_queue.md#FCFSRequestQueue.prepend_request), [`prepend_requests`](request_queue.md#FCFSRequestQueue.prepend_requests), [`remove_request`](request_queue.md#FCFSRequestQueue.remove_request), [`remove_requests`](request_queue.md#FCFSRequestQueue.remove_requests)
- used by: [`schedule`](scheduler.md#Scheduler.schedule), [`update_from_output`](scheduler.md#Scheduler.update_from_output), [`finish_requests`](scheduler.md#Scheduler.finish_requests), [`create_request_queue`](request_queue.md#create_request_queue), [`add_request`](scheduler.md#Scheduler.add_request), [`FCFSRequestQueue`](request_queue.md#FCFSRequestQueue), [`PriorityRequestQueue`](request_queue.md#PriorityRequestQueue), [`prepend_requests`](request_queue.md#PriorityRequestQueue.prepend_requests), [`prepend_requests`](request_queue.md#FCFSRequestQueue.prepend_requests)

### `SchedulingPolicy`  ·  implements/extends Enum
- def: [`easydel/inference/esurge/scheduler/request_queue.py:67`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L67)
- doc: Enum for scheduling policies.
- signature: `class SchedulingPolicy(Enum):`
- members:
  - `FCFS` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L85)
  - `PRIORITY` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L88)
- used by: [`schedule`](scheduler.md#Scheduler.schedule), [`policy`](scheduler.md#Scheduler.policy), [`create_request_queue`](request_queue.md#create_request_queue)

## Functions
- `create_request_queue(policy: SchedulingPolicy)` — [`L665`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/request_queue.py#L665) — Create request queue based on scheduling policy.

