---
title: 'Module: python/sgl_jax/srt/managers/communication.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/communication.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.communication`/
symbols:
  CommunicationBackend.send_pyobj: CommunicationBackend#send_pyobj().
  CommunicationBackend.recv_requests: CommunicationBackend#recv_requests().
  CommunicationBackend: CommunicationBackend#
  QueueBackend: QueueBackend#
  QueueBackend.recv_requests: QueueBackend#recv_requests().
  QueueBackend.send_pyobj: QueueBackend#send_pyobj().
  QueueBackend._in_queue: QueueBackend#_in_queue.
  QueueBackend._out_queue: QueueBackend#_out_queue.
  logger: logger.
  QueueBackend.__init__: QueueBackend#__init__().
  QueueBackend.wait_for_new_requests: QueueBackend#wait_for_new_requests().
---
# Module: [`python/sgl_jax/srt/managers/communication.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py)

## Classes
### `CommunicationBackend`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/managers/communication.py:10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L10)
- doc: Abstract communication backend.
- signature: `class CommunicationBackend(ABC):`
- members:
  - `recv_requests(self)` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L14) — Receive requests (non-blocking, return empty list if none).
  - `send_pyobj(self, result: Any)` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L19) — Send result to other components.
- uses (calls/refs, reference-scoped): [`QueueBackend`](communication.md#QueueBackend), [`recv_requests`](communication.md#QueueBackend.recv_requests), [`send_pyobj`](communication.md#QueueBackend.send_pyobj)
- used by: [`stream_output_generation`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.stream_output_generation), [`abort_request`](scheduler.md#Scheduler.abort_request), [`result_queue`](scheduler.md#Scheduler.result_queue), [`update_running_batch`](scheduler.md#Scheduler.update_running_batch), [`event_loop_normal_disagg_decode`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode), [`event_loop_normal_disagg_prefill`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill), [`event_loop_normal`](scheduler.md#Scheduler.event_loop_normal), [`_abort_decode_request`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._abort_decode_request), [`process_input_requests`](scheduler.md#Scheduler.process_input_requests), [`__init__`](scheduler.md#Scheduler.__init__), [`QueueBackend`](communication.md#QueueBackend)

### `QueueBackend`  ·  implements/extends CommunicationBackend
- def: [`python/sgl_jax/srt/managers/communication.py:24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L24)
- doc: Queue-based communication for Stage mode.
- signature: `class QueueBackend(CommunicationBackend):`
- members:
  - `recv_requests(self)` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L31)
  - `send_pyobj(self, result: Any)` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L44)
  - `wait_for_new_requests(self, timeout: float = 0)` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L41)
- protocol/private: `__init__`[`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L27), `_in_queue`[`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L28), `_out_queue`[`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L29)
- uses (calls/refs, reference-scoped): [`CommunicationBackend`](communication.md#CommunicationBackend)
- used by: [`send_pyobj`](communication.md#CommunicationBackend.send_pyobj), [`recv_requests`](communication.md#CommunicationBackend.recv_requests), [`CommunicationBackend`](communication.md#CommunicationBackend)

## Module values
- `logger` — [`L7`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/communication.py#L7)

