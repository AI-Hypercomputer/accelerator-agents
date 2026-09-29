---
title: 'Module: python/sgl_jax/srt/managers/scheduler.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/scheduler.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.scheduler`/
symbols:
  Scheduler.handle_generate_request: Scheduler#handle_generate_request().
  Scheduler.get_new_batch_prefill: Scheduler#get_new_batch_prefill().
  Scheduler.run_batch: Scheduler#run_batch().
  Scheduler.abort_request: Scheduler#abort_request().
  Scheduler.result_queue: Scheduler#result_queue.
  Scheduler.get_internal_state: Scheduler#get_internal_state().
  Scheduler._run_speculative_batch: Scheduler#_run_speculative_batch().
  Scheduler.running_batch: Scheduler#running_batch.
  Scheduler.flush_cache: Scheduler#flush_cache().
  Scheduler.update_running_batch: Scheduler#update_running_batch().
  Scheduler._request_dispatcher: Scheduler#_request_dispatcher.
  Scheduler.get_next_batch_to_run: Scheduler#get_next_batch_to_run().
  Scheduler: Scheduler#
  Scheduler.tree_cache: Scheduler#tree_cache.
  Scheduler.dp_size: Scheduler#dp_size.
  Scheduler.spec_algorithm: Scheduler#spec_algorithm.
  logger: logger.
  Scheduler.grammar_queue: Scheduler#grammar_queue.
  Scheduler.pause_generation: Scheduler#pause_generation().
  Scheduler.enable_overlap: Scheduler#enable_overlap.
  Scheduler.set_internal_state: Scheduler#set_internal_state().
  Scheduler.check_memory: Scheduler#check_memory().
  Scheduler.mesh: Scheduler#mesh.
  Scheduler.event_loop_normal: Scheduler#event_loop_normal().
  Scheduler.process_batch_result: Scheduler#process_batch_result().
  Scheduler.token_to_kv_pool_allocator: Scheduler#token_to_kv_pool_allocator.
  run_scheduler_process: run_scheduler_process().
  GenerationBatchResult.next_draft_input: GenerationBatchResult#next_draft_input.
  Scheduler._comm_backend: Scheduler#_comm_backend.
  Scheduler._estimate_req_tokens: Scheduler#_estimate_req_tokens().
  Scheduler.select_dp_for_request: Scheduler#select_dp_for_request().
  Scheduler.move_ready_grammar_requests: Scheduler#move_ready_grammar_requests().
  Scheduler._can_flush_cache: Scheduler#_can_flush_cache().
  Scheduler._spec_multi_layer: Scheduler#_spec_multi_layer.
  Scheduler.last_batch: Scheduler#last_batch.
  Scheduler._get_dp_load_snapshot: Scheduler#_get_dp_load_snapshot().
  Scheduler.processor: Scheduler#processor.
  Scheduler.draft_worker: Scheduler#draft_worker.
  run_scheduler_loop_thread_after_create: run_scheduler_loop_thread_after_create().
  Scheduler.watchdog_last_time: Scheduler#watchdog_last_time.
  dispatch_scheduler_event_loop: dispatch_scheduler_event_loop().
  Scheduler.node_rank: Scheduler#node_rank.
  Scheduler.req_to_token_pool: Scheduler#req_to_token_pool.
  GenerationBatchResult: GenerationBatchResult#
  Scheduler.chunked_reqs: Scheduler#chunked_reqs.
  Scheduler.waiting_queue: Scheduler#waiting_queue.
  Scheduler._get_swa_token_info: Scheduler#_get_swa_token_info().
  Scheduler.server_args: Scheduler#server_args.
  Scheduler.recv_requests: Scheduler#recv_requests().
  Scheduler.random_seed: Scheduler#random_seed.
  Scheduler.flush_cache_wrapped: Scheduler#flush_cache_wrapped().
  Scheduler.tp_worker: Scheduler#tp_worker.
  Scheduler.set_next_batch_sampling_info_done: Scheduler#set_next_batch_sampling_info_done().
  Scheduler.model_config: Scheduler#model_config.
  Scheduler._dp_load_and_eligible: Scheduler#_dp_load_and_eligible().
  Scheduler._get_token_info: Scheduler#_get_token_info().
  Scheduler.init_new_token_ratio: Scheduler#init_new_token_ratio.
  Scheduler._cached_prefix_len: Scheduler#_cached_prefix_len().
  Scheduler._select_cache_aware_dp: Scheduler#_select_cache_aware_dp().
  Scheduler.new_token_ratio: Scheduler#new_token_ratio.
  Scheduler._extract_dp_output_ids: Scheduler#_extract_dp_output_ids().
  Scheduler.cur_batch: Scheduler#cur_batch.
  Scheduler.subscriber_sync: Scheduler#subscriber_sync.
  Scheduler.process_input_requests: Scheduler#process_input_requests().
  Scheduler._add_request_to_queue: Scheduler#_add_request_to_queue().
  Scheduler.num_subscribers: Scheduler#num_subscribers.
  Scheduler.sync_pub: Scheduler#sync_pub().
  Scheduler.broadcast_pyobj: Scheduler#broadcast_pyobj().
  Scheduler.send_to_detokenizer: Scheduler#send_to_detokenizer.
  GenerationBatchResult.logits_output: GenerationBatchResult#logits_output.
  Scheduler.send_to_tokenizer: Scheduler#send_to_tokenizer.
  GenerationBatchResult.accept_lens: GenerationBatchResult#accept_lens.
  Scheduler.disagg_shutdown: Scheduler#disagg_shutdown.
  Scheduler.is_hybrid: Scheduler#is_hybrid.
  Scheduler.sliding_window_size: Scheduler#sliding_window_size.
  Scheduler.check_tree_cache: Scheduler#check_tree_cache().
  Scheduler._current_sampling_info_owner: Scheduler#_current_sampling_info_owner().
  Scheduler.is_generation: Scheduler#is_generation.
  Scheduler.policy: Scheduler#policy.
  Scheduler.new_token_ratio_decay: Scheduler#new_token_ratio_decay.
  Scheduler.sync_pub_sub: Scheduler#sync_pub_sub().
  Scheduler._get_input_token_len: Scheduler#_get_input_token_len().
  scheduler_loop_after_create: scheduler_loop_after_create().
  Scheduler.nnodes: Scheduler#nnodes.
  Scheduler.page_size: Scheduler#page_size.
  Scheduler.subscriber: Scheduler#subscriber.
  Scheduler._batch_size: Scheduler#_batch_size().
  GenerationBatchResult.next_token_ids: GenerationBatchResult#next_token_ids.
  Scheduler.__init__: Scheduler#__init__().
  Scheduler.grammar_backend: Scheduler#grammar_backend.
  Scheduler.recv_from_rpc: Scheduler#recv_from_rpc.
  Scheduler.swa_tokens_per_layer: Scheduler#swa_tokens_per_layer.
  Scheduler.min_new_token_ratio: Scheduler#min_new_token_ratio.
  GenerationBatchResult.bid: GenerationBatchResult#bid.
  GenerationBatchResult.cache_miss_count: GenerationBatchResult#cache_miss_count.
  Scheduler.chunked_prefill_size: Scheduler#chunked_prefill_size.
  Scheduler.pending_dp_reqs: Scheduler#pending_dp_reqs.
  Scheduler._setup_jit_cache: Scheduler#_setup_jit_cache().
  Scheduler.sync_sub: Scheduler#sync_sub().
  Scheduler.run_publisher: Scheduler#run_publisher().
  Scheduler.run_subscriber: Scheduler#run_subscriber().
  Scheduler.continue_generation: Scheduler#continue_generation().
  Scheduler.publisher_sync: Scheduler#publisher_sync.
  Scheduler.per_dp_max_running_requests: Scheduler#per_dp_max_running_requests.
  Scheduler.watchdog_timeout: Scheduler#watchdog_timeout.
  Scheduler._is_spec_decode_enabled: Scheduler#_is_spec_decode_enabled().
  Scheduler._engine_paused: Scheduler#_engine_paused.
  GenerationBatchResult.extend_input_len_per_req: GenerationBatchResult#extend_input_len_per_req.
  GenerationBatchResult.extend_logprob_start_len_per_req: GenerationBatchResult#extend_logprob_start_len_per_req.
  Scheduler.recv_from_tokenizer: Scheduler#recv_from_tokenizer.
  Scheduler.is_mixed_chunk: Scheduler#is_mixed_chunk.
  Scheduler._extend_requests_to_queue: Scheduler#_extend_requests_to_queue().
  Scheduler.disagg_decode_watchdog: Scheduler#disagg_decode_watchdog.
  Scheduler.publisher: Scheduler#publisher.
  Scheduler.max_total_num_tokens: Scheduler#max_total_num_tokens.
  Scheduler.num_generated_tokens: Scheduler#num_generated_tokens.
  Scheduler.parent_process: Scheduler#parent_process.
  Scheduler._select_round_robin_dp: Scheduler#_select_round_robin_dp().
  Scheduler._select_min_running_dp: Scheduler#_select_min_running_dp().
  Scheduler.forward_ct: Scheduler#forward_ct.
  Scheduler.forward_ct_decode: Scheduler#forward_ct_decode.
  TEST_RETRACT: TEST_RETRACT.
  Scheduler.pub_sub_addr: Scheduler#pub_sub_addr.
  Scheduler.pub_sub_sync_addr: Scheduler#pub_sub_sync_addr.
  Scheduler.tp_size: Scheduler#tp_size.
  Scheduler.dp_schedule_policy: Scheduler#dp_schedule_policy.
  Scheduler.lora_paths: Scheduler#lora_paths.
  Scheduler._pd_pending_bootstrap: Scheduler#_pd_pending_bootstrap.
  Scheduler.max_running_requests: Scheduler#max_running_requests.
  Scheduler.accept_token: Scheduler#accept_token.
  Scheduler.spec_num_forward_ct: Scheduler#spec_num_forward_ct.
  Scheduler.draft_token: Scheduler#draft_token.
  Scheduler.schedule_policy: Scheduler#schedule_policy.
  Scheduler.skip_tokenizer_init: Scheduler#skip_tokenizer_init.
  Scheduler.stream_interval: Scheduler#stream_interval.
  Scheduler.max_loras_per_batch: Scheduler#max_loras_per_batch.
  Scheduler.init_tokenizer: Scheduler#init_tokenizer().
  Scheduler.init_memory_pool_and_cache: Scheduler#init_memory_pool_and_cache().
  Scheduler._extract_max_new_tokens: Scheduler#_extract_max_new_tokens().
  Scheduler.disagg_heartbeat: Scheduler#disagg_heartbeat.
  Scheduler.tokenizer: Scheduler#tokenizer.
  Scheduler.max_req_input_len: Scheduler#max_req_input_len.
  RECORD_STEP_TIME: RECORD_STEP_TIME.
  Scheduler.max_seq_len: Scheduler#max_seq_len.
  Scheduler.aborted_reqs: Scheduler#aborted_reqs.
  GenerationBatchResult.spec_relay_buffers: GenerationBatchResult#spec_relay_buffers.
  GenerationBatchResult.prefill_relay_future_indices: GenerationBatchResult#prefill_relay_future_indices.
  Scheduler.disagg_bootstrap_server: Scheduler#disagg_bootstrap_server.
  Scheduler.disagg_bootstrap_key: Scheduler#disagg_bootstrap_key.
  Scheduler.max_req_len: Scheduler#max_req_len.
  Scheduler.full_tokens_per_layer: Scheduler#full_tokens_per_layer.
  Scheduler.last_prefill_tokens: Scheduler#last_prefill_tokens.
  Scheduler.last_decode_stats_tic: Scheduler#last_decode_stats_tic.
  Scheduler.last_prefill_stats_tic: Scheduler#last_prefill_stats_tic.
  Scheduler.dp_round_robin_counter: Scheduler#dp_round_robin_counter.
  Scheduler.watchdog_last_forward_ct: Scheduler#watchdog_last_forward_ct.
  TEST_RETRACT_INTERVAL: TEST_RETRACT_INTERVAL.
  TEST_RETRACT_NO_PREFILL_BS: TEST_RETRACT_NO_PREFILL_BS.
  GRAMMAR_TIMEOUT: GRAMMAR_TIMEOUT.
  SyncError: SyncError#
  SendDataError: SendDataError#
  ReceiveDataError: ReceiveDataError#
  GenerationBatchResult.num_accepted_tokens: GenerationBatchResult#num_accepted_tokens.
  Scheduler.max_prefill_tokens: Scheduler#max_prefill_tokens.
  Scheduler._extract_ignore_eos: Scheduler#_extract_ignore_eos().
  Scheduler.event_loop_overlap: Scheduler#event_loop_overlap().
  Scheduler.watchdog_thread: Scheduler#watchdog_thread().
  _reserve_host_slot_for_pd: _reserve_host_slot_for_pd().
  Scheduler.num_retracted_reqs: Scheduler#num_retracted_reqs.
  Scheduler.num_paused_reqs: Scheduler#num_paused_reqs.
---
# Module: [`python/sgl_jax/srt/managers/scheduler.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py)

## Classes
### `GenerationBatchResult`
- def: [`python/sgl_jax/srt/managers/scheduler.py:127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L127)
- signature: `class GenerationBatchResult:`
- members:
  - `accept_lens` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L140)
  - `bid` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L132)
  - `cache_miss_count` — [`L133`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L133)
  - `extend_input_len_per_req` — [`L130`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L130)
  - `extend_logprob_start_len_per_req` — [`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L131)
  - `logits_output` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L128)
  - `next_draft_input` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L135)
  - `next_token_ids` — [`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L129)
  - `num_accepted_tokens` — [`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L139)
  - `prefill_relay_future_indices` — [`L137`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L137)
  - `spec_relay_buffers` — [`L136`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L136)
- uses (calls/refs, reference-scoped): [`EagleDraftInput`](../speculative/eagle_util.md#EagleDraftInput), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput)
- used by: [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`process_batch_result_prefill`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`prepare_for_extend_after_verify`](../speculative/eagle_util.md#EagleDraftInput.prepare_for_extend_after_verify), [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`run_batch`](scheduler.md#Scheduler.run_batch), [`verify`](../speculative/base_worker.md#BaseSpecWorker.verify), [`spec_prefill`](../speculative/draft_extend_fused.md#spec_prefill), [`_run_speculative_batch`](scheduler.md#Scheduler._run_speculative_batch), [`spec_decode_verify`](../speculative/draft_extend_fused.md#spec_decode_verify), [`draft_extend_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`process_batch_result`](scheduler.md#Scheduler.process_batch_result), [`spec_decode_overlap`](../speculative/draft_extend_fused.md#spec_decode_overlap), [`draft_extend_for_decode`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_decode)

### `ReceiveDataError`  ·  implements/extends Exception
- def: [`python/sgl_jax/srt/managers/scheduler.py:122`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L122)
- signature: `class ReceiveDataError(Exception):`
- used by: [`broadcast_pyobj`](scheduler.md#Scheduler.broadcast_pyobj)

### `Scheduler`  ·  implements/extends SchedulerDisaggregationDecodeMixin, SchedulerDisaggregationPrefillMixin, SchedulerMetricsMixin, SchedulerOutputProcessorMixin, SchedulerProfilerMixin
- def: [`python/sgl_jax/srt/managers/scheduler.py:143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L143) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
- doc: A scheduler that manages a tensor parallel TPU worker, which managaes fixed multi TPU devices.
- signature: `class Scheduler(SchedulerOutputProcessorMixin, SchedulerProfilerMixin, SchedulerMetricsMixin, SchedulerDisaggregationPrefillMixin, SchedulerDisaggregationDecodeMixin):`
- members:
  - `_cached_prefix_len(self, token_ids: list[int], extra_key: str | None, dp_rank: int)` — [`L813`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L813) — Length of the longest cached prefix for ``token_ids`` on ``dp_rank``.
  - `_can_flush_cache(self)` — [`L1401`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1401) — Return whether cache flush can proceed and an optional error message.
  - `_dp_load_and_eligible(self, extra_counts: list[int], extra_token_counts: list[int])` — [`L767`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L767) — Per-DP (running + pending) load and the ranks that can accept a request.
  - `_estimate_req_tokens(self, req: Req | TokenizedGenerateReqInput)` — [`L703`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L703) — Estimate per-request token load as input + expected output.
  - `_extract_dp_output_ids(self, next_token_ids_flat: np.ndarray, model_worker_batch, batch: ScheduleBatch)` — [`L1950`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1950) — Extract output IDs from DP-formatted array and assign to reqs_info.
  - `_extract_max_new_tokens(sampling_params: object)` — [`L666`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L666) — Extract max_new_tokens from sampling params with a conservative fallback.
  - `_get_dp_load_snapshot(self)` — [`L732`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L732) — Return per-DP (request_count, token_count) for in-flight scheduled work.
  - `_select_cache_aware_dp(self, req: TokenizedGenerateReqInput, extra_counts: list[int], extra_token_counts: list[int])` — [`L827`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L827) — Route ``req`` by cache affinity with soft load balancing.
  - `_select_min_running_dp(self, extra_counts: list[int] | None = None, extra_token_counts: list[int] | None = None)` — [`L788`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L788) — Select a DP rank with the minimum (running requests, scheduled tokens) load.
  - `abort_request(self, recv_req: AbortReq)` — [`L2261`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2261) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `broadcast_pyobj(self, recv_reqs)` — [`L1062`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1062)
  - `check_memory(self)` — [`L1503`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1503)
  - `check_tree_cache(self)` — [`L1557`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1557)
  - `continue_generation(self, recv_req: ContinueGenerationReqInput)` — [`L2383`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2383)
  - `event_loop_normal(self)` — [`L928`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L928) — A normal scheduler loop.
  - `event_loop_overlap(self)` — [`L962`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L962) — A scheduler loop that overlaps the CPU processing and Accelerator computation.
  - `flush_cache(self)` — [`L1444`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1444)
  - `flush_cache_wrapped(self, recv_req: FlushCacheReqInput)` — [`L1392`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1392)
  - `get_internal_state(self, recv_req: GetInternalStateReq)` — [`L1274`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1274) — documented in [root](../../../../../concepts/root.md)
  - `get_new_batch_prefill(self)` — [`L1681`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1681) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `get_next_batch_to_run(self)` — [`L1601`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1601)
  - `handle_generate_request(self, recv_req: TokenizedGenerateReqInput)` — [`L1106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1106) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `init_memory_pool_and_cache(self)` — [`L623`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L623)
  - `init_tokenizer(self)` — [`L602`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L602)
  - `move_ready_grammar_requests(self)` — [`L1227`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1227) — Poll grammar futures and move ready requests to waiting queue.
  - `pause_generation(self, recv_req: PauseGenerationReqInput)` — [`L2357`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2357)
  - `process_batch_result(self, batch: ScheduleBatch, result: GenerationBatchResult, launch_done: threading.Event | None = None)` — [`L2100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2100)
  - `process_input_requests(self, recv_reqs: list)` — [`L1097`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1097)
  - `recv_requests(self)` — [`L1072`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1072) — Receive results at node_rank = 0 and broadcast it to all other Node ranks.
  - `run_batch(self, batch: ScheduleBatch)` — [`L1977`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1977) — Run a batch. — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `run_publisher(self, recv_reqs)` — [`L1028`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1028)
  - `run_subscriber(self)` — [`L1043`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1043)
  - `select_dp_for_request(self, recv_reqs: list[Req])` — [`L858`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L858) — Assign dp_rank to incoming requests using the configured DP policy.
  - `set_internal_state(self, recv_req: SetInternalStateReq)` — [`L1332`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1332) — Handle internal state updates, including precision tracer configuration
  - `set_next_batch_sampling_info_done(self, batch: ScheduleBatch)` — [`L2117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2117)
  - `sync_pub(self)` — [`L549`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L549)
  - `sync_pub_sub(self)` — [`L597`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L597)
  - `sync_sub(self)` — [`L578`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L578)
  - `update_running_batch(self, batch: ScheduleBatch)` — [`L1882`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1882) — Update the current running decoding batch.
  - `watchdog_thread(self)` — [`L2236`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2236) — A watch dog thread that will try to kill the server itself if one forward batch takes too long.
  - `aborted_reqs` — [`L398`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L398)
  - `accept_token` — [`L423`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L423)
  - `chunked_prefill_size` — [`L427`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L427)
  - `chunked_reqs` — [`L430`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L430)
  - `cur_batch` — [`L412`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L412)
  - `disagg_bootstrap_key` — [`L200`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L200)
  - `disagg_bootstrap_server` — [`L198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L198)
  - `disagg_decode_watchdog` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L206)
  - `disagg_heartbeat` — [`L199`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L199)
  - `disagg_shutdown` — [`L201`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L201)
  - `dp_round_robin_counter` — [`L468`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L468)
  - `dp_schedule_policy` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L180)
  - `dp_size` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L177) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `draft_token` — [`L425`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L425)
  - `draft_worker` — [`L352`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L352)
  - `enable_overlap` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L185) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `forward_ct` — [`L415`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L415)
  - `forward_ct_decode` — [`L416`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L416)
  - `full_tokens_per_layer` — [`L386`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L386)
  - `grammar_backend` — [`L272`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L272)
  - `grammar_queue` — [`L273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L273)
  - `init_new_token_ratio` — [`L444`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L444)
  - `is_generation` — [`L605`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L605)
  - `is_hybrid` — [`L382`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L382)
  - `is_mixed_chunk` — [`L431`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L431)
  - `last_batch` — [`L414`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L414)
  - `last_decode_stats_tic` — [`L419`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L419)
  - `last_prefill_stats_tic` — [`L420`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L420)
  - `last_prefill_tokens` — [`L418`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L418)
  - `lora_paths` — [`L213`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L213)
  - `max_loras_per_batch` — [`L214`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L214)
  - `max_prefill_tokens` — [`L362`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L362)
  - `max_req_input_len` — [`L365`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L365)
  - `max_req_len` — [`L364`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L364)
  - `max_running_requests` — [`L363`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L363)
  - `max_seq_len` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L183)
  - `max_total_num_tokens` — [`L361`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L361)
  - `mesh` — [`L300`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L300) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `min_new_token_ratio` — [`L448`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L448)
  - `model_config` — [`L604`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L604)
  - `new_token_ratio` — [`L455`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L455)
  - `new_token_ratio_decay` — [`L452`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L452)
  - `nnodes` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L172)
  - `node_rank` — [`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L171)
  - `num_generated_tokens` — [`L417`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L417)
  - `num_paused_reqs` — [`L422`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L422)
  - `num_retracted_reqs` — [`L421`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L421)
  - `num_subscribers` — [`L251`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L251)
  - `page_size` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L184)
  - `parent_process` — [`L461`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L461)
  - `pending_dp_reqs` — [`L396`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L396)
  - `per_dp_max_running_requests` — [`L380`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L380)
  - `policy` — [`L439`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L439)
  - `processor` — [`L607`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L607)
  - `pub_sub_addr` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L174)
  - `pub_sub_sync_addr` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L175)
  - `publisher` — [`L247`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L247)
  - `publisher_sync` — [`L248`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L248)
  - `random_seed` — [`L366`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L366)
  - `recv_from_rpc` — [`L243`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L243)
  - `recv_from_tokenizer` — [`L225`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L225)
  - `req_to_token_pool` — [`L626`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L626)
  - `result_queue` — [`L964`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L964)
  - `running_batch` — [`L400`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L400) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `schedule_policy` — [`L179`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L179)
  - `send_to_detokenizer` — [`L234`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L234)
  - `send_to_tokenizer` — [`L228`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L228)
  - `server_args` — [`L170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L170)
  - `skip_tokenizer_init` — [`L181`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L181)
  - `sliding_window_size` — [`L383`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L383)
  - `spec_algorithm` — [`L192`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L192) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `spec_num_forward_ct` — [`L424`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L424)
  - `stream_interval` — [`L182`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L182)
  - `subscriber` — [`L258`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L258)
  - `subscriber_sync` — [`L261`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L261)
  - `swa_tokens_per_layer` — [`L386`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L386)
  - `token_to_kv_pool_allocator` — [`L626`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L626)
  - `tokenizer` — [`L607`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L607)
  - `tp_size` — [`L178`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L178)
  - `tp_worker` — [`L323`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L323)
  - `tree_cache` — [`L627`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L627) — documented in [python-sgl_jax-srt-managers-scheduler](../../../../../concepts/python-sgl_jax-srt-managers-scheduler.md)
  - `waiting_queue` — [`L394`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L394)
  - `watchdog_last_forward_ct` — [`L2238`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2238)
  - `watchdog_last_time` — [`L2239`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2239)
  - `watchdog_timeout` — [`L458`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L458)
- protocol/private: `__init__`[`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L154), `_add_request_to_queue`[`L1492`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1492), `_batch_size`[`L1404`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1404), `_comm_backend`[`L218`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L218), `_current_sampling_info_owner`[`L2124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2124), `_engine_paused`[`L436`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L436), `_extend_requests_to_queue`[`L1500`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1500), `_extract_ignore_eos`[`L696`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L696), `_get_input_token_len`[`L646`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L646), `_get_swa_token_info`[`L1573`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1573), `_get_token_info`[`L1561`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L1561), `_is_spec_decode_enabled`[`L546`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L546), `_pd_pending_bootstrap`[`L210`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L210), `_request_dispatcher`[`L471`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L471), `_run_speculative_batch`[`L2129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2129), `_select_round_robin_dp`[`L640`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L640), `_setup_jit_cache`[`L494`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L494), `_spec_multi_layer`[`L331`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L331)
- uses (calls/refs, reference-scoped): `reqs`, [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), `get_model_worker_batch`, [`process_batch_result_prefill`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), `reqs_info`, `prepare_for_extend`, [`ServerArgs`](../server_args.md#ServerArgs), [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), `seq_lens`, `Req`, [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), `prepare_for_decode`, [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), `dp_size`, [`add_one_req`](schedule_policy.md#PrefillAdder.add_one_req), `filter_batch`, `mix_with_running`, `forward_mode`, [`build_kv_cache`](../mem_cache/kv_cache_builder.md#build_kv_cache), `sampling_params`, [`RadixKey`](../mem_cache/radix_cache.md#RadixKey), `spec_info`, [`ModelConfig`](../configs/model_config.md#ModelConfig), `merge_batch`, `init_new`, [`precision_tracer`](../precision_tracer.md#precision_tracer), [`logger`](scheduler.md#logger), `origin_input_ids`, `rid`, [`is_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_extend), `ScheduleBatch`, [`calc_priority`](schedule_policy.md#SchedulePolicy.calc_priority), `copy`, [`log_prefill_stats`](scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_prefill_stats), `retract_decode`, [`from_server_args`](../configs/model_config.md#ModelConfig.from_server_args), `return_logprob`, [`EagleDraftInput`](../speculative/eagle_util.md#EagleDraftInput), `dp_rank`, [`add_chunked_req`](schedule_policy.md#PrefillAdder.add_chunked_req)  (+285 more)
- used by: [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`install_disaggregation_wiring`](../disaggregation/runtime.md#install_disaggregation_wiring), [`process_batch_result_prefill`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`stream_output_generation`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.stream_output_generation), [`_admit_decode_prealloc`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`add_input_logprob_return_values`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.add_input_logprob_return_values), [`process_decode_queue`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin.process_decode_queue), [`process_input_requests_disagg_decode`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`log_decode_stats`](scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_decode_stats), [`process_prefill_chunk`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin.process_prefill_chunk), [`_write_kv_to_pool`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._write_kv_to_pool), [`add_logprob_return_values`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.add_logprob_return_values), [`log_prefill_stats`](scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_prefill_stats), [`_extract_req_kv`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._extract_req_kv), [`event_loop_normal_disagg_decode`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode), [`event_loop_normal_disagg_prefill`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill), [`_on_prefill_transfer_terminal`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._on_prefill_transfer_terminal), [`_drain_transfer_queue_synced`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced), [`_abort_decode_request`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._abort_decode_request), [`run_scheduler_process`](scheduler.md#run_scheduler_process), [`maybe_collect_routed_experts`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.maybe_collect_routed_experts), [`_build_kv_spec_for_req`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._build_kv_spec_for_req), [`_finish_prefill_only_failure`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._finish_prefill_only_failure), [`_shutdown`](../disaggregation/runtime.md#_make_disagg_shutdown._shutdown), [`_pd_mark_time`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._pd_mark_time), [`_abort_prefill_req`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._abort_prefill_req), [`run_scheduler_loop_thread_after_create`](scheduler.md#run_scheduler_loop_thread_after_create), [`dispatch_scheduler_event_loop`](scheduler.md#dispatch_scheduler_event_loop), [`_decode_backlog_snapshot`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._decode_backlog_snapshot), [`_finish_prefill_only_success`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._finish_prefill_only_success), [`_release_decode_req_resources`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._release_decode_req_resources), [`send_kv_chunk`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin.send_kv_chunk), [`_stream_prefill_req`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._stream_prefill_req), [`_pd_mark_time`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._pd_mark_time), [`_release_decode_kv_indices`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._release_decode_kv_indices), [`_extract_pd_reqs_from_waiting_queue`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._extract_pd_reqs_from_waiting_queue), [`_release_prefill_req_resources`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._release_prefill_req_resources), [`stream_output`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.stream_output), [`_release_prefill_kv_pool`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._release_prefill_kv_pool), [`_enqueue_for_decode`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._enqueue_for_decode)  (+10 more)

### `SendDataError`  ·  implements/extends Exception
- def: [`python/sgl_jax/srt/managers/scheduler.py:118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L118)
- signature: `class SendDataError(Exception):`
- used by: [`broadcast_pyobj`](scheduler.md#Scheduler.broadcast_pyobj)

### `SyncError`  ·  implements/extends Exception
- def: [`python/sgl_jax/srt/managers/scheduler.py:114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L114)
- signature: `class SyncError(Exception):`
- used by: [`sync_pub_sub`](scheduler.md#Scheduler.sync_pub_sub)

## Functions
- `_reserve_host_slot_for_pd(host_pool, use_d2h_staging, req)` — [`L2388`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2388) — D1 admission. Returns (admit_ok, reserved_buffer_id).
- `dispatch_scheduler_event_loop(scheduler: Scheduler, server_args: ServerArgs)` — [`L2409`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2409) — Choose and run the appropriate scheduler event loop.
- `run_scheduler_loop_thread_after_create(server_args: ServerArgs, port_args: PortArgs)` — [`L2463`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2463)
- `run_scheduler_process(server_args: ServerArgs, port_args: PortArgs, dp_rank: int | None, pipe_writer)` — [`L2423`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2423)
- `scheduler_loop_after_create(server_args, scheduler)` — [`L2490`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L2490)

## Module values
- `GRAMMAR_TIMEOUT` — [`L111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L111)
- `RECORD_STEP_TIME` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L110)
- `TEST_RETRACT` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L107)
- `TEST_RETRACT_INTERVAL` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L108)
- `TEST_RETRACT_NO_PREFILL_BS` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L109)
- `logger` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler.py#L104)

