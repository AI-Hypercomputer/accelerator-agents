---
title: 'Module: python/sgl_jax/srt/server_args.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/server_args.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.server_args`/
symbols:
  ServerArgs.add_cli_args: ServerArgs#add_cli_args().
  ServerArgs: ServerArgs#
  ServerArgs.__post_init__: ServerArgs#__post_init__().
  ServerArgs.check_lora_server_args: ServerArgs#check_lora_server_args().
  ServerArgs.check_server_args: ServerArgs#check_server_args().
  ServerArgs.check_dynamic_lora_args: ServerArgs#check_dynamic_lora_args().
  PortArgs.init_new: PortArgs#init_new().
  ServerArgs.lora_paths: ServerArgs#lora_paths.
  ServerArgs.from_cli: ServerArgs#from_cli().
  ServerArgs.get_hf_config: ServerArgs#get_hf_config().
  ServerArgs.page_size: ServerArgs#page_size.
  ServerArgs.max_recurrent_state_size: ServerArgs#max_recurrent_state_size.
  ServerArgs.model_path: ServerArgs#model_path.
  ServerArgs.max_running_requests: ServerArgs#max_running_requests.
  ServerArgs.chunked_prefill_size: ServerArgs#chunked_prefill_size.
  ServerArgs.device: ServerArgs#device.
  ServerArgs.dp_size: ServerArgs#dp_size.
  ServerArgs.speculative_algorithm: ServerArgs#speculative_algorithm.
  ServerArgs.tp_size: ServerArgs#tp_size.
  ServerArgs.random_seed: ServerArgs#random_seed.
  ServerArgs.enable_lora: ServerArgs#enable_lora.
  ServerArgs.from_cli_args: ServerArgs#from_cli_args().
  ServerArgs.check_static_lora_args: ServerArgs#check_static_lora_args().
  ServerArgs.dtype_config: ServerArgs#dtype_config.
  ServerArgs.lora_target_modules: ServerArgs#lora_target_modules.
  ServerArgs.enable_static_lora: ServerArgs#enable_static_lora.
  ServerArgs.disaggregation_mode: ServerArgs#disaggregation_mode.
  ServerArgs.url: ServerArgs#url().
  logger: logger.
  ServerArgs.ep_size: ServerArgs#ep_size.
  ServerArgs.multimodal: ServerArgs#multimodal.
  ServerArgs.load_format: ServerArgs#load_format.
  ServerArgs.device_indexes: ServerArgs#device_indexes.
  ServerArgs.nnodes: ServerArgs#nnodes.
  ServerArgs.attention_backend: ServerArgs#attention_backend.
  ServerArgs.max_loras_per_batch: ServerArgs#max_loras_per_batch.
  ServerArgs.disaggregation_enable_d2h: ServerArgs#disaggregation_enable_d2h.
  PortArgs: PortArgs#
  ServerArgs.tokenizer_path: ServerArgs#tokenizer_path.
  ServerArgs.mem_fraction_static: ServerArgs#mem_fraction_static.
  ServerArgs.ep_num_redundant_experts: ServerArgs#ep_num_redundant_experts.
  ServerArgs.max_seq_len: ServerArgs#max_seq_len.
  ServerArgs.disaggregation_bootstrap_url: ServerArgs#disaggregation_bootstrap_url.
  ServerArgs.disaggregation_host_ip: ServerArgs#disaggregation_host_ip.
  ServerArgs.skip_tokenizer_init: ServerArgs#skip_tokenizer_init.
  ServerArgs.revision: ServerArgs#revision.
  ServerArgs.host: ServerArgs#host.
  ServerArgs.kv_cache_dtype: ServerArgs#kv_cache_dtype.
  ServerArgs.node_rank: ServerArgs#node_rank.
  ServerArgs.disable_radix_cache: ServerArgs#disable_radix_cache.
  ServerArgs.moe_backend: ServerArgs#moe_backend.
  ServerArgs.grammar_backend: ServerArgs#grammar_backend.
  ServerArgs.speculative_num_steps: ServerArgs#speculative_num_steps.
  ServerArgs.expert_balance_output_file: ServerArgs#expert_balance_output_file.
  ServerArgs.disaggregation_shared_secret: ServerArgs#disaggregation_shared_secret.
  _validate_disaggregation_host_ip: _validate_disaggregation_host_ip().
  ServerArgs.trust_remote_code: ServerArgs#trust_remote_code.
  ServerArgs.port: ServerArgs#port.
  ServerArgs.quantization: ServerArgs#quantization.
  ServerArgs.max_prefill_tokens: ServerArgs#max_prefill_tokens.
  ServerArgs.schedule_conservativeness: ServerArgs#schedule_conservativeness.
  ServerArgs.ep_dispatch_algorithm: ServerArgs#ep_dispatch_algorithm.
  ServerArgs.decode_log_interval: ServerArgs#decode_log_interval.
  ServerArgs.served_model_name: ServerArgs#served_model_name.
  ServerArgs.json_model_override_args: ServerArgs#json_model_override_args.
  ServerArgs.enable_unified_radix_tree: ServerArgs#enable_unified_radix_tree.
  ServerArgs.speculative_eagle_topk: ServerArgs#speculative_eagle_topk.
  ServerArgs.speculative_num_draft_tokens: ServerArgs#speculative_num_draft_tokens.
  ServerArgs.enable_expert_balance_debug: ServerArgs#enable_expert_balance_debug.
  ServerArgs.expert_balance_segment_counter: ServerArgs#expert_balance_segment_counter.
  ServerArgs.disaggregation_bootstrap_port: ServerArgs#disaggregation_bootstrap_port.
  ServerArgs.disaggregation_d2h_pool_size: ServerArgs#disaggregation_d2h_pool_size.
  ServerArgs.disaggregation_channel_number: ServerArgs#disaggregation_channel_number.
  PortArgs.tokenizer_ipc_name: PortArgs#tokenizer_ipc_name.
  ServerArgs.tokenizer_mode: ServerArgs#tokenizer_mode.
  ServerArgs.tokenizer_backend: ServerArgs#tokenizer_backend.
  ServerArgs.context_length: ServerArgs#context_length.
  ServerArgs.model_impl: ServerArgs#model_impl.
  ServerArgs.model_layer_nums: ServerArgs#model_layer_nums.
  ServerArgs.skip_server_warmup: ServerArgs#skip_server_warmup.
  ServerArgs.dtype: ServerArgs#dtype.
  ServerArgs.quantization_config_path: ServerArgs#quantization_config_path.
  ServerArgs.max_total_tokens: ServerArgs#max_total_tokens.
  ServerArgs.schedule_policy: ServerArgs#schedule_policy.
  ServerArgs.swa_full_tokens_ratio: ServerArgs#swa_full_tokens_ratio.
  ServerArgs.recurrent_state_memory_ratio: ServerArgs#recurrent_state_memory_ratio.
  ServerArgs.enable_sequence_parallel: ServerArgs#enable_sequence_parallel.
  ServerArgs.stream_interval: ServerArgs#stream_interval.
  ServerArgs.constrained_json_whitespace_pattern: ServerArgs#constrained_json_whitespace_pattern.
  ServerArgs.watchdog_timeout: ServerArgs#watchdog_timeout.
  ServerArgs.download_dir: ServerArgs#download_dir.
  ServerArgs.dp_schedule_policy: ServerArgs#dp_schedule_policy.
  ServerArgs.disable_overlap_schedule: ServerArgs#disable_overlap_schedule.
  ServerArgs.enable_precision_tracer: ServerArgs#enable_precision_tracer.
  ServerArgs.disable_jax_allreduce_metadata: ServerArgs#disable_jax_allreduce_metadata.
  ServerArgs.speculative_draft_model_path: ServerArgs#speculative_draft_model_path.
  ServerArgs.max_lora_rank: ServerArgs#max_lora_rank.
  ServerArgs.max_loaded_loras: ServerArgs#max_loaded_loras.
  ServerArgs.lora_scaling: ServerArgs#lora_scaling.
  ServerArgs.enable_return_routed_experts: ServerArgs#enable_return_routed_experts.
  ServerArgs.init_expert_location: ServerArgs#init_expert_location.
  ServerArgs.expert_distribution_recorder_buffer_size: ServerArgs#expert_distribution_recorder_buffer_size.
  ServerArgs.disaggregation_transfer_port: ServerArgs#disaggregation_transfer_port.
  ServerArgs.disaggregation_side_channel_port: ServerArgs#disaggregation_side_channel_port.
  ServerArgs.disaggregation_d2h_max_tokens: ServerArgs#disaggregation_d2h_max_tokens.
  ServerArgs.disaggregation_bootstrap_timeout_seconds: ServerArgs#disaggregation_bootstrap_timeout_seconds.
  ServerArgs.disaggregation_pull_timeout_seconds: ServerArgs#disaggregation_pull_timeout_seconds.
  ServerArgs.disaggregation_ack_timeout_seconds: ServerArgs#disaggregation_ack_timeout_seconds.
  ServerArgs.disaggregation_orphan_reaper_interval_seconds: ServerArgs#disaggregation_orphan_reaper_interval_seconds.
  ServerArgs.disaggregation_decode_watchdog_seconds: ServerArgs#disaggregation_decode_watchdog_seconds.
  ServerArgs.disaggregation_num_reserved_decode_tokens: ServerArgs#disaggregation_num_reserved_decode_tokens.
  ServerArgs.disaggregation_max_inflight_transfers: ServerArgs#disaggregation_max_inflight_transfers.
  PortArgs.scheduler_input_ipc_name: PortArgs#scheduler_input_ipc_name.
  PortArgs.detokenizer_ipc_name: PortArgs#detokenizer_ipc_name.
  PortArgs.pub_sub_addr: PortArgs#pub_sub_addr.
  PortArgs.pub_sub_sync_addr: PortArgs#pub_sub_sync_addr.
  PortArgs.rpc_ipc_name: PortArgs#rpc_ipc_name.
  GRAMMAR_BACKEND_CHOICES: GRAMMAR_BACKEND_CHOICES.
  _REJECTED_PD_HOST_ALIASES: _REJECTED_PD_HOST_ALIASES.
  ServerArgs.model_loader_extra_config: ServerArgs#model_loader_extra_config.
  ServerArgs.is_embedding: ServerArgs#is_embedding.
  ServerArgs.enable_mixed_chunk: ServerArgs#enable_mixed_chunk.
  ServerArgs.disable_hybrid_swa_memory: ServerArgs#disable_hybrid_swa_memory.
  ServerArgs.constrained_json_disable_any_whitespace: ServerArgs#constrained_json_disable_any_whitespace.
  ServerArgs.dist_timeout: ServerArgs#dist_timeout.
  ServerArgs.log_level: ServerArgs#log_level.
  ServerArgs.log_level_http: ServerArgs#log_level_http.
  ServerArgs.crash_dump_folder: ServerArgs#crash_dump_folder.
  ServerArgs.bucket_time_to_first_token: ServerArgs#bucket_time_to_first_token.
  ServerArgs.bucket_inter_token_latency: ServerArgs#bucket_inter_token_latency.
  ServerArgs.bucket_e2e_request_latency: ServerArgs#bucket_e2e_request_latency.
  ServerArgs.enable_request_time_stats_logging: ServerArgs#enable_request_time_stats_logging.
  ServerArgs.api_key: ServerArgs#api_key.
  ServerArgs.file_storage_path: ServerArgs#file_storage_path.
  ServerArgs.reasoning_parser: ServerArgs#reasoning_parser.
  ServerArgs.tool_call_parser: ServerArgs#tool_call_parser.
  ServerArgs.dist_init_addr: ServerArgs#dist_init_addr.
  ServerArgs.allow_auto_truncate: ServerArgs#allow_auto_truncate.
  ServerArgs.precompile_token_paddings: ServerArgs#precompile_token_paddings.
  ServerArgs.precompile_bs_paddings: ServerArgs#precompile_bs_paddings.
  ServerArgs.disable_precompile: ServerArgs#disable_precompile.
  ServerArgs.speculative_draft_model_revision: ServerArgs#speculative_draft_model_revision.
  ServerArgs.speculative_accept_threshold_single: ServerArgs#speculative_accept_threshold_single.
  ServerArgs.speculative_accept_threshold_acc: ServerArgs#speculative_accept_threshold_acc.
  ServerArgs.enable_deterministic_sampling: ServerArgs#enable_deterministic_sampling.
  ServerArgs.enable_single_process: ServerArgs#enable_single_process.
  ServerArgs.use_sort_for_toppk_minp: ServerArgs#use_sort_for_toppk_minp.
  ServerArgs.enable_expert_distribution_recorder: ServerArgs#enable_expert_distribution_recorder.
  ServerArgs.expert_distribution_recorder_output_file: ServerArgs#expert_distribution_recorder_output_file.
  PortArgs.metrics_ipc_name: PortArgs#metrics_ipc_name.
  ServerArgs.warmups: ServerArgs#warmups.
  ServerArgs.quantization_param_path: ServerArgs#quantization_param_path.
  ServerArgs.stream_output: ServerArgs#stream_output.
  ServerArgs.sleep_on_idle: ServerArgs#sleep_on_idle.
  ServerArgs.log_requests: ServerArgs#log_requests.
  ServerArgs.log_requests_level: ServerArgs#log_requests_level.
  ServerArgs.show_time_cost: ServerArgs#show_time_cost.
  ServerArgs.enable_metrics: ServerArgs#enable_metrics.
  ServerArgs.kv_events_config: ServerArgs#kv_events_config.
  ServerArgs.enable_cache_report: ServerArgs#enable_cache_report.
  ServerArgs.preferred_sampling_params: ServerArgs#preferred_sampling_params.
  ServerArgs.enable_tokenizer_batch_encode: ServerArgs#enable_tokenizer_batch_encode.
  ServerArgs.enable_nan_detection: ServerArgs#enable_nan_detection.
  ServerArgs.lora_eviction_policy: ServerArgs#lora_eviction_policy.
  ServerArgs.enable_engine_loop_run_forever_daemon: ServerArgs#enable_engine_loop_run_forever_daemon.
  ZMQ_TCP_PORT_DELTA: ZMQ_TCP_PORT_DELTA.
---
# Module: [`python/sgl_jax/srt/server_args.py`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py)

## Classes
### `PortArgs`
- def: [`python/sgl_jax/srt/server_args.py:1674`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1674)
- signature: `class PortArgs:`
- members:
  - `init_new(server_args, dp_rank: int | None = None)` — [`L1694`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1694)
  - `detokenizer_ipc_name` — [`L1680`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1680)
  - `metrics_ipc_name` — [`L1691`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1691)
  - `pub_sub_addr` — [`L1683`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1683)
  - `pub_sub_sync_addr` — [`L1685`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1685)
  - `rpc_ipc_name` — [`L1688`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1688)
  - `scheduler_input_ipc_name` — [`L1678`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1678)
  - `tokenizer_ipc_name` — [`L1676`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1676)
- used by: [`run_scheduler_process`](managers/scheduler.md#run_scheduler_process), [`run_scheduler_loop_thread_after_create`](managers/scheduler.md#run_scheduler_loop_thread_after_create), [`send_to_detokenizer`](managers/scheduler.md#Scheduler.send_to_detokenizer), [`send_to_tokenizer`](managers/scheduler.md#Scheduler.send_to_tokenizer), [`__init__`](managers/scheduler.md#Scheduler.__init__), [`recv_from_rpc`](managers/scheduler.md#Scheduler.recv_from_rpc), [`recv_from_tokenizer`](managers/scheduler.md#Scheduler.recv_from_tokenizer), [`pub_sub_addr`](managers/scheduler.md#Scheduler.pub_sub_addr), [`pub_sub_sync_addr`](managers/scheduler.md#Scheduler.pub_sub_sync_addr)

### `ServerArgs`
- def: [`python/sgl_jax/srt/server_args.py:54`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L54) — documented in [python-sgl_jax-srt-configs-model_config](../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
- signature: `class ServerArgs:`
- members:
  - `add_cli_args(parser: argparse.ArgumentParser)` — [`L444`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L444) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `check_dynamic_lora_args()` — [`L1596`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1596) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `check_lora_server_args(self)` — [`L1549`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1549) — Validate and normalize LoRA-related server arguments. — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `check_server_args(self)` — [`L1520`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1520) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `check_static_lora_args()` — [`L1584`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1584) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `from_cli(cls, argv: list[str] | None = None)` — [`L1484`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1484) — Create ServerArgs from command line arguments.
  - `from_cli_args(cls, args: argparse.Namespace)` — [`L1472`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1472)
  - `get_hf_config(self)` — [`L1509`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1509)
  - `url(self)` — [`L1503`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1503)
  - `allow_auto_truncate` — [`L156`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L156)
  - `api_key` — [`L137`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L137)
  - `attention_backend` — [`L162`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L162)
  - `bucket_e2e_request_latency` — [`L131`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L131)
  - `bucket_inter_token_latency` — [`L130`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L130)
  - `bucket_time_to_first_token` — [`L129`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L129)
  - `chunked_prefill_size` — [`L89`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L89) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `constrained_json_disable_any_whitespace` — [`L111`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L111)
  - `constrained_json_whitespace_pattern` — [`L110`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L110)
  - `context_length` — [`L64`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L64)
  - `crash_dump_folder` — [`L126`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L126)
  - `decode_log_interval` — [`L132`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L132)
  - `device` — [`L100`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L100)
  - `device_indexes` — [`L101`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L101)
  - `disable_hybrid_swa_memory` — [`L97`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L97)
  - `disable_jax_allreduce_metadata` — [`L164`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L164)
  - `disable_overlap_schedule` — [`L158`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L158)
  - `disable_precompile` — [`L173`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L173)
  - `disable_radix_cache` — [`L154`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L154)
  - `disaggregation_ack_timeout_seconds` — [`L245`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L245)
  - `disaggregation_bootstrap_port` — [`L222`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L222)
  - `disaggregation_bootstrap_timeout_seconds` — [`L243`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L243)
  - `disaggregation_bootstrap_url` — [`L221`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L221)
  - `disaggregation_channel_number` — [`L235`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L235)
  - `disaggregation_d2h_max_tokens` — [`L231`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L231)
  - `disaggregation_d2h_pool_size` — [`L230`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L230)
  - `disaggregation_decode_watchdog_seconds` — [`L256`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L256)
  - `disaggregation_enable_d2h` — [`L228`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L228)
  - `disaggregation_host_ip` — [`L239`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L239)
  - `disaggregation_max_inflight_transfers` — [`L272`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L272)
  - `disaggregation_mode` — [`L220`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L220) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `disaggregation_num_reserved_decode_tokens` — [`L263`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L263)
  - `disaggregation_orphan_reaper_interval_seconds` — [`L246`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L246)
  - `disaggregation_pull_timeout_seconds` — [`L244`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L244)
  - `disaggregation_shared_secret` — [`L251`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L251)
  - `disaggregation_side_channel_port` — [`L229`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L229)
  - `disaggregation_transfer_port` — [`L223`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L223)
  - `dist_init_addr` — [`L145`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L145)
  - `dist_timeout` — [`L113`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L113)
  - `download_dir` — [`L114`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L114)
  - `dp_schedule_policy` — [`L119`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L119)
  - `dp_size` — [`L118`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L118) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `dtype` — [`L77`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L77)
  - `dtype_config` — [`L82`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L82)
  - `enable_cache_report` — [`L140`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L140)
  - `enable_deterministic_sampling` — [`L186`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L186)
  - `enable_engine_loop_run_forever_daemon` — [`L205`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L205)
  - `enable_expert_balance_debug` — [`L211`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L211)
  - `enable_expert_distribution_recorder` — [`L215`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L215)
  - `enable_lora` — [`L194`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L194)
  - `enable_metrics` — [`L128`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L128)
  - `enable_mixed_chunk` — [`L90`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L90)
  - `enable_nan_detection` — [`L188`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L188)
  - `enable_precision_tracer` — [`L159`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L159)
  - `enable_request_time_stats_logging` — [`L133`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L133)
  - `enable_return_routed_experts` — [`L210`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L210)
  - `enable_sequence_parallel` — [`L106`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L106)
  - `enable_single_process` — [`L187`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L187)
  - `enable_static_lora` — [`L201`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L201)
  - `enable_tokenizer_batch_encode` — [`L157`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L157)
  - `enable_unified_radix_tree` — [`L155`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L155)
  - `ep_dispatch_algorithm` — [`L105`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L105)
  - `ep_num_redundant_experts` — [`L104`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L104)
  - `ep_size` — [`L103`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L103) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `expert_balance_output_file` — [`L213`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L213)
  - `expert_balance_segment_counter` — [`L212`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L212)
  - `expert_distribution_recorder_buffer_size` — [`L216`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L216)
  - `expert_distribution_recorder_output_file` — [`L217`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L217)
  - `file_storage_path` — [`L139`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L139)
  - `grammar_backend` — [`L166`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L166)
  - `host` — [`L71`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L71)
  - `init_expert_location` — [`L214`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L214)
  - `is_embedding` — [`L65`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L65)
  - `json_model_override_args` — [`L150`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L150)
  - `kv_cache_dtype` — [`L81`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L81)
  - `kv_events_config` — [`L134`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L134)
  - `load_format` — [`L61`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L61)
  - `log_level` — [`L122`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L122)
  - `log_level_http` — [`L123`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L123)
  - `log_requests` — [`L124`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L124)
  - `log_requests_level` — [`L125`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L125)
  - `lora_eviction_policy` — [`L200`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L200)
  - `lora_paths` — [`L197`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L197)
  - `lora_scaling` — [`L202`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L202)
  - `lora_target_modules` — [`L196`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L196)
  - `max_loaded_loras` — [`L198`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L198)
  - `max_lora_rank` — [`L195`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L195)
  - `max_loras_per_batch` — [`L199`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L199)
  - `max_prefill_tokens` — [`L88`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L88)
  - `max_recurrent_state_size` — [`L96`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L96)
  - `max_running_requests` — [`L86`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L86)
  - `max_seq_len` — [`L168`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L168)
  - `max_total_tokens` — [`L87`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L87)
  - `mem_fraction_static` — [`L85`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L85)
  - `model_impl` — [`L67`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L67)
  - `model_layer_nums` — [`L68`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L68)
  - `model_loader_extra_config` — [`L62`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L62)
  - `model_path` — [`L56`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L56)
  - `moe_backend` — [`L163`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L163)
  - `multimodal` — [`L208`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L208)
  - `nnodes` — [`L146`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L146)
  - `node_rank` — [`L147`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L147)
  - `page_size` — [`L93`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L93) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `port` — [`L72`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L72)
  - `precompile_bs_paddings` — [`L171`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L171)
  - `precompile_token_paddings` — [`L170`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L170)
  - `preferred_sampling_params` — [`L151`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L151)
  - `quantization` — [`L78`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L78)
  - `quantization_config_path` — [`L80`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L80)
  - `quantization_param_path` — [`L79`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L79)
  - `random_seed` — [`L109`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L109)
  - `reasoning_parser` — [`L141`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L141)
  - `recurrent_state_memory_ratio` — [`L95`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L95)
  - `revision` — [`L66`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L66)
  - `schedule_conservativeness` — [`L92`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L92)
  - `schedule_policy` — [`L91`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L91)
  - `served_model_name` — [`L138`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L138)
  - `show_time_cost` — [`L127`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L127)
  - `skip_server_warmup` — [`L73`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L73)
  - `skip_tokenizer_init` — [`L60`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L60)
  - `sleep_on_idle` — [`L115`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L115)
  - `speculative_accept_threshold_acc` — [`L183`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L183)
  - `speculative_accept_threshold_single` — [`L182`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L182)
  - `speculative_algorithm` — [`L176`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L176) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `speculative_draft_model_path` — [`L177`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L177)
  - `speculative_draft_model_revision` — [`L178`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L178)
  - `speculative_eagle_topk` — [`L180`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L180)
  - `speculative_num_draft_tokens` — [`L181`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L181)
  - `speculative_num_steps` — [`L179`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L179)
  - `stream_interval` — [`L107`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L107)
  - `stream_output` — [`L108`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L108)
  - `swa_full_tokens_ratio` — [`L94`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L94)
  - `tokenizer_backend` — [`L59`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L59)
  - `tokenizer_mode` — [`L58`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L58)
  - `tokenizer_path` — [`L57`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L57)
  - `tool_call_parser` — [`L142`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L142)
  - `tp_size` — [`L102`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L102) — documented in [python-sgl_jax-srt-server_args](../../../../concepts/python-sgl_jax-srt-server_args.md)
  - `trust_remote_code` — [`L63`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L63)
  - `use_sort_for_toppk_minp` — [`L191`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L191)
  - `warmups` — [`L74`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L74)
  - `watchdog_timeout` — [`L112`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L112)
- protocol/private: `__post_init__`[`L274`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L274)
- uses (calls/refs, reference-scoped): [`add_cli_args`](multimodal/common/ServerArgs.md#MultimodalServerArgs.add_cli_args), [`LoRARef`](lora/lora_registry.md#LoRARef), [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap), [`ToolCallParserEnum`](function_call/function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`get_config`](hf_transformers_utils.md#get_config), [`MultimodalServerArgs`](multimodal/common/ServerArgs.md#MultimodalServerArgs), [`lora_name`](lora/lora_registry.md#LoRARef.lora_name), [`download_from_hf`](hf_transformers_utils.md#download_from_hf), [`logger`](server_args.md#logger), [`resolve_secret`](disaggregation/pd_auth.md#resolve_secret), [`is_remote_url`](utils/common_utils.md#is_remote_url), [`pinned`](lora/lora_registry.md#LoRARef.pinned), [`lora_path`](lora/lora_registry.md#LoRARef.lora_path), [`check_gguf_file`](hf_transformers_utils.md#check_gguf_file), [`_validate_disaggregation_host_ip`](server_args.md#_validate_disaggregation_host_ip), [`ReasoningParser`](reasoning_parser.md#ReasoningParser), [`SUPPORTED_LORA_TARGET_MODULES`](utils/common_utils.md#SUPPORTED_LORA_TARGET_MODULES), [`FunctionCallParser`](function_call/function_call_parser.md#FunctionCallParser), [`LORA_TARGET_ALL_MODULES`](utils/common_utils.md#LORA_TARGET_ALL_MODULES), [`is_valid_ipv6_address`](utils/common_utils.md#is_valid_ipv6_address), [`nullable_str`](utils/common_utils.md#nullable_str), [`GRAMMAR_BACKEND_CHOICES`](server_args.md#GRAMMAR_BACKEND_CHOICES), [`from_cli_args`](multimodal/common/ServerArgs.md#MultimodalServerArgs.from_cli_args)
- used by: [`process_batch_result_decode`](managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`install_disaggregation_wiring`](disaggregation/runtime.md#install_disaggregation_wiring), [`handle_generate_request`](managers/scheduler.md#Scheduler.handle_generate_request), [`run_batch`](managers/scheduler.md#Scheduler.run_batch), [`_admit_decode_prealloc`](disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`forward_batch_generation`](managers/tp_worker.md#ModelWorker.forward_batch_generation), [`_run_speculative_batch`](managers/scheduler.md#Scheduler._run_speculative_batch), [`_init_pools`](model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_pools), [`default_radix_cache_factory`](mem_cache/registry.md#default_radix_cache_factory), [`model_runner`](managers/tp_worker.md#ModelWorker.model_runner), [`build_kv_cache`](mem_cache/kv_cache_builder.md#build_kv_cache), [`_request_dispatcher`](managers/scheduler.md#Scheduler._request_dispatcher), [`process_input_requests_disagg_decode`](disaggregation/decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`log_decode_stats`](managers/scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_decode_stats), [`dp_size`](managers/scheduler.md#Scheduler.dp_size), [`spec_algorithm`](managers/scheduler.md#Scheduler.spec_algorithm), [`init_memory_pool`](model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.init_memory_pool), [`max_running_requests`](managers/tp_worker.md#ModelWorker.max_running_requests), [`cur_sampling_info`](managers/tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`init_routed_experts_capturer`](model_executor/model_runner.md#ModelRunner.init_routed_experts_capturer), [`grammar_queue`](managers/scheduler.md#Scheduler.grammar_queue), [`_get_attention_backend`](model_executor/model_runner.md#ModelRunner._get_attention_backend), [`compilation_manager`](managers/tp_worker.md#ModelWorker.compilation_manager), [`enable_overlap`](managers/scheduler.md#Scheduler.enable_overlap), [`load_model`](model_executor/model_runner.md#ModelRunner.load_model), `retract_decode`, [`from_server_args`](configs/model_config.md#ModelConfig.from_server_args), [`mesh`](managers/scheduler.md#Scheduler.mesh), [`_compute_cell_size`](model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._compute_cell_size), [`set_num_token_hybrid`](model_executor/model_runner.md#ModelRunner.set_num_token_hybrid), [`max_total_num_tokens`](managers/tp_worker.md#ModelWorker.max_total_num_tokens), [`run_scheduler_process`](managers/scheduler.md#run_scheduler_process), [`_resolve_max_num_reqs`](model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._resolve_max_num_reqs), [`handle_recurrent_cache`](model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.handle_recurrent_cache), [`model_loader`](model_executor/model_runner.md#ModelRunner.model_loader), [`sampler`](model_executor/model_runner.md#ModelRunner.sampler), [`lora_manager`](model_executor/model_runner.md#ModelRunner.lora_manager), `release_req`, [`model_config`](managers/tp_worker.md#ModelWorker.model_config), [`_spec_multi_layer`](managers/scheduler.md#Scheduler._spec_multi_layer)  (+79 more)

## Functions
- `_validate_disaggregation_host_ip(host_ip: str)` — [`L34`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L34)

## Module values
- `GRAMMAR_BACKEND_CHOICES` — [`L30`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L30)
- `ZMQ_TCP_PORT_DELTA` — [`L1670`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L1670)
- `_REJECTED_PD_HOST_ALIASES` — [`L31`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L31)
- `logger` — [`L28`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/server_args.py#L28)

