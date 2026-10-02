---
title: 'Module: easydel/infra/elarge/types/engine.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/engine.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.engine`/
symbols:
  BaseCfg: BaseCfg#
  eSurgeCfg: eSurgeCfg#
  BaseCfg.values: BaseCfg#values.
  BaseCfg.operation_configs: BaseCfg#operation_configs.
  eSurgeCfg.sampling_params_callback: eSurgeCfg#sampling_params_callback.
  eSurgeCfg.tool_parser: eSurgeCfg#tool_parser.
  eSurgeCfg.reasoning_parser: eSurgeCfg#reasoning_parser.
  eSurgeCfg.max_model_len: eSurgeCfg#max_model_len.
  eSurgeCfg.min_input_pad: eSurgeCfg#min_input_pad.
  eSurgeCfg.min_token_pad: eSurgeCfg#min_token_pad.
  eSurgeCfg.max_num_seqs: eSurgeCfg#max_num_seqs.
  eSurgeCfg.max_num_seq_buckets: eSurgeCfg#max_num_seq_buckets.
  eSurgeCfg.max_num_batched_tokens: eSurgeCfg#max_num_batched_tokens.
  eSurgeCfg.hbm_utilization: eSurgeCfg#hbm_utilization.
  eSurgeCfg.page_size: eSurgeCfg#page_size.
  eSurgeCfg.use_aot_forward: eSurgeCfg#use_aot_forward.
  eSurgeCfg.bind_graphstate_for_aot: eSurgeCfg#bind_graphstate_for_aot.
  eSurgeCfg.enable_window_aware_runtime_cap: eSurgeCfg#enable_window_aware_runtime_cap.
  eSurgeCfg.enable_prefix_caching: eSurgeCfg#enable_prefix_caching.
  eSurgeCfg.auto_shard_model: eSurgeCfg#auto_shard_model.
  eSurgeCfg.sharding_axis_dims: eSurgeCfg#sharding_axis_dims.
  eSurgeCfg.compile_runner: eSurgeCfg#compile_runner.
  eSurgeCfg.async_scheduling: eSurgeCfg#async_scheduling.
  eSurgeCfg.runner_verbose: eSurgeCfg#runner_verbose.
  eSurgeCfg.verbose: eSurgeCfg#verbose.
  eSurgeCfg.overlap_execution: eSurgeCfg#overlap_execution.
  eSurgeCfg.sampler_metrics: eSurgeCfg#sampler_metrics.
  eSurgeCfg.data_parallelism_axis: eSurgeCfg#data_parallelism_axis.
  eSurgeCfg.esurge_name: eSurgeCfg#esurge_name.
  eSurgeCfg.reserve_tokens: eSurgeCfg#reserve_tokens.
  eSurgeCfg.auto_truncate_prompt: eSurgeCfg#auto_truncate_prompt.
  eSurgeCfg.auto_cap_new_tokens: eSurgeCfg#auto_cap_new_tokens.
  eSurgeCfg.strict_context: eSurgeCfg#strict_context.
  eSurgeCfg.truncate_mode: eSurgeCfg#truncate_mode.
  eSurgeCfg.prefer_preserve_prompt: eSurgeCfg#prefer_preserve_prompt.
  eSurgeCfg.decode_truncated_prompt: eSurgeCfg#decode_truncated_prompt.
  eSurgeCfg.destroy_pages_on_pause: eSurgeCfg#destroy_pages_on_pause.
  eSurgeCfg.detokenizer_max_states: eSurgeCfg#detokenizer_max_states.
  eSurgeCfg.worker_startup_timeout: eSurgeCfg#worker_startup_timeout.
  eSurgeCfg.idle_reset_seconds: eSurgeCfg#idle_reset_seconds.
  eSurgeCfg.idle_reset_min_interval: eSurgeCfg#idle_reset_min_interval.
  eSurgeCfg.tokenizer_endpoint: eSurgeCfg#tokenizer_endpoint.
  eSurgeCfg.detokenizer_endpoint: eSurgeCfg#detokenizer_endpoint.
  eSurgeCfg.long_prefill_token_threshold: eSurgeCfg#long_prefill_token_threshold.
  eSurgeCfg.extra_eos_token_ids: eSurgeCfg#extra_eos_token_ids.
  eSurgeCfg.extra_stops: eSurgeCfg#extra_stops.
  eSurgeCfg.silent_mode: eSurgeCfg#silent_mode.
  eSurgeCfg.ignore_stop_strings_in_reasoning: eSurgeCfg#ignore_stop_strings_in_reasoning.
  eSurgeCfg.distributed_mode: eSurgeCfg#distributed_mode.
  eSurgeCfg.distributed_role: eSurgeCfg#distributed_role.
  eSurgeCfg.distributed_service_name: eSurgeCfg#distributed_service_name.
  eSurgeCfg.distributed_world_size: eSurgeCfg#distributed_world_size.
  eSurgeCfg.distributed_rank: eSurgeCfg#distributed_rank.
  eSurgeCfg.distributed_control_port: eSurgeCfg#distributed_control_port.
  eSurgeCfg.distributed_control_bind_host: eSurgeCfg#distributed_control_bind_host.
  eSurgeCfg.distributed_advertise_addr: eSurgeCfg#distributed_advertise_addr.
  eSurgeCfg.distributed_auth_token: eSurgeCfg#distributed_auth_token.
  eSurgeCfg.distributed_step_timeout_s: eSurgeCfg#distributed_step_timeout_s.
  eSurgeCfg.distributed_connect_timeout_s: eSurgeCfg#distributed_connect_timeout_s.
  eSurgeCfg.distributed_verify_sampling_digest: eSurgeCfg#distributed_verify_sampling_digest.
---
# Module: [`easydel/infra/elarge/types/engine.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py)

## Classes
### `BaseCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/engine.py:33`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L33)
- doc: Container for base model configuration values and ejkernel operation overrides.
- signature: `class BaseCfg(TypedDict, total=False):`
- members:
  - `operation_configs` — [`L46`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L46)
  - `values` — [`L45`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L45)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseConfigDict`](../../base_config.md#EasyDeLBaseConfigDict), [`OperationConfigsDict`](model.md#OperationConfigsDict)
- used by: [`base_config`](root.md#eLMConfig.base_config)

### `eSurgeCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/engine.py:49`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L49)
- doc: Configuration for the eSurge high-throughput inference engine (PagedAttention-based).
- signature: `class eSurgeCfg(TypedDict, total=False):`
- members:
  - `async_scheduling` — [`L170`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L170)
  - `auto_cap_new_tokens` — [`L179`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L179)
  - `auto_shard_model` — [`L167`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L167)
  - `auto_truncate_prompt` — [`L178`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L178)
  - `bind_graphstate_for_aot` — [`L164`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L164)
  - `compile_runner` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L169)
  - `data_parallelism_axis` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L175)
  - `decode_truncated_prompt` — [`L183`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L183)
  - `destroy_pages_on_pause` — [`L184`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L184)
  - `detokenizer_endpoint` — [`L190`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L190)
  - `detokenizer_max_states` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L185)
  - `distributed_advertise_addr` — [`L210`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L210)
  - `distributed_auth_token` — [`L211`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L211)
  - `distributed_connect_timeout_s` — [`L213`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L213)
  - `distributed_control_bind_host` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L209)
  - `distributed_control_port` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L208)
  - `distributed_mode` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L203)
  - `distributed_rank` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L207)
  - `distributed_role` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L204)
  - `distributed_service_name` — [`L205`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L205)
  - `distributed_step_timeout_s` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L212)
  - `distributed_verify_sampling_digest` — [`L214`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L214)
  - `distributed_world_size` — [`L206`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L206)
  - `enable_prefix_caching` — [`L166`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L166)
  - `enable_window_aware_runtime_cap` — [`L165`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L165)
  - `esurge_name` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L176)
  - `extra_eos_token_ids` — [`L195`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L195)
  - `extra_stops` — [`L196`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L196)
  - `hbm_utilization` — [`L161`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L161)
  - `idle_reset_min_interval` — [`L188`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L188)
  - `idle_reset_seconds` — [`L187`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L187)
  - `ignore_stop_strings_in_reasoning` — [`L200`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L200)
  - `long_prefill_token_threshold` — [`L194`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L194)
  - `max_model_len` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L155)
  - `max_num_batched_tokens` — [`L160`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L160)
  - `max_num_seq_buckets` — [`L159`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L159)
  - `max_num_seqs` — [`L158`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L158)
  - `min_input_pad` — [`L156`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L156)
  - `min_token_pad` — [`L157`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L157)
  - `overlap_execution` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L173)
  - `page_size` — [`L162`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L162)
  - `prefer_preserve_prompt` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L182)
  - `reasoning_parser` — [`L199`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L199)
  - `reserve_tokens` — [`L177`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L177)
  - `runner_verbose` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L171)
  - `sampler_metrics` — [`L174`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L174)
  - `sampling_params_callback` — [`L191`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L191)
  - `sharding_axis_dims` — [`L168`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L168)
  - `silent_mode` — [`L197`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L197)
  - `strict_context` — [`L180`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L180)
  - `tokenizer_endpoint` — [`L189`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L189)
  - `tool_parser` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L198)
  - `truncate_mode` — [`L181`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L181)
  - `use_aot_forward` — [`L163`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L163)
  - `verbose` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L172)
  - `worker_startup_timeout` — [`L186`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/engine.py#L186)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../../inference/sampling_params.md#SamplingParams), [`ReasoningParserName`](../../../inference/reasoning/abstract_reasoning.md#ReasoningParserName.ReasoningParserName), [`ToolParserName`](../../../inference/tools/abstract_tool.md#ToolParserName.ToolParserName)
- used by: [`esurge`](root.md#eLMConfig.esurge)

