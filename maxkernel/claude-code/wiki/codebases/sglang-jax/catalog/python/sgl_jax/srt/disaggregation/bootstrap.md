---
title: 'Module: python/sgl_jax/srt/disaggregation/bootstrap.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/bootstrap.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.bootstrap`/
symbols:
  PrefillInfoCache.pick_for_room: PrefillInfoCache#pick_for_room().
  BootstrapServer.start: BootstrapServer#start().
  BootstrapClient.register_prefill: BootstrapClient#register_prefill().
  _Registry.register: _Registry#register().
  BootstrapServer.stop: BootstrapServer#stop().
  PrefillInfoCache._refresh_locked: PrefillInfoCache#_refresh_locked().
  HeartbeatDaemon._loop: HeartbeatDaemon#_loop().
  _Registry._evict_stale_locked: _Registry#_evict_stale_locked().
  _Registry.prefills: _Registry#prefills.
  HeartbeatDaemon.start: HeartbeatDaemon#start().
  BootstrapClient.get_prefill_info: BootstrapClient#get_prefill_info().
  BootstrapClient.list_prefills: BootstrapClient#list_prefills().
  _Registry.heartbeat: _Registry#heartbeat().
  _Registry.unregister: _Registry#unregister().
  _Registry.list_all: _Registry#list_all().
  _Registry.pick_for_room: _Registry#pick_for_room().
  BootstrapClient.heartbeat: BootstrapClient#heartbeat().
  BootstrapClient.unregister_prefill: BootstrapClient#unregister_prefill().
  BootstrapClient._headers: BootstrapClient#_headers().
  BootstrapClient.health: BootstrapClient#health().
  HeartbeatDaemon.stop: HeartbeatDaemon#stop().
  _set_registry_size: _set_registry_size().
  build_app.register_prefill: build_app().register_prefill().
  build_app.heartbeat: build_app().heartbeat().
  build_app.unregister_prefill: build_app().unregister_prefill().
  BootstrapClient._base_url: BootstrapClient#_base_url.
  logger: logger.
  build_app: build_app().
  BootstrapServer._port: BootstrapServer#_port.
  BootstrapServer._server: BootstrapServer#_server.
  BootstrapServer._thread: BootstrapServer#_thread.
  BootstrapServer._wait_until_ready: BootstrapServer#_wait_until_ready().
  BootstrapClient: BootstrapClient#
  BootstrapClient._timeout_s: BootstrapClient#_timeout_s.
  BootstrapClient._client: BootstrapClient#_client.
  PrefillInfoCache._pick_locked: PrefillInfoCache#_pick_locked().
  _Registry.now: _Registry#now().
  PrefillInfo: PrefillInfo#
  _Registry.last_seen: _Registry#last_seen.
  _Registry.lock: _Registry#lock.
  build_app.list_prefills: build_app().list_prefills().
  build_app.get_prefill_info: build_app().get_prefill_info().
  BootstrapServer._started: BootstrapServer#_started.
  PrefillInfoCache._sorted_keys: PrefillInfoCache#_sorted_keys.
  HeartbeatDaemon.__init__: HeartbeatDaemon#__init__().
  HeartbeatDaemon._thread: HeartbeatDaemon#_thread.
  _reject_if_below_protocol_floor: _reject_if_below_protocol_floor().
  resolve_kv_dtype_name: resolve_kv_dtype_name().
  BootstrapServer._host: BootstrapServer#_host.
  BootstrapClient._register_retries: BootstrapClient#_register_retries.
  PrefillInfoCache: PrefillInfoCache#
  HeartbeatDaemon._stop_event: HeartbeatDaemon#_stop_event.
  HeartbeatDaemon._started: HeartbeatDaemon#_started.
  HEARTBEAT_INTERVAL_SECONDS: HEARTBEAT_INTERVAL_SECONDS.
  _Registry.ttl_seconds: _Registry#ttl_seconds.
  _Registry: _Registry#
  PrefillInfoCache._last_refresh: PrefillInfoCache#_last_refresh.
  HeartbeatDaemon._bootstrap_key: HeartbeatDaemon#_bootstrap_key.
  PROTOCOL_VERSION.PROTOCOL_VERSION: PROTOCOL_VERSION.PROTOCOL_VERSION.
  PrefillInfo.protocol_version: PrefillInfo#protocol_version.
  RegisterPrefillRequest.protocol_version: RegisterPrefillRequest#protocol_version.
  build_app._auth_mw: build_app()._auth_mw().
  BootstrapServer.registry: BootstrapServer#registry.
  BootstrapServer.host: BootstrapServer#host().
  BootstrapServer.port: BootstrapServer#port().
  BootstrapServer.is_started: BootstrapServer#is_started().
  BootstrapClient.base_url: BootstrapClient#base_url().
  PrefillInfoCache.__init__: PrefillInfoCache#__init__().
  HEARTBEAT_TTL_SECONDS: HEARTBEAT_TTL_SECONDS.
  PrefillInfo.bootstrap_key: PrefillInfo#bootstrap_key.
  PrefillInfo.to_dict: PrefillInfo#to_dict().
  check_prefill_compat: check_prefill_compat().
  HeartbeatRequest.bootstrap_key: HeartbeatRequest#bootstrap_key.
  BootstrapServer: BootstrapServer#
  BootstrapClient._register_retry_delay_s: BootstrapClient#_register_retry_delay_s.
  PrefillInfoCache._clock: PrefillInfoCache#_clock.
  PrefillInfoCache._by_key: PrefillInfoCache#_by_key.
  PrefillInfoCache._refresh_failures: PrefillInfoCache#_refresh_failures.
  PrefillInfoCache._last_fail_log: PrefillInfoCache#_last_fail_log.
  HeartbeatDaemon: HeartbeatDaemon#
  HeartbeatDaemon._interval_s: HeartbeatDaemon#_interval_s.
  MIN_COMPATIBLE_VERSION.MIN_COMPATIBLE_VERSION: MIN_COMPATIBLE_VERSION.MIN_COMPATIBLE_VERSION.
  RegisterPrefillRequest: RegisterPrefillRequest#
  HeartbeatRequest: HeartbeatRequest#
  UnregisterPrefillRequest: UnregisterPrefillRequest#
  UnregisterPrefillRequest.bootstrap_key: UnregisterPrefillRequest#bootstrap_key.
  _Registry.clock: _Registry#clock.
  BootstrapServer._log_level: BootstrapServer#_log_level.
  BootstrapServer.app: BootstrapServer#app.
  BootstrapClient._shared_secret: BootstrapClient#_shared_secret.
  PrefillInfoCache._client: PrefillInfoCache#_client.
  PrefillInfoCache._refresh_interval_s: PrefillInfoCache#_refresh_interval_s.
  PrefillInfoCache._lock: PrefillInfoCache#_lock.
  HeartbeatDaemon._client: HeartbeatDaemon#_client.
  PrefillInfo.host: PrefillInfo#host.
  PrefillInfo.transfer_port: PrefillInfo#transfer_port.
  PrefillInfo.side_channel_port: PrefillInfo#side_channel_port.
  PrefillInfo.tp_rank: PrefillInfo#tp_rank.
  PrefillInfo.tp_size: PrefillInfo#tp_size.
  PrefillInfo.system_dp_rank: PrefillInfo#system_dp_rank.
  PrefillInfo.jax_process_index: PrefillInfo#jax_process_index.
  PrefillInfo.jax_process_count: PrefillInfo#jax_process_count.
  PrefillInfo.page_size: PrefillInfo#page_size.
  PrefillInfo.kv_dtype: PrefillInfo#kv_dtype.
  RegisterPrefillRequest.bootstrap_key: RegisterPrefillRequest#bootstrap_key.
  RegisterPrefillRequest.host: RegisterPrefillRequest#host.
  RegisterPrefillRequest.transfer_port: RegisterPrefillRequest#transfer_port.
  RegisterPrefillRequest.side_channel_port: RegisterPrefillRequest#side_channel_port.
  RegisterPrefillRequest.tp_rank: RegisterPrefillRequest#tp_rank.
  RegisterPrefillRequest.tp_size: RegisterPrefillRequest#tp_size.
  RegisterPrefillRequest.system_dp_rank: RegisterPrefillRequest#system_dp_rank.
  RegisterPrefillRequest.jax_process_index: RegisterPrefillRequest#jax_process_index.
  RegisterPrefillRequest.jax_process_count: RegisterPrefillRequest#jax_process_count.
  RegisterPrefillRequest.page_size: RegisterPrefillRequest#page_size.
  RegisterPrefillRequest.kv_dtype: RegisterPrefillRequest#kv_dtype.
  build_app.health: build_app().health().
  build_app.metrics: build_app().metrics().
  BootstrapServer.__init__: BootstrapServer#__init__().
  BootstrapClient.__init__: BootstrapClient#__init__().
---
# Module: [`python/sgl_jax/srt/disaggregation/bootstrap.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py)

## Classes
### `BootstrapClient`
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:397`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L397)
- doc: Stateless HTTP client for the bootstrap server.
- signature: `class BootstrapClient:`
- members:
  - `base_url(self)` — [`L423`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L423)
  - `get_prefill_info(self, bootstrap_room: int)` — [`L520`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L520)
  - `health(self)` — [`L431`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L431)
  - `heartbeat(self, bootstrap_key: str)` — [`L493`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L493)
  - `list_prefills(self)` — [`L511`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L511)
  - `register_prefill(self, bootstrap_key: str, host: str, transfer_port: int, side_channel_port: int, *, tp_rank: int = 0, tp_size: int = 1, system_dp_rank: int = 0, jax_process_index: int = 0, jax_process_count: int = 1, protocol_version: int = PROTOCOL_VERSION, page_size: int = 0, kv_dtype: str = "")` — [`L435`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L435)
  - `unregister_prefill(self, bootstrap_key: str)` — [`L502`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L502)
- protocol/private: `__init__`[`L400`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L400), `_base_url`[`L411`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L411), `_client`[`L420`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L420), `_headers`[`L426`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L426), `_register_retries`[`L413`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L413), `_register_retry_delay_s`[`L414`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L414), `_shared_secret`[`L415`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L415), `_timeout_s`[`L412`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L412)
- uses (calls/refs, reference-scoped): [`logger`](bootstrap.md#logger), [`_reject_if_below_protocol_floor`](bootstrap.md#_reject_if_below_protocol_floor), [`PROTOCOL_VERSION`](bootstrap.md#PROTOCOL_VERSION.PROTOCOL_VERSION), [`bearer_header`](pd_auth.md#bearer_header), `status_code`
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`_shutdown`](runtime.md#_make_disagg_shutdown._shutdown), [`_loop`](bootstrap.md#HeartbeatDaemon._loop), [`_refresh_locked`](bootstrap.md#PrefillInfoCache._refresh_locked), [`disagg_bootstrap_client`](decode.md#SchedulerDisaggregationDecodeMixin.disagg_bootstrap_client), [`_pick_prefill_peer_for_this_host`](decode.md#SchedulerDisaggregationDecodeMixin._pick_prefill_peer_for_this_host), [`__init__`](bootstrap.md#HeartbeatDaemon.__init__), [`__init__`](bootstrap.md#PrefillInfoCache.__init__)

### `BootstrapServer`
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:307`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L307)
- doc: Runs the bootstrap FastAPI app in a background uvicorn thread.
- signature: `class BootstrapServer:`
- members:
  - `host(self)` — [`L327`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L327)
  - `is_started(self)` — [`L335`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L335)
  - `port(self)` — [`L331`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L331)
  - `start(self)` — [`L338`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L338)
  - `stop(self)` — [`L363`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L363)
  - `app` — [`L321`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L321)
  - `registry` — [`L321`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L321)
- protocol/private: `__init__`[`L310`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L310), `_host`[`L318`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L318), `_log_level`[`L320`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L320), `_port`[`L319`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L319), `_server`[`L322`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L322), `_started`[`L324`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L324), `_thread`[`L323`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L323), `_wait_until_ready`[`L380`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L380)
- uses (calls/refs, reference-scoped): [`build_app`](bootstrap.md#build_app), [`logger`](bootstrap.md#logger), `status_code`, `should_exit`
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring)

### `HeartbeatDaemon`
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:620`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L620)
- doc: Background thread that heartbeats the bootstrap server on
- signature: `class HeartbeatDaemon:`
- members:
  - `start(self)` — [`L639`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L639)
  - `stop(self)` — [`L651`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L651)
- protocol/private: `__init__`[`L626`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L626), `_bootstrap_key`[`L633`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L633), `_client`[`L632`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L632), `_interval_s`[`L634`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L634), `_loop`[`L660`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L660), `_started`[`L637`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L637), `_stop_event`[`L635`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L635), `_thread`[`L636`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L636)
- uses (calls/refs, reference-scoped): [`heartbeat`](bootstrap.md#BootstrapClient.heartbeat), [`BootstrapClient`](bootstrap.md#BootstrapClient), [`logger`](bootstrap.md#logger), [`HEARTBEAT_INTERVAL_SECONDS`](bootstrap.md#HEARTBEAT_INTERVAL_SECONDS)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring)

### `HeartbeatRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L146)
- signature: `class HeartbeatRequest(BaseModel):`
- members:
  - `bootstrap_key` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L147)
- used by: [`heartbeat`](bootstrap.md#build_app.heartbeat)

### `PrefillInfo`
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L43)
- doc: One prefill worker's connection info as seen by decode workers.
- signature: `class PrefillInfo:`
- members:
  - `to_dict(self)` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L62)
  - `bootstrap_key` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L46)
  - `host` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L47)
  - `jax_process_count` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L54)
  - `jax_process_index` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L53)
  - `kv_dtype` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L60)
  - `page_size` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L59)
  - `protocol_version` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L55)
  - `side_channel_port` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L49)
  - `system_dp_rank` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L52)
  - `tp_rank` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L50)
  - `tp_size` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L51)
  - `transfer_port` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L48)
- uses (calls/refs, reference-scoped): [`PROTOCOL_VERSION`](bootstrap.md#PROTOCOL_VERSION.PROTOCOL_VERSION)
- used by: [`register`](bootstrap.md#_Registry.register), [`prefills`](bootstrap.md#_Registry.prefills), [`list_all`](bootstrap.md#_Registry.list_all), [`pick_for_room`](bootstrap.md#_Registry.pick_for_room), [`register_prefill`](bootstrap.md#build_app.register_prefill), [`get_prefill_info`](bootstrap.md#build_app.get_prefill_info), [`list_prefills`](bootstrap.md#build_app.list_prefills)

### `PrefillInfoCache`
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:534`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L534)
- doc: Decode-side cache of the prefill registry.
- signature: `class PrefillInfoCache:`
- members:
  - `pick_for_room(self, bootstrap_room: int)` — [`L577`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L577) — Return prefill info for ``bootstrap_room``, or ``None`` if no
- protocol/private: `__init__`[`L545`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L545), `_by_key`[`L556`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L556), `_client`[`L552`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L552), `_clock`[`L554`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L554), `_last_fail_log`[`L562`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L562), `_last_refresh`[`L559`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L559), `_lock`[`L555`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L555), `_pick_locked`[`L571`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L571), `_refresh_failures`[`L561`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L561), `_refresh_interval_s`[`L553`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L553), `_refresh_locked`[`L564`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L564), `_sorted_keys`[`L557`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L557)
- uses (calls/refs, reference-scoped): [`list_prefills`](bootstrap.md#BootstrapClient.list_prefills), [`BootstrapClient`](bootstrap.md#BootstrapClient), [`logger`](bootstrap.md#logger), [`_reject_if_below_protocol_floor`](bootstrap.md#_reject_if_below_protocol_floor)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`process_input_requests_disagg_decode`](decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`disagg_prefill_info_cache`](decode.md#SchedulerDisaggregationDecodeMixin.disagg_prefill_info_cache)

### `RegisterPrefillRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L131)
- signature: `class RegisterPrefillRequest(BaseModel):`
- members:
  - `bootstrap_key` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L132)
  - `host` — [`L133`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L133)
  - `jax_process_count` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L140)
  - `jax_process_index` — [`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L139)
  - `kv_dtype` — [`L143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L143)
  - `page_size` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L142)
  - `protocol_version` — [`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L141)
  - `side_channel_port` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L135)
  - `system_dp_rank` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L138)
  - `tp_rank` — [`L136`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L136)
  - `tp_size` — [`L137`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L137)
  - `transfer_port` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L134)
- uses (calls/refs, reference-scoped): [`PROTOCOL_VERSION`](bootstrap.md#PROTOCOL_VERSION.PROTOCOL_VERSION)
- used by: [`register_prefill`](bootstrap.md#build_app.register_prefill)

### `UnregisterPrefillRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L150)
- signature: `class UnregisterPrefillRequest(BaseModel):`
- members:
  - `bootstrap_key` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L151)
- used by: [`unregister_prefill`](bootstrap.md#build_app.unregister_prefill)

### `_Registry`
- def: [`python/sgl_jax/srt/disaggregation/bootstrap.py:155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L155)
- doc: In-memory state for the FastAPI app. Carries its own lock.
- signature: `class _Registry:`
- members:
  - `heartbeat(self, key: str)` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L174)
  - `list_all(self)` — [`L200`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L200)
  - `now(self)` — [`L164`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L164)
  - `pick_for_room(self, bootstrap_room: int)` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L205)
  - `register(self, info: PrefillInfo)` — [`L167`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L167)
  - `unregister(self, key: str)` — [`L181`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L181)
  - `clock` — [`L162`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L162)
  - `last_seen` — [`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L159)
  - `lock` — [`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L160)
  - `prefills` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L158)
  - `ttl_seconds` — [`L161`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L161)
- protocol/private: `_evict_stale_locked`[`L190`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L190)
- uses (calls/refs, reference-scoped): [`_set_registry_size`](bootstrap.md#_set_registry_size), [`PrefillInfo`](bootstrap.md#PrefillInfo), [`HEARTBEAT_TTL_SECONDS`](bootstrap.md#HEARTBEAT_TTL_SECONDS), [`bootstrap_key`](bootstrap.md#PrefillInfo.bootstrap_key)
- used by: [`heartbeat`](bootstrap.md#build_app.heartbeat), [`register_prefill`](bootstrap.md#build_app.register_prefill), [`unregister_prefill`](bootstrap.md#build_app.unregister_prefill), [`build_app`](bootstrap.md#build_app), [`get_prefill_info`](bootstrap.md#build_app.get_prefill_info), [`list_prefills`](bootstrap.md#build_app.list_prefills)

## Functions
- `_auth_mw(request: Request, call_next)` — [`L238`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L238)
- `_reject_if_below_protocol_floor(info: dict[str, object])` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L66) — Raise ``RuntimeError`` if a prefill peer reports a protocol version
- `_set_registry_size(n: int)` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L30) — Mirror the registry's current count into Prometheus. Best-effort —
- `build_app(registry: _Registry | None = None, *, shared_secret: str | None = None)` — [`L215`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L215) — Build the FastAPI app + return its registry.
- `check_prefill_compat(info: dict[str, object], *, local_page_size: int, local_kv_dtype: str)` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L102) — Raise ``ValueError`` if the prefill peer's KV layout is incompatible.
- `get_prefill_info(bootstrap_room: int)` — [`L281`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L281)
- `health()` — [`L253`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L253)
- `heartbeat(req: HeartbeatRequest)` — [`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L263)
- `list_prefills()` — [`L277`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L277)
- `metrics()` — [`L298`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L298)
- `register_prefill(req: RegisterPrefillRequest)` — [`L257`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L257)
- `resolve_kv_dtype_name(dtype: object)` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L83) — Canonical dtype name for KV-layout compatibility advertising.
- `unregister_prefill(req: UnregisterPrefillRequest)` — [`L272`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L272)

## Module values
- `HEARTBEAT_INTERVAL_SECONDS` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L22)
- `HEARTBEAT_TTL_SECONDS` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L20)
- `MIN_COMPATIBLE_VERSION` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L27)
- `PROTOCOL_VERSION` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L26)
- `logger` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/bootstrap.py#L17)

