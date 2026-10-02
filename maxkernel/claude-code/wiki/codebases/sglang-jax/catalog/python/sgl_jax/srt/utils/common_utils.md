---
title: 'Module: python/sgl_jax/srt/utils/common_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/utils/common_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.utils.common_utils`/
symbols:
  get_bool_env_var: get_bool_env_var().
  get_zmq_socket: get_zmq_socket().
  cdiv: cdiv().
  RWLock.acquire_reader: RWLock#acquire_reader().
  RWLock.acquire_writer: RWLock#acquire_writer().
  logger: logger.
  pad_to_bucket: pad_to_bucket().
  lru_cache_frozenset._to_hashable: lru_cache_frozenset()._to_hashable().
  configure_logger: configure_logger().
  RWLock.release_reader: RWLock#release_reader().
  RWLock.release_writer: RWLock#release_writer().
  ConcurrentCounter.increment: ConcurrentCounter#increment().
  ConcurrentCounter.decrement: ConcurrentCounter#decrement().
  RWLock._cond: RWLock#_cond.
  is_remote_url: is_remote_url().
  ConcurrentCounter._condition: ConcurrentCounter#_condition.
  ConcurrentCounter.wait_for: ConcurrentCounter#wait_for().
  dataclass_to_string_truncated: dataclass_to_string_truncated().
  kill_itself_when_parent_died: kill_itself_when_parent_died().
  lru_cache_frozenset: lru_cache_frozenset().
  _logger_initialized: _logger_initialized.
  next_power_of_2: next_power_of_2().
  RWLock: RWLock#
  RWLock._lock: RWLock#_lock.
  _ReaderLock.__aenter__: _ReaderLock#__aenter__().
  _ReaderLock.__aexit__: _ReaderLock#__aexit__().
  _WriterLock.__aenter__: _WriterLock#__aenter__().
  _WriterLock.__aexit__: _WriterLock#__aexit__().
  pyspy_dump_schedulers: pyspy_dump_schedulers().
  lru_cache_frozenset.decorator.wrapper: lru_cache_frozenset().decorator().wrapper().
  RWLock.reader_lock: RWLock#reader_lock().
  RWLock.writer_lock: RWLock#writer_lock().
  RWLock._readers: RWLock#_readers.
  RWLock._writer_active: RWLock#_writer_active.
  ConcurrentCounter: ConcurrentCounter#
  ConcurrentCounter._count: ConcurrentCounter#_count.
  set_ulimit: set_ulimit().
  set_prometheus_multiproc_dir: set_prometheus_multiproc_dir().
  add_prometheus_middleware: add_prometheus_middleware().
  prepare_model_and_tokenizer: prepare_model_and_tokenizer().
  launch_dummy_health_check_server: launch_dummy_health_check_server().
  lru_cache_frozenset.decorator: lru_cache_frozenset().decorator().
  ConcurrentCounter.wait_for_zero: ConcurrentCounter#wait_for_zero().
  RWLock._waiting_writers: RWLock#_waiting_writers.
  SUPPORTED_LORA_TARGET_MODULES: SUPPORTED_LORA_TARGET_MODULES.
  retry: retry().
  _ReaderLock.__init__: _ReaderLock#__init__().
  _WriterLock.__init__: _WriterLock#__init__().
  ConcurrentCounter.value: ConcurrentCounter#value().
  PRECOMPILE_DEFAULT_TOKEN_PADDINGS: PRECOMPILE_DEFAULT_TOKEN_PADDINGS.
  PRECOMPILE_DEFAULT_BS_PADDINGS: PRECOMPILE_DEFAULT_BS_PADDINGS.
  align_bs_for_fused_ep: align_bs_for_fused_ep().
  _warned_bool_env_var_keys: _warned_bool_env_var_keys.
  set_random_seed: set_random_seed().
  get_zmq_socket.set_send_opt: get_zmq_socket().set_send_opt().
  get_zmq_socket.set_recv_opt: get_zmq_socket().set_recv_opt().
  nullable_str: nullable_str().
  is_valid_ipv6_address: is_valid_ipv6_address().
  _ReaderLock._rwlock: _ReaderLock#_rwlock.
  _WriterLock._rwlock: _WriterLock#_rwlock.
  LORA_TARGET_ALL_MODULES: LORA_TARGET_ALL_MODULES.
  kill_process_tree: kill_process_tree().
  add_api_key_middleware: add_api_key_middleware().
  _logger_lock: _logger_lock.
  delete_directory: delete_directory().
  set_uvicorn_logging_configs: set_uvicorn_logging_configs().
  _ReaderLock: _ReaderLock#
  _WriterLock: _WriterLock#
  add_api_key_middleware.authentication: add_api_key_middleware().authentication().
  launch_dummy_health_check_server.health: launch_dummy_health_check_server().health().
  launch_dummy_health_check_server.health_generate: launch_dummy_health_check_server().health_generate().
  RWLock.__init__: RWLock#__init__().
  ConcurrentCounter.__init__: ConcurrentCounter#__init__().
  get_or_create_loop: get_or_create_loop().
---
# Module: [`python/sgl_jax/srt/utils/common_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py)

## Classes
### `ConcurrentCounter`
- def: [`python/sgl_jax/srt/utils/common_utils.py:663`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L663)
- doc: An asynchronous counter for managing concurrent tasks that need
- signature: `class ConcurrentCounter:`
- members:
  - `__init__(self, initial: int = 0)` — [`L672`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L672) — Initialize the counter with an optional initial value.
  - `decrement(self, n: int = 1, notify_all: bool = True)` — [`L708`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L708) — Atomically decrement the counter by a given amount and notify all waiters.
  - `increment(self, n: int = 1, notify_all: bool = True)` — [`L695`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L695) — Atomically increment the counter by a given amount and notify all waiters.
  - `value(self)` — [`L682`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L682) — Return the current value of the counter.
  - `wait_for(self, condition: Callable[[int], bool])` — [`L721`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L721) — Asynchronously wait until the counter satisfies a given condition.
  - `wait_for_zero(self)` — [`L736`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L736) — Asynchronously wait until the counter reaches zero.
- protocol/private: `_condition`[`L680`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L680), `_count`[`L679`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L679)
- used by: [`_register_adapter`](../lora/lora_registry.md#LoRARegistry._register_adapter), [`_counters`](../lora/lora_registry.md#LoRARegistry._counters), [`acquire`](../lora/lora_registry.md#LoRARegistry.acquire), [`release`](../lora/lora_registry.md#LoRARegistry.release), [`wait_for_unload`](../lora/lora_registry.md#LoRARegistry.wait_for_unload)

### `RWLock`
- def: [`python/sgl_jax/srt/utils/common_utils.py:548`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L548)
- doc: An async reader-writer lock for LoRA adapter management.
- signature: `class RWLock:`
- members:
  - `acquire_reader(self)` — [`L589`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L589)
  - `acquire_writer(self)` — [`L605`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L605)
  - `reader_lock(self)` — [`L568`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L568) — A context manager for acquiring a shared (reader) lock.
  - `release_reader(self)` — [`L597`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L597)
  - `release_writer(self)` — [`L618`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L618)
  - `writer_lock(self)` — [`L579`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L579) — A context manager for acquiring an exclusive (writer) lock.
- protocol/private: `__init__`[`L551`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L551), `_cond`[`L556`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L556), `_lock`[`L553`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L553), `_readers`[`L559`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L559), `_waiting_writers`[`L565`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L565), `_writer_active`[`L562`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L562)
- uses (calls/refs, reference-scoped): [`_ReaderLock`](common_utils.md#_ReaderLock), [`_WriterLock`](common_utils.md#_WriterLock)
- used by: [`acquire`](../lora/lora_registry.md#LoRARegistry.acquire), [`register`](../lora/lora_registry.md#LoRARegistry.register), [`release`](../lora/lora_registry.md#LoRARegistry.release), [`unregister`](../lora/lora_registry.md#LoRARegistry.unregister), [`_registry_lock`](../lora/lora_registry.md#LoRARegistry._registry_lock), [`__aenter__`](common_utils.md#_ReaderLock.__aenter__), [`__aenter__`](common_utils.md#_WriterLock.__aenter__), [`__aexit__`](common_utils.md#_ReaderLock.__aexit__), [`__aexit__`](common_utils.md#_WriterLock.__aexit__), [`__init__`](common_utils.md#_ReaderLock.__init__), [`__init__`](common_utils.md#_WriterLock.__init__)

### `_ReaderLock`
- def: [`python/sgl_jax/srt/utils/common_utils.py:625`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L625)
- protocol/private: `__aenter__`[`L629`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L629), `__aexit__`[`L633`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L633), `__init__`[`L626`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L626), `_rwlock`[`L627`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L627)
- uses (calls/refs, reference-scoped): [`acquire_reader`](common_utils.md#RWLock.acquire_reader), [`release_reader`](common_utils.md#RWLock.release_reader), [`RWLock`](common_utils.md#RWLock)
- used by: [`reader_lock`](common_utils.md#RWLock.reader_lock)

### `_WriterLock`
- def: [`python/sgl_jax/srt/utils/common_utils.py:637`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L637)
- protocol/private: `__aenter__`[`L641`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L641), `__aexit__`[`L645`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L645), `__init__`[`L638`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L638), `_rwlock`[`L639`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L639)
- uses (calls/refs, reference-scoped): [`acquire_writer`](common_utils.md#RWLock.acquire_writer), [`release_writer`](common_utils.md#RWLock.release_writer), [`RWLock`](common_utils.md#RWLock)
- used by: [`writer_lock`](common_utils.md#RWLock.writer_lock)

## Functions
- `_to_hashable(o)` — [`L488`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L488)
- `add_api_key_middleware(app, api_key: str)` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L173)
- `add_prometheus_middleware(app)` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L206)
- `align_bs_for_fused_ep(bs: int, ep_size: int)` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L43) — Down-align a global batch size so it is launchable by fused_ep_moe.
- `authentication(request, call_next)` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L175)
- `cdiv(a, b)` — [`L528`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L528)
- `configure_logger(server_args, prefix: str = "")` — [`L234`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L234)
- `dataclass_to_string_truncated(data, max_length=2048, skip_names: set[str] | None = None)` — [`L317`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L317)
- `decorator(func)` — [`L505`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L505)
- `delete_directory(dirpath)` — [`L309`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L309)
- `get_bool_env_var(name: str, default: str = "false")` — [`L88`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L88)
- `get_or_create_loop()` — [`L746`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L746)
- `get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool)` — [`L273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L273)
- `health()` — [`L413`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L413) — Check the health of the http server.
- `health_generate()` — [`L418`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L418) — Check the health of the http server.
- `is_remote_url(url: str | Path)` — [`L443`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L443) — Check if the URL is a remote URL of the format:
- `is_valid_ipv6_address(address: str)` — [`L396`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L396)
- `kill_itself_when_parent_died()` — [`L375`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L375)
- `kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int | None = None)` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L113) — Kill the process and all its child processes.
- `launch_dummy_health_check_server(host, port)` — [`L404`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L404)
- `lru_cache_frozenset(maxsize=128)` — [`L487`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L487)
- `next_power_of_2(x: int)` — [`L533`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L533) — Finds the smallest power of 2 >= x using bit manipulation.
- `nullable_str(val: str)` — [`L357`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L357)
- `pad_to_bucket(value: int, buckets: list[int])` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L64) — Return the smallest bucket >= value and its index.
- `prepare_model_and_tokenizer(model_path: str, tokenizer_path: str)` — [`L219`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L219)
- `pyspy_dump_schedulers()` — [`L363`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L363) — py-spy dump on all scheduler in a local node.
- `retry(fn, max_retry: int, initial_delay: float = 2, max_delay: float = 60, should_retry: Callable[[Any], bool] = lambda e: True)` — [`L456`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L456)
- `set_prometheus_multiproc_dir()` — [`L187`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L187)
- `set_random_seed(seed: int)` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L107) — Set the random seed for all libraries.
- `set_recv_opt()` — [`L287`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L287)
- `set_send_opt()` — [`L283`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L283)
- `set_ulimit(target_soft_limit=65535)` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L151)
- `set_uvicorn_logging_configs()` — [`L385`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L385)
- `wrapper(*args, **kwargs)` — [`L509`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L509)

## Module values
- `LORA_TARGET_ALL_MODULES` — [`L660`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L660)
- `PRECOMPILE_DEFAULT_BS_PADDINGS` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L40)
- `PRECOMPILE_DEFAULT_TOKEN_PADDINGS` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L39)
- `SUPPORTED_LORA_TARGET_MODULES` — [`L650`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L650)
- `_logger_initialized` — [`L231`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L231)
- `_logger_lock` — [`L230`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L230)
- `_warned_bool_env_var_keys` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L85)
- `logger` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/common_utils.py#L37)

