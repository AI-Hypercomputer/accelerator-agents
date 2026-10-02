---
title: 'Module: jax/_src/xla_bridge.py'
type: catalog
provenance: extracted
module: jax/_src/xla_bridge.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.xla_bridge`/
symbols:
  backends: backends().
  get_backend: get_backend().
  local_devices: local_devices().
  process_count: process_count().
  _options_from_jax_configs: _options_from_jax_configs().
  _discover_and_register_pjrt_plugins: _discover_and_register_pjrt_plugins().
  logger: logger.
  register_backend_factory: register_backend_factory().
  devices: devices().
  make_cpu_client: make_cpu_client().
  register_plugin: register_plugin().
  _get_backend_uncached: _get_backend_uncached().
  make_pjrt_c_api_client: make_pjrt_c_api_client().
  _init_backend: _init_backend().
  _make_transfer_server_factory: _make_transfer_server_factory().
  make_tpu_client: make_tpu_client().
  register_plugin_callbacks: register_plugin_callbacks().
  _clear_backends: _clear_backends().
  _backends._backends: _backends._backends.
  tpu_client_timer_callback: tpu_client_timer_callback().
  register_pjrt_plugin_factories_from_env: register_pjrt_plugin_factories_from_env().
  _suggest_missing_backends: _suggest_missing_backends().
  e: e.
  _alias: _alias.
  device_count: device_count().
  make_pjrt_topology: make_pjrt_topology().
  _default_backend._default_backend: _default_backend._default_backend.
  local_device_count: local_device_count().
  _backend_factories._backend_factories: _backend_factories._backend_factories.
  known_platforms: known_platforms().
  process_index: process_index().
  canonicalize_platform: canonicalize_platform().
  _validate_backend_not_initialized: _validate_backend_not_initialized().
  _backend_errors._backend_errors: _backend_errors._backend_errors.
  register_backend_initialization_hook: register_backend_initialization_hook().
  jax_plugins.jax_plugins: jax_plugins.jax_plugins.
  backends_are_initialized: backends_are_initialized().
  num_cpu_devices: num_cpu_devices.
  _backend_lock: _backend_lock.
  discover_pjrt_plugins: discover_pjrt_plugins().
  cpu_get_local_topology_timeout_minutes: cpu_get_local_topology_timeout_minutes.
  cpu_get_global_topology_timeout_minutes: cpu_get_global_topology_timeout_minutes.
  default_backend: default_backend().
  _topology_factories._topology_factories: _topology_factories._topology_factories.
  FORCE_DCN_CROSS_HOST_TRANSFERS: FORCE_DCN_CROSS_HOST_TRANSFERS.
  SORT_DEVICES_BY_PROCESS_INDEX: SORT_DEVICES_BY_PROCESS_INDEX.
  CROSS_HOST_TRANSFER_SOCKET_ADDRESS: CROSS_HOST_TRANSFER_SOCKET_ADDRESS.
  CROSS_HOST_TRANSPORT_ADDRESSES: CROSS_HOST_TRANSPORT_ADDRESSES.
  CROSS_HOST_TRANSFER_TIMEOUT_SECONDS: CROSS_HOST_TRANSFER_TIMEOUT_SECONDS.
  CROSS_HOST_TRANSFER_TRANSFER_SIZE: CROSS_HOST_TRANSFER_TRANSFER_SIZE.
  process_indices: process_indices().
  BackendRegistration.c_api: BackendRegistration#c_api.
  OptionsDict: OptionsDict.
  _plugins_registered._plugins_registered: _plugins_registered._plugins_registered.
  _XLA_BACKEND: _XLA_BACKEND.
  _PLATFORM_NAME: _PLATFORM_NAME.
  CUDA_VISIBLE_DEVICES: CUDA_VISIBLE_DEVICES.
  _ROCM_VISIBLE_DEVICES: _ROCM_VISIBLE_DEVICES.
  _ONEAPI_VISIBLE_DEVICES: _ONEAPI_VISIBLE_DEVICES.
  MOCK_NUM_GPU_PROCESSES: MOCK_NUM_GPU_PROCESSES.
  MOCK_GPU_TOPOLOGY: MOCK_GPU_TOPOLOGY.
  _CPU_ENABLE_ASYNC_DISPATCH: _CPU_ENABLE_ASYNC_DISPATCH.
  BackendRegistration.factory: BackendRegistration#factory.
  _get_pjrt_plugin_names_and_library_paths: _get_pjrt_plugin_names_and_library_paths().
  expand_platform_alias: expand_platform_alias().
  host_id: host_id().
  host_count: host_count().
  host_ids: host_ids().
  _at_fork_handler_installed: _at_fork_handler_installed.
  BackendFactory: BackendFactory.
  TopologyFactory: TopologyFactory.
  _platform_aliases: _platform_aliases.
  _alias_to_platforms._alias_to_platforms: _alias_to_platforms._alias_to_platforms.
  BACKEND_TARGET: BACKEND_TARGET.
  is_known_platform: is_known_platform().
  get_device_backend: get_device_backend().
  backend_pjrt_c_api_version: backend_pjrt_c_api_version().
  backend_xla_version: backend_xla_version().
  backend_stablehlo_version: backend_stablehlo_version().
  using_pjrt_c_api: using_pjrt_c_api().
  make_pjrt_tpu_topology: make_pjrt_tpu_topology().
  BackendRegistration: BackendRegistration#
  _plugin_callback_lock: _plugin_callback_lock.
  _plugin_callbacks._plugin_callbacks: _plugin_callbacks._plugin_callbacks.
  _backend_initialization_hooks._backend_initialization_hooks: _backend_initialization_hooks._backend_initialization_hooks.
  _nonexperimental_plugins._nonexperimental_plugins: _nonexperimental_plugins._nonexperimental_plugins.
  _at_fork: _at_fork().
  _NameValueMapping: _NameValueMapping.
  tpu_client_timer_callback._log_warning: tpu_client_timer_callback()._log_warning().
  BackendRegistration.priority: BackendRegistration#priority.
  BackendRegistration.fail_quietly: BackendRegistration#fail_quietly.
  BackendRegistration.experimental: BackendRegistration#experimental.
  _plugin_lock: _plugin_lock.
  get_num_nodes_from_gpu_topology: get_num_nodes_from_gpu_topology().
  _get_pjrt_plugin_config: _get_pjrt_plugin_config().
  _platform: _platform.
  MIN_COMPUTE_CAPABILITY: MIN_COMPUTE_CAPABILITY.
  FORCE_FORWARD_COMPAT_LOWERING_RUNTIMES.FORCE_FORWARD_COMPAT_LOWERING_RUNTIMES: FORCE_FORWARD_COMPAT_LOWERING_RUNTIMES.FORCE_FORWARD_COMPAT_LOWERING_RUNTIMES.
  _experimental_plugins._experimental_plugins: _experimental_plugins._experimental_plugins.
---
# Module: [`jax/_src/xla_bridge.py`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py)

## Classes
### `BackendRegistration`
- def: [`jax/_src/xla_bridge.py:254`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L254)
- signature: `class BackendRegistration:`
- members:
  - `c_api` — [`L273`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L273)
  - `experimental` — [`L270`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L270)
  - `factory` — [`L255`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L255)
  - `fail_quietly` — [`L264`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L264)
  - `priority` — [`L259`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L259)
- uses (calls/refs, reference-scoped): [`BackendFactory`](xla_bridge.md#BackendFactory)
- used by: [`backends`](xla_bridge.md#backends), [`_discover_and_register_pjrt_plugins`](xla_bridge.md#_discover_and_register_pjrt_plugins), [`register_backend_factory`](xla_bridge.md#register_backend_factory), [`_init_backend`](xla_bridge.md#_init_backend), [`register_plugin_callbacks`](xla_bridge.md#register_plugin_callbacks), [`_backend_factories`](xla_bridge.md#_backend_factories._backend_factories)

## Functions
- `_at_fork()` — [`L163`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L163)
- `_clear_backends()` — [`L900`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L900)
- `_discover_and_register_pjrt_plugins()` — [`L693`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L693)
- `_get_backend_uncached(platform: None | str | xla_client.Client = None)` — [`L942`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L942)
- `_get_pjrt_plugin_config(json_path: str)` — [`L425`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L425) — Gets PJRT plugin configuration from a json file.
- `_get_pjrt_plugin_names_and_library_paths(plugins_from_env: str)` — [`L396`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L396) — Gets the names and library paths of PJRT plugins to load from env var.
- `_init_backend(platform: str)` — [`L912`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L912)
- `_log_warning()` — [`L223`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L223)
- `_make_transfer_server_factory()` — [`L175`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L175) — Creates a transfer server interface factory.
- `_options_from_jax_configs(plugin_name)` — [`L514`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L514)
- `_suggest_missing_backends()` — [`L868`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L868)
- `_validate_backend_not_initialized(name, new_val)` — [`L1212`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1212)
- `backend_pjrt_c_api_version(platform=None)` — [`L1047`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1047) — Returns the PJRT C API version of the backend.
- `backend_stablehlo_version(platform=None)` — [`L1071`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1071) — Returns the StableHLO version of the backend.
- `backend_xla_version(platform=None)` — [`L1060`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1060) — Returns the XLA version of the backend.
- `backends()` — [`L799`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L799)
- `backends_are_initialized()` — [`L772`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L772) — Returns true if backends have already been initialized.
- `canonicalize_platform(platform: str)` — [`L742`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L742) — Replaces platform aliases with their concrete equivalent.
- `default_backend()` — [`L1042`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1042) — Returns the platform name of the default XLA backend.
- `device_count(backend: str | xla_client.Client | None = None)` — [`L985`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L985) — Returns the total number of devices.
- `devices(backend: str | xla_client.Client | None = None)` — [`L1014`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1014) — Returns a list of all devices for a given backend.
- `discover_pjrt_plugins()` — [`L446`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L446) — Discovers plugins in the namespace package `jax_plugins` and import them.
- `expand_platform_alias(platform: str)` — [`L763`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L763) — Expands, e.g., "gpu" to ["cuda", "rocm", "oneapi"].
- `get_backend(platform: None | str | xla_client.Client = None)` — [`L970`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L970)
- `get_device_backend(device: xla_client.Device | None = None)` — [`L976`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L976) — Returns the Backend associated with `device`, or the default Backend.
- `get_num_nodes_from_gpu_topology(topology: str)` — [`L380`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L380)
- `host_count(backend: str | xla_client.Client | None = None)` — [`L1150`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1150)
- `host_id(backend: str | xla_client.Client | None = None)` — [`L1133`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1133)
- `host_ids(backend: str | xla_client.Client | None = None)` — [`L1174`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1174)
- `is_known_platform(platform: str)` — [`L735`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L735)
- `known_platforms()` — [`L726`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L726)
- `local_device_count(backend: str | xla_client.Client | None = None)` — [`L1007`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1007) — Returns the number of devices addressable by this process.
- `local_devices(process_index: int | None = None, backend: str | xla_client.Client | None = None, host_id: int | None = None)` — [`L1083`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1083) — Like :py:func:`jax.devices`, but only returns devices local to a given process.
- `make_cpu_client(collectives: _jax.CpuCollectives | None = None)` — [`L327`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L327) — Creates a CPU client with the requested collectives implementation.
- `make_pjrt_c_api_client(plugin_name: str, options: OptionsDict | Callable[[], OptionsDict] | None = None)` — [`L557`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L557) — Creates a PjRt client for the given plugin.
- `make_pjrt_topology(platform: str, topology_name='', **kwargs)` — [`L1186`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1186)
- `make_pjrt_tpu_topology(topology_name='', **kwargs)` — [`L1196`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1196)
- `make_tpu_client(library_path: str | None = None, options: _NameValueMapping | None = None)` — [`L198`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L198) — Returns a TPU client. Defaults to allowing 32 in-flight computations.
- `process_count(backend: str | xla_client.Client | None = None)` — [`L1141`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1141) — Returns the number of JAX processes associated with the backend.
- `process_index(backend: str | xla_client.Client | None = None)` — [`L1113`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1113) — Returns the integer process index of this process.
- `process_indices(backend: str | xla_client.Client | None = None)` — [`L1157`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1157) — Returns the list of all JAX process indices associated with the backend.
- `register_backend_factory(name: str, factory: BackendFactory, *, priority: int = 0, fail_quietly: bool = True, experimental: bool = False, make_topology: TopologyFactory | None = None, c_api: Any | None = None)` — [`L312`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L312)
- `register_backend_initialization_hook(hook: Callable[[xla_client.Client], None])` — [`L289`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L289) — Registers a callback to run on all initialized and future backends.
- `register_pjrt_plugin_factories_from_env()` — [`L664`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L664) — Registers backend factories for PJRT plugins.
- `register_plugin(plugin_name: str, *, priority: int = 400, library_path: str | None = None, options: OptionsDict | Callable[[], OptionsDict] | None = None, c_api: Any | None = None, factory: BackendFactory | None = None, make_topology: TopologyFactory | None = None)` — [`L596`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L596) — Registers a backend factory for the PJRT plugin.
- `register_plugin_callbacks(callback)` — [`L778`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L778) — Registers a callback to be called with c_api after plugins discovery.
- `tpu_client_timer_callback(timer_secs: float)` — [`L222`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L222)
- `using_pjrt_c_api(backend=None)` — [`L1183`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1183)

## Module values
- `BACKEND_TARGET` — [`L70`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L70)
- `BackendFactory` — [`L250`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L250)
- `CROSS_HOST_TRANSFER_SOCKET_ADDRESS` — [`L131`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L131)
- `CROSS_HOST_TRANSFER_TIMEOUT_SECONDS` — [`L147`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L147)
- `CROSS_HOST_TRANSFER_TRANSFER_SIZE` — [`L156`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L156)
- `CROSS_HOST_TRANSPORT_ADDRESSES` — [`L138`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L138)
- `CUDA_VISIBLE_DEVICES` — [`L79`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L79)
- `FORCE_DCN_CROSS_HOST_TRANSFERS` — [`L116`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L116)
- `FORCE_FORWARD_COMPAT_LOWERING_RUNTIMES` — [`L62`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L62)
- `MIN_COMPUTE_CAPABILITY` — [`L64`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L64)
- `MOCK_GPU_TOPOLOGY` — [`L101`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L101)
- `MOCK_NUM_GPU_PROCESSES` — [`L95`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L95)
- `OptionsDict` — [`L554`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L554)
- `SORT_DEVICES_BY_PROCESS_INDEX` — [`L123`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L123)
- `TopologyFactory` — [`L251`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L251)
- `_CPU_ENABLE_ASYNC_DISPATCH` — [`L109`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L109)
- `_NameValueMapping` — [`L173`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L173)
- `_ONEAPI_VISIBLE_DEVICES` — [`L89`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L89)
- `_PLATFORM_NAME` — [`L75`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L75)
- `_ROCM_VISIBLE_DEVICES` — [`L84`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L84)
- `_XLA_BACKEND` — [`L67`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L67)
- `_alias` — [`L722`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L722)
- `_alias_to_platforms` — [`L721`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L721)
- `_at_fork_handler_installed` — [`L169`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L169)
- `_backend_errors` — [`L278`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L278)
- `_backend_factories` — [`L275`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L275)
- `_backend_initialization_hooks` — [`L286`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L286)
- `_backend_lock` — [`L279`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L279)
- `_backends` — [`L277`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L277)
- `_default_backend` — [`L276`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L276)
- `_experimental_plugins` — [`L310`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L310)
- `_nonexperimental_plugins` — [`L307`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L307)
- `_platform` — [`L722`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L722)
- `_platform_aliases` — [`L715`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L715)
- `_plugin_callback_lock` — [`L284`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L284)
- `_plugin_callbacks` — [`L283`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L283)
- `_plugin_lock` — [`L281`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L281)
- `_plugins_registered` — [`L280`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L280)
- `_topology_factories` — [`L282`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L282)
- `cpu_get_global_topology_timeout_minutes` — [`L1242`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1242)
- `cpu_get_local_topology_timeout_minutes` — [`L1231`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1231)
- `e` — [`L55`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L55)
- `jax_plugins` — [`L50`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L50)
- `logger` — [`L48`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L48)
- `num_cpu_devices` — [`L1221`](../../../../../../raw/code/jax/jax/_src/xla_bridge.py#L1221)

