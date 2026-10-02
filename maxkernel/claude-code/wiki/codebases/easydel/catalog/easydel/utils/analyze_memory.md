---
title: 'Module: easydel/utils/analyze_memory.py'
type: catalog
provenance: extracted
module: easydel/utils/analyze_memory.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.analyze_memory`/
symbols:
  MemoryMonitorServer._process_data: MemoryMonitorServer#_process_data().
  MemoryMonitorServer.start: MemoryMonitorServer#start().
  MemoryMonitorServer._analyze_trends: MemoryMonitorServer#_analyze_trends().
  SMPMemoryMonitor._monitor_loop: SMPMemoryMonitor#_monitor_loop().
  MemoryMonitorClient._monitor_loop: MemoryMonitorClient#_monitor_loop().
  SMPMemoryMonitor.history: SMPMemoryMonitor#history.
  MemoryMonitorClient.start_monitoring: MemoryMonitorClient#start_monitoring().
  MemoryMonitorClient._send_stats: MemoryMonitorClient#_send_stats().
  SMPMemoryMonitor.start_monitoring: SMPMemoryMonitor#start_monitoring().
  MemoryMonitorServer._run_server.ThreadedTCPServer.allow_reuse_address: MemoryMonitorServer#_run_server().ThreadedTCPServer#allow_reuse_address.
  logger: logger.
  SMPMemoryMonitor.check_all_devices: SMPMemoryMonitor#check_all_devices().
  _pandas_available: _pandas_available.
  MemoryMonitorServer.data_store: MemoryMonitorServer#data_store.
  MemoryMonitorClient.analyze_memory: MemoryMonitorClient#analyze_memory().
  SMPMemoryMonitor._monitor_thread: SMPMemoryMonitor#_monitor_thread.
  SMPMemoryMonitor.stop_monitoring: SMPMemoryMonitor#stop_monitoring().
  SMPMemoryMonitor.get_summary: SMPMemoryMonitor#get_summary().
  MemoryMonitorServer.get_device_stats: MemoryMonitorServer#get_device_stats().
  start_server: start_server().
  start_client: start_client().
  SMPMemoryMonitor: SMPMemoryMonitor#
  DeviceStats: DeviceStats#
  SMPMemoryMonitor.print_current_status: SMPMemoryMonitor#print_current_status().
  MemoryMonitorServer._cleanup_old_data: MemoryMonitorServer#_cleanup_old_data().
  SMPMemoryMonitor.running: SMPMemoryMonitor#running.
  MemoryMonitorClient.running: MemoryMonitorClient#running.
  SMPMemoryMonitor.get_device_history: SMPMemoryMonitor#get_device_history().
  SMPMemoryMonitor.print_history_summary: SMPMemoryMonitor#print_history_summary().
  MemoryMonitorServer._run_server.RequestHandler.handle: MemoryMonitorServer#_run_server().RequestHandler#handle().
  MemoryMonitorServer._run_server.ThreadedTCPServer.stats_queue: MemoryMonitorServer#_run_server().ThreadedTCPServer#stats_queue.
  MemoryMonitorClient.stop_monitoring: MemoryMonitorClient#stop_monitoring().
  SMPMemoryMonitor.check_interval: SMPMemoryMonitor#check_interval.
  DeviceStats.device_id: DeviceStats#device_id.
  DeviceStats.hostname: DeviceStats#hostname.
  DeviceStats.utilization_percent: DeviceStats#utilization_percent.
  DeviceStats.fragmentation_ratio: DeviceStats#fragmentation_ratio.
  MemoryMonitorServer.host: MemoryMonitorServer#host.
  MemoryMonitorServer.port: MemoryMonitorServer#port.
  MemoryMonitorServer.stats_queue: MemoryMonitorServer#stats_queue.
  MemoryMonitorServer.running: MemoryMonitorServer#running.
  MemoryMonitorServer.lock: MemoryMonitorServer#lock.
  MemoryMonitorClient.hostname: MemoryMonitorClient#hostname.
  SMPMemoryMonitor.quiet: SMPMemoryMonitor#quiet.
  SMPMemoryMonitor.analyze_device: SMPMemoryMonitor#analyze_device().
  MemoryMonitorServer: MemoryMonitorServer#
  MemoryMonitorServer._run_server: MemoryMonitorServer#_run_server().
  MemoryMonitorServer._run_server.RequestHandler: MemoryMonitorServer#_run_server().RequestHandler#
  MemoryMonitorServer._run_server.ThreadedTCPServer: MemoryMonitorServer#_run_server().ThreadedTCPServer#
  MemoryMonitorClient: MemoryMonitorClient#
  MemoryMonitorClient.server_host: MemoryMonitorClient#server_host.
  MemoryMonitorClient.server_port: MemoryMonitorClient#server_port.
  MemoryMonitorClient.interval: MemoryMonitorClient#interval.
  SMPMemoryMonitor.__init__: SMPMemoryMonitor#__init__().
  DeviceStats.timestamp: DeviceStats#timestamp.
  DeviceStats.peak_utilization_percent: DeviceStats#peak_utilization_percent.
  DeviceStats.allocation_efficiency: DeviceStats#allocation_efficiency.
  DeviceStats.memory_pressure: DeviceStats#memory_pressure.
  DeviceStats.raw_stats: DeviceStats#raw_stats.
  MemoryMonitorServer.__init__: MemoryMonitorServer#__init__().
  MemoryMonitorClient.__init__: MemoryMonitorClient#__init__().
---
# Module: [`easydel/utils/analyze_memory.py`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py)

## Classes
### `DeviceStats`
- def: [`easydel/utils/analyze_memory.py:215`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L215)
- doc: Dataclass to store device statistics.
- signature: `class DeviceStats:`
- members:
  - `allocation_efficiency` — [`L249`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L249)
  - `device_id` — [`L231`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L231)
  - `fragmentation_ratio` — [`L246`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L246)
  - `hostname` — [`L234`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L234)
  - `memory_pressure` — [`L252`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L252)
  - `peak_utilization_percent` — [`L243`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L243)
  - `raw_stats` — [`L255`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L255)
  - `timestamp` — [`L237`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L237)
  - `utilization_percent` — [`L240`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L240)
- used by: [`_analyze_trends`](analyze_memory.md#MemoryMonitorServer._analyze_trends), [`_send_stats`](analyze_memory.md#MemoryMonitorClient._send_stats), [`analyze_memory`](analyze_memory.md#MemoryMonitorClient.analyze_memory)

### `MemoryMonitorClient`
- def: [`easydel/utils/analyze_memory.py:352`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L352)
- doc: Client that monitors local JAX device memory and reports to a server.
- signature: `class MemoryMonitorClient:`
- members:
  - `_monitor_loop(self)` — [`L420`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L420) — Main monitoring loop
  - `_send_stats(self, stats: DeviceStats)` — [`L432`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L432) — Send statistics to the server
  - `analyze_memory(self, memory_stats: dict[str, tp.Any])` — [`L374`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L374) — Analyze memory statistics for a single device
  - `start_monitoring(self)` — [`L408`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L408) — Start monitoring memory usage
  - `stop_monitoring(self)` — [`L416`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L416) — Stop monitoring memory usage
  - `hostname` — [`L372`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L372)
  - `interval` — [`L370`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L370)
  - `running` — [`L371`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L371)
  - `server_host` — [`L368`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L368)
  - `server_port` — [`L369`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L369)
- protocol/private: `__init__`[`L367`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L367)
- uses (calls/refs, reference-scoped): [`logger`](analyze_memory.md#logger), [`DeviceStats`](analyze_memory.md#DeviceStats)
- used by: [`start_client`](analyze_memory.md#start_client)

### `MemoryMonitorServer`
- def: [`easydel/utils/analyze_memory.py:261`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L261)
- doc: TCP server that collects device memory statistics from remote clients.
- signature: `class MemoryMonitorServer:`
- members:
  - `_analyze_trends(self, stats: DeviceStats)` — [`L333`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L333) — Analyze memory usage trends and log warnings
  - `_cleanup_old_data(self)` — [`L328`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L328) — Keep only recent data to prevent memory issues
  - `get_device_stats(self, device_id=None)` — [`L344`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L344) — Get statistics for all devices or a specific device
  - `start(self)` — [`L284`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L284) — Start the monitoring server
  - `data_store` — [`L281`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L281)
  - `host` — [`L277`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L277)
  - `lock` — [`L282`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L282)
  - `port` — [`L278`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L278)
  - `running` — [`L280`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L280)
  - `stats_queue` — [`L279`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L279)
- protocol/private: `__init__`[`L276`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L276), `_process_data`[`L314`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L314), `_run_server`[`L298`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L298)
- uses (calls/refs, reference-scoped): [`logger`](analyze_memory.md#logger), [`DeviceStats`](analyze_memory.md#DeviceStats), [`device_id`](analyze_memory.md#DeviceStats.device_id), [`fragmentation_ratio`](analyze_memory.md#DeviceStats.fragmentation_ratio), [`hostname`](analyze_memory.md#DeviceStats.hostname), [`utilization_percent`](analyze_memory.md#DeviceStats.utilization_percent)
- used by: [`allow_reuse_address`](analyze_memory.md#MemoryMonitorServer._run_server.ThreadedTCPServer.allow_reuse_address), [`start_server`](analyze_memory.md#start_server), [`stats_queue`](analyze_memory.md#MemoryMonitorServer._run_server.ThreadedTCPServer.stats_queue)

### `RequestHandler`  ·  implements/extends BaseRequestHandler
- def: [`easydel/utils/analyze_memory.py:299`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L299)
- signature: `class RequestHandler(socketserver.BaseRequestHandler):`
- members:
  - `handle(self)` — [`L300`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L300)
- uses (calls/refs, reference-scoped): [`logger`](analyze_memory.md#logger)
- used by: [`allow_reuse_address`](analyze_memory.md#MemoryMonitorServer._run_server.ThreadedTCPServer.allow_reuse_address)

### `SMPMemoryMonitor`
- def: [`easydel/utils/analyze_memory.py:41`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L41)
- doc: Simple memory monitor for JAX devices.
- signature: `class SMPMemoryMonitor:`
- members:
  - `__init__(self, check_interval: int = 60, quiet: bool = False)` — [`L55`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L55) — Initialize the memory monitor.
  - `_monitor_loop(self)` — [`L106`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L106) — Internal monitoring loop.
  - `analyze_device(self, device_stats: dict, dev)` — [`L69`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L69) — Analyze memory stats for a single device.
  - `check_all_devices(self)` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L115) — Check memory usage on all available devices.
  - `get_device_history(self, device_id: str | None = None)` — [`L178`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L178) — Get memory history for a specific device or all devices.
  - `get_summary(self, format: str = "auto")` — [`L139`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L139) — Get a summary of memory usage history.
  - `print_current_status(self)` — [`L162`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L162) — Print current memory status for all devices.
  - `print_history_summary(self, n_entries: int = 5)` — [`L192`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L192) — Print a summary of recent memory usage without using pandas.
  - `start_monitoring(self)` — [`L91`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L91) — Start automatic memory monitoring.
  - `stop_monitoring(self)` — [`L99`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L99) — Stop automatic memory monitoring.
  - `check_interval` — [`L63`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L63)
  - `history` — [`L66`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L66)
  - `quiet` — [`L64`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L64)
  - `running` — [`L65`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L65)
- protocol/private: `_monitor_thread`[`L67`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L67)
- uses (calls/refs, reference-scoped): [`_pandas_available`](analyze_memory.md#_pandas_available)
- used by: [`_initialize_memory_tracking`](../trainers/base_trainer.md#BaseTrainer._initialize_memory_tracking), [`mem_ops`](parameters_transformation.md#mem_ops)

### `ThreadedTCPServer`  ·  implements/extends TCPServer, ThreadingMixIn
- def: [`easydel/utils/analyze_memory.py:307`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L307)
- signature: `class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):`
- members:
  - `allow_reuse_address` — [`L309`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L309)
  - `stats_queue` — [`L308`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L308)
- uses (calls/refs, reference-scoped): [`host`](analyze_memory.md#MemoryMonitorServer.host), [`port`](analyze_memory.md#MemoryMonitorServer.port), [`stats_queue`](analyze_memory.md#MemoryMonitorServer.stats_queue), [`RequestHandler`](analyze_memory.md#MemoryMonitorServer._run_server.RequestHandler)

## Functions
- `start_client(server_host)` — [`L453`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L453) — Create and start a ``MemoryMonitorClient`` that reports to the given host.
- `start_server()` — [`L442`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L442) — Create and start a ``MemoryMonitorServer`` on the default address.

## Module values
- `_pandas_available` — [`L35`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L35)
- `logger` — [`L30`](../../../../../../raw/code/EasyDeL/easydel/utils/analyze_memory.py#L30)

