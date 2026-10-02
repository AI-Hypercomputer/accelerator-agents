---
title: 'Module: axlearn/common/monitoring/device_monitor.py'
type: catalog
provenance: extracted
module: axlearn/common/monitoring/device_monitor.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.monitoring.device_monitor`/DeviceMonitor
symbols:
  DeviceMonitor._check_host_and_log_metrics: '#_check_host_and_log_metrics().'
  DeviceMonitor._start_monitoring: '#_start_monitoring().'
  DeviceMonitor._monitor_loop: '#_monitor_loop().'
  DeviceMonitor.Config: '#Config#'
  DeviceMonitorClient.Config: Client#Config#
  DeviceMonitor.Config.monitor_client: '#Config#monitor_client.'
  DeviceMonitorClient.__init__: Client#__init__().
  DeviceMonitor.__init__: '#__init__().'
  DeviceMonitor._monitor_client: '#_monitor_client.'
  DeviceMonitor._stop_monitor: '#_stop_monitor().'
  DeviceMonitor: '#'
  DeviceMonitor._monitor_thread: '#_monitor_thread.'
  DeviceMonitor.start_monitoring: '#start_monitoring().'
  DeviceMonitorClient: Client#
  DeviceMonitor._monitor_stopping: '#_monitor_stopping.'
  DeviceMonitorClient.collect_metrics: Client#collect_metrics().
  DeviceMonitorClient.is_host_idle: Client#is_host_idle().
  DeviceMonitor.is_host_idle: '#is_host_idle().'
  DeviceMonitor._idle: '#_idle.'
  DeviceMonitorClient.Config.platform: Client#Config#platform.
  DeviceMonitor.Config.log_every_n: '#Config#log_every_n.'
  DeviceMonitor.Config.check_interval_in_sec: '#Config#check_interval_in_sec.'
---
# Module: [`axlearn/common/monitoring/device_monitor.py`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/monitoring/device_monitor.py:47`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L47)
- doc: Configures DeviceMonitor.
- signature: `class Config(Configurable.Config):`
- members:
  - `check_interval_in_sec` — [`L60`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L60)
  - `log_every_n` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L61)
  - `monitor_client` — [`L59`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L59)
  - `platform` — [`L23`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L23)
- uses (calls/refs, reference-scoped): [`default_config`](../config.md#Configurable.default_config), [`config_class`](../config.md#config_class), [`Configurable`](../config.md#Configurable), [`Config`](../config.md#Configurable.Config), [`Config`](device_monitor.md#DeviceMonitorClient.Config), [`DeviceMonitorClient`](device_monitor.md#DeviceMonitorClient)
- used by: [`_check_host_and_log_metrics`](device_monitor.md#DeviceMonitor._check_host_and_log_metrics), [`__init__`](device_monitor.md#DeviceMonitor.__init__), [`_monitor_client`](device_monitor.md#DeviceMonitor._monitor_client), [`device_monitor`](../trainer.md#SpmdTrainer.Config.device_monitor)

### `DeviceMonitor`  ·  implements/extends Configurable
- def: [`axlearn/common/monitoring/device_monitor.py:41`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L41)
- doc: Device Monitor to collect and report system metrics.
- signature: `class DeviceMonitor(Configurable):`
- members:
  - `_check_host_and_log_metrics(self)` — [`L84`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L84) — Check if the devices on the host are idle.
  - `_start_monitoring(self)` — [`L93`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L93) — Start the monitor.
  - `_stop_monitor(self)` — [`L104`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L104) — Stops the monitor.
  - `is_host_idle(self)` — [`L80`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L80) — Check if the TPU device on the host are idle.
  - `start_monitoring(self)` — [`L72`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L72) — Start the monitor.
- protocol/private: `__init__`[`L63`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L63), `_idle`[`L67`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L67), `_monitor_client`[`L66`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L66), `_monitor_loop`[`L113`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L113), `_monitor_stopping`[`L69`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L69), `_monitor_thread`[`L68`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L68)
- uses (calls/refs, reference-scoped): [`config`](../config.md#Configurable.config), [`maybe_instantiate`](../config.md#maybe_instantiate), [`Configurable`](../config.md#Configurable), [`__init__`](../config.md#Configurable.__init__), [`Config`](device_monitor.md#DeviceMonitor.Config), [`monitor_client`](device_monitor.md#DeviceMonitor.Config.monitor_client), [`DeviceUsage`](../utils.md#DeviceUsage), [`log_every_n`](device_monitor.md#DeviceMonitor.Config.log_every_n), [`platform`](device_monitor.md#DeviceMonitorClient.Config.platform)
- used by: [`Configurable`](../config.md#Configurable), [`device_monitor`](../trainer.md#SpmdTrainer.Config.device_monitor)

### `DeviceMonitorClient`  ·  implements/extends Configurable
- def: [`axlearn/common/monitoring/device_monitor.py:15`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L15)
- doc: Base Client for fetching metrics from devices.
- signature: `class DeviceMonitorClient(Configurable):`
- members:
  - `__init__(self, cfg: Config)` — [`L25`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L25) — Initialize the DeviceMonitorClient.
  - `collect_metrics(self)` — [`L30`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L30) — Collect metrics from the device, it should be empty.
  - `is_host_idle(self, usages: list[Usage])` — [`L34`](../../../../../../../raw/code/axlearn/axlearn/common/monitoring/device_monitor.py#L34) — Check if the devices on the host are idle, always return False.
- uses (calls/refs, reference-scoped): [`config`](../config.md#Configurable.config), [`Configurable`](../config.md#Configurable), [`__init__`](../config.md#Configurable.__init__), [`Config`](device_monitor.md#DeviceMonitorClient.Config), [`DeviceUsage`](../utils.md#DeviceUsage)
- used by: [`Configurable`](../config.md#Configurable), [`monitor_client`](device_monitor.md#DeviceMonitor.Config.monitor_client)

