---
title: 'Module: internal/xprof_client.py'
type: catalog
provenance: extracted
module: internal/xprof_client.py
status: fresh
symbol_base: scip-python python xprof-mcp 0.0.0 `internal.xprof_client`/
symbols:
  OSSXprofClient._logdir: OSSXprofClient#_logdir.
  OSSXprofClient.read_xplane_bytes: OSSXprofClient#read_xplane_bytes().
  OSSXprofClient._base_url: OSSXprofClient#_base_url.
  OSSXprofClient._session: OSSXprofClient#_session.
  _INSTANCE._INSTANCE: _INSTANCE._INSTANCE.
  OSSXprofClient.read_hlo_proto_bytes: OSSXprofClient#read_hlo_proto_bytes().
  OSSXprofClient.get_session_dir: OSSXprofClient#get_session_dir().
  OSSXprofClient.fetch: OSSXprofClient#fetch().
  OSSXprofClient.get_runs: OSSXprofClient#get_runs().
  OSSXprofClient.get_run_tools: OSSXprofClient#get_run_tools().
  OSSXprofClient.get_hosts: OSSXprofClient#get_hosts().
  OSSXprofClient.get_hlo_module_list: OSSXprofClient#get_hlo_module_list().
  OSSXprofClient.list_xplane_hosts: OSSXprofClient#list_xplane_hosts().
  get_client: get_client().
  set_client: set_client().
  _gfile_exists: _gfile_exists().
  _gfile_listdir: _gfile_listdir().
  _gfile_read: _gfile_read().
  OSSXprofClient: OSSXprofClient#
  OSSXprofClient._require_logdir: OSSXprofClient#_require_logdir().
  OSSXprofClient.get_xplane_file_path: OSSXprofClient#get_xplane_file_path().
  OSSXprofClient.get_hlo_proto_file_path: OSSXprofClient#get_hlo_proto_file_path().
  _is_gcs: _is_gcs().
  OSSXprofClient.base_url: OSSXprofClient#base_url().
  OSSXprofClient.logdir: OSSXprofClient#logdir().
  OSSXprofClient.fetch_json: OSSXprofClient#fetch_json().
  _DEFAULT_XPROF_URL: _DEFAULT_XPROF_URL.
  _detect_logdir_from_procfs: _detect_logdir_from_procfs().
  OSSXprofClient.__init__: OSSXprofClient#__init__().
---
# Module: [`internal/xprof_client.py`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py)

## Classes
### `OSSXprofClient`
- def: [`internal/xprof_client.py:136`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L136)
- members:
  - `base_url(self)` — [`L152`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L152) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `fetch(self, tool: str, run: str, host: str = "", timeout: int = 120, **kwargs)` — [`L211`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L211) — Fetches raw tool data bytes from the xprof server. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `fetch_json(self, tool: str, run: str, host: str = "", **kwargs)` — [`L244`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L244) — Fetches tool data, raising if the response is not JSON-like.
  - `get_hlo_module_list(self, run: str, host: str = "")` — [`L194`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L194) — Returns a comma-separated string of HLO module names. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `get_hlo_proto_file_path(self, run: str, module_name: str)` — [`L271`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L271) — Returns path to <module_name>.hlo_proto.pb for the given run.
  - `get_hosts(self, run: str, tool: str = "overview_page")` — [`L181`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L181) — Returns host metadata for a run and tool. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `get_run_tools(self, run: str)` — [`L171`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L171) — Returns available tools for a given run. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `get_runs(self)` — [`L163`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L163) — Returns a sorted list of available profiling run names. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `get_session_dir(self, run: str)` — [`L262`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L262) — Returns the on-disk session directory for a run. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `get_xplane_file_path(self, run: str, host: str)` — [`L267`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L267) — Returns path to <host>.xplane.pb for the given run. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `list_xplane_hosts(self, run: str)` — [`L297`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L297) — Lists hosts by scanning xplane.pb files in the session directory.
  - `logdir(self)` — [`L156`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L156) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
  - `read_hlo_proto_bytes(self, run: str, module_name: str)` — [`L290`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L290) — Reads and returns raw bytes from <module_name>.hlo_proto.pb.
  - `read_xplane_bytes(self, run: str, host: str)` — [`L275`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L275) — Reads and returns raw bytes from <host>.xplane.pb. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- protocol/private: `__init__`[`L139`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L139), `_base_url`[`L140`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L140), `_logdir`[`L143`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L143), `_require_logdir`[`L252`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L252), `_session`[`L148`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L148)
- uses (calls/refs, reference-scoped): [`_gfile_exists`](xprof_client.md#_gfile_exists), [`_gfile_listdir`](xprof_client.md#_gfile_listdir), [`_gfile_read`](xprof_client.md#_gfile_read), [`_DEFAULT_XPROF_URL`](xprof_client.md#_DEFAULT_XPROF_URL), [`_detect_logdir_from_procfs`](xprof_client.md#_detect_logdir_from_procfs)
- used by: [`_INSTANCE`](xprof_client.md#_INSTANCE._INSTANCE), [`get_client`](xprof_client.md#get_client), [`set_client`](xprof_client.md#set_client)

## Functions
- `_detect_logdir_from_procfs(xprof_url: str)` — [`L61`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L61) — Try to detect the xprof logdir from the server process cmdline via /proc. — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- `_gfile_exists(path: str)` — [`L29`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L29) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- `_gfile_listdir(path: str)` — [`L39`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L39) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- `_gfile_read(path: str)` — [`L49`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L49) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- `_is_gcs(path: str)` — [`L25`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L25) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- `get_client()` — [`L315`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L315)
- `set_client(client: Optional[OSSXprofClient])` — [`L322`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L322) — Override the global client (useful for testing).

## Module values
- `_DEFAULT_XPROF_URL` — [`L57`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L57) — documented in [internal-xprof_client](../../concepts/internal-xprof_client.md)
- `_INSTANCE` — [`L58`](../../../../../raw/code/xprof-mcp/internal/xprof_client.py#L58)

