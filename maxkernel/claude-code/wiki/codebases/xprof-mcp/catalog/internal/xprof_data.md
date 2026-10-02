---
title: 'Module: internal/xprof_data.py'
type: catalog
provenance: extracted
module: internal/xprof_data.py
status: fresh
symbol_base: scip-python python xprof-mcp 0.0.0 `internal.xprof_data`/
symbols:
  get_profile_summary: get_profile_summary().
  _parse_op_profile_tree: _parse_op_profile_tree().
  get_hlo_op_profile: get_hlo_op_profile().
  get_op_profile: get_op_profile().
  _parse_datatable: _parse_datatable().
  _extract_p_dict: _extract_p_dict().
  _aggregate_ops_from_profile: _aggregate_ops_from_profile().
  get_hosts: get_hosts().
  get_device_information: get_device_information().
---
# Module: [`internal/xprof_data.py`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py)

## Functions
- `_aggregate_ops_from_profile(profile: dict)` — [`L279`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L279) — Flattens op_profile tree, aggregating ops by name across all programs.
- `_extract_p_dict(data: bytes | str)` — [`L45`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L45) — Extracts the merged 'p' (properties) dict from all DataTable sections.
- `_parse_datatable(data: bytes | str)` — [`L18`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L18) — Parses a Google DataTable JSON response into a list of row dicts.
- `_parse_op_profile_tree(node: dict, total_time_ps: float, top_n: int)` — [`L162`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L162) — Recursively converts an op_profile node into a summary dict.
- `get_device_information(run: str)` — [`L334`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L334) — Returns hardware device information from the Roofline Model analysis.
- `get_hlo_op_profile(run: str, top_n: int = 15)` — [`L121`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L121) — Summarizes the most expensive HLO operations in the run.
- `get_hosts(run: str)` — [`L316`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L316) — Returns the list of hosts profiled in the run.
- `get_op_profile(run: str, host: str = "", top_n: int = 10)` — [`L196`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L196) — Fetches a breakdown of device time by program and operation type.
- `get_profile_summary(run: str)` — [`L62`](../../../../../raw/code/xprof-mcp/internal/xprof_data.py#L62) — Provides a high-level performance summary of a profiling run.

