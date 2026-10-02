---
title: 'Module: internal/xplane_tools.py'
type: catalog
provenance: extracted
module: internal/xplane_tools.py
status: fresh
symbol_base: scip-python python xprof-mcp 0.0.0 `internal.xplane_tools`/
symbols:
  _require_xplane_proto: _require_xplane_proto().
  list_xplane_events: list_xplane_events().
  aggregate_xplane_events: aggregate_xplane_events().
  _HAS_XPLANE_PROTO: _HAS_XPLANE_PROTO.
  get_xspace_proto: get_xspace_proto().
  _fetch_xspace: _fetch_xspace().
  _XPLANE_IMPORT_ERROR: _XPLANE_IMPORT_ERROR.
---
# Module: [`internal/xplane_tools.py`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py)

## Functions
- `_fetch_xspace(run: str, host: str = "")` — [`L62`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L62) — Reads and parses an XSpace proto from disk.
- `_require_xplane_proto()` — [`L57`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L57)
- `aggregate_xplane_events(run: str, host: str = "", plane_regex: str = ".*", event_regex: str = ".*", top_n: int = 50)` — [`L246`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L246) — Calculates statistical aggregates for matching timeline events.
- `get_xspace_proto(run: str, host: str = "", as_text: bool = False, output_path: Optional[str] = None)` — [`L87`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L87) — Returns the XSpace proto for a host in the run.
- `list_xplane_events(run: str, host: str = "", plane_regex: str = ".*", event_regex: str = ".*", start_time_ps: Optional[int] = None, end_time_ps: Optional[int] = None, max_events: int = 100, offset: int = 0)` — [`L143`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L143) — Searches and filters timeline events across XPlanes.

## Module values
- `_HAS_XPLANE_PROTO` — [`L35`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L35)
- `_XPLANE_IMPORT_ERROR` — [`L49`](../../../../../raw/code/xprof-mcp/internal/xplane_tools.py#L49)

