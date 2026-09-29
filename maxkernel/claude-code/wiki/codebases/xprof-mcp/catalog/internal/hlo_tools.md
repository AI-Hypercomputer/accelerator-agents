---
title: 'Module: internal/hlo_tools.py'
type: catalog
provenance: extracted
module: internal/hlo_tools.py
status: fresh
symbol_base: scip-python python xprof-mcp 0.0.0 `internal.hlo_tools`/
symbols:
  _fetch_hlo_text: _fetch_hlo_text().
  get_hlo_module_content: get_hlo_module_content().
  get_hlo_neighborhood: get_hlo_neighborhood().
  _NO_HLO_DATA_MSG: _NO_HLO_DATA_MSG.
  list_hlo_modules: list_hlo_modules().
---
# Module: [`internal/hlo_tools.py`](../../../../../raw/code/xprof-mcp/internal/hlo_tools.py)

## Functions
- `_fetch_hlo_text(run: str, module_name: Optional[str], *, print_metadata: bool = False)` — [`L34`](../../../../../raw/code/xprof-mcp/internal/hlo_tools.py#L34) — Fetches the full HLO text for a module via the graph_viewer endpoint.
- `get_hlo_module_content(run: str, module_name: Optional[str] = None, fmt: str = "text", max_lines: int = 2000, *, print_metadata: bool = False)` — [`L92`](../../../../../raw/code/xprof-mcp/internal/hlo_tools.py#L92) — Returns the full HLO module content (instruction graph) as text.
- `get_hlo_neighborhood(run: str, instruction_name: str, radius: int = 2, module_name: Optional[str] = None, *, print_metadata: bool = False)` — [`L148`](../../../../../raw/code/xprof-mcp/internal/hlo_tools.py#L148) — Returns the neighborhood of a specific HLO instruction.
- `list_hlo_modules(run: str)` — [`L65`](../../../../../raw/code/xprof-mcp/internal/hlo_tools.py#L65) — Lists all HLO modules available in the profiling run.

## Module values
- `_NO_HLO_DATA_MSG` — [`L23`](../../../../../raw/code/xprof-mcp/internal/hlo_tools.py#L23)

