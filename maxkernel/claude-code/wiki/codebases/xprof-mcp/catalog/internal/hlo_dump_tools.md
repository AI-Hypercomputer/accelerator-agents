---
title: 'Module: internal/hlo_dump_tools.py'
type: catalog
provenance: extracted
module: internal/hlo_dump_tools.py
status: fresh
symbol_base: scip-python python xprof-mcp 0.0.0 `internal.hlo_dump_tools`/
symbols:
  _parse_filename: _parse_filename().
  diff_hlo_stages: diff_hlo_stages().
  get_hlo_dump: get_hlo_dump().
  _resolve_dump_dir: _resolve_dump_dir().
  _scan_dump_dir: _scan_dump_dir().
  list_hlo_dump_modules: list_hlo_dump_modules().
  get_hlo_dump_neighborhood: get_hlo_dump_neighborhood().
  _match_prefix: _match_prefix().
  diff_hlo_stages._read: diff_hlo_stages()._read().
  _STAGE_RE: _STAGE_RE.
  _PROTO_RE: _PROTO_RE.
  _VIZ_RE: _VIZ_RE.
  _PTXLA_RE: _PTXLA_RE.
  _get_dump_dir: _get_dump_dir().
---
# Module: [`internal/hlo_dump_tools.py`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py)

## Functions
- `_get_dump_dir()` — [`L95`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L95) — Returns XLA_HLO_DUMP_DIR env var, or raises if not set.
- `_match_prefix(modules: dict, pattern: str)` — [`L174`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L174) — Returns prefixes matching the glob pattern (case-insensitive).
- `_parse_filename(fname: str)` — [`L70`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L70) — Returns a dict with keys {prefix, stage, ext, format} or None.
- `_read(stage: str)` — [`L426`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L426)
- `_resolve_dump_dir(dump_dir: str)` — [`L108`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L108)
- `_scan_dump_dir(dump_dir: str)` — [`L118`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L118) — Scans dump_dir and returns {prefix -> {stage -> filename}}.
- `diff_hlo_stages(module_pattern: str, stage_before: str = "before_optimizations", stage_after: str = "after_optimizations", dump_dir: str = "", context_lines: int = 5, max_lines: int = 500)` — [`L371`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L371) — Shows a unified diff of HLO text between two compilation stages.
- `get_hlo_dump(module_pattern: str, stage: str = "after_optimizations", dump_dir: str = "", max_lines: int = 2000)` — [`L268`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L268) — Returns the HLO text for a module at a specific compilation stage.
- `get_hlo_dump_neighborhood(instruction_name: str, module_pattern: str, stage: str = "after_optimizations", dump_dir: str = "", radius: int = 2)` — [`L471`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L471) — Returns the neighborhood of an HLO instruction from a dump file.
- `list_hlo_dump_modules(dump_dir: str = "")` — [`L189`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L189) — Lists HLO modules and their available compilation stages in a dump directory.

## Module values
- `_PROTO_RE` — [`L52`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L52)
- `_PTXLA_RE` — [`L62`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L62)
- `_STAGE_RE` — [`L46`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L46)
- `_VIZ_RE` — [`L56`](../../../../../raw/code/xprof-mcp/internal/hlo_dump_tools.py#L56)

