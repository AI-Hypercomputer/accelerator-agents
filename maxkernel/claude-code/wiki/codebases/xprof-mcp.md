---
title: "xprof-mcp"
type: codebase
tags: [mcp, profiler, agent-tooling, xprof, hlo, xla]
commit: 9970d651402194b487e7025b3f9dc66643cb34bc
created: 2026-04-22
updated: 2026-04-22
---

> **Grounded front door → [xprof-mcp overview](xprof-mcp/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

`xprof-mcp` is an MCP (Model Context Protocol) server that exposes the OSS [xprof](https://github.com/openxla/xprof) profiler to an AI agent. In this wiki, it is the bridge between `raw/profiles/` artifacts and `observation` / `experiment` pages: the same agent that maintains this knowledge base calls these tools to read step time, HLO, memory, and timeline data out of captured profiles.

## Overview

The server is a thin FastMCP wrapper. It registers ~17 tools that either (a) proxy HTTP requests to a locally running `xprof --logdir=... --port=8791` server, or (b) read `.xplane.pb` / HLO dump files directly from disk. The agent never parses raw proto bytes itself — every tool returns JSON or formatted text.

Two filesystem contracts matter:
- **xprof logdir** (`XPROF_LOGDIR`): `<logdir>/plugins/profile/<run>/host*.xplane.pb` — the layout the `xprof` server watches. Auto-detected from `/proc/<pid>/cmdline` if the server is on localhost.
- **XLA dump dir** (`XLA_HLO_DUMP_DIR`, or per-call `dump_dir`): what `XLA_FLAGS=--xla_dump_to=...` writes. Used by the dump-family tools; does not require the xprof server.

Transport modes: `stdio` (assistant spawns the process), `sse`, and `streamable-http` (default port `MCP_PORT=8792`, stateless — restart without restarting the agent). See [raw/code/xprof-mcp/server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) lines 197-215.

For this wiki, the agent is expected to run the server in `streamable-http` mode against a persistent `xprof` process pointed at the shared `raw/profiles/` tree. Each `experiment` page's `run` slug corresponds to a directory under that tree, and observation extraction reduces to calling the tools below with that slug.

## Architecture

```
agent (MCP client)
   |
   | MCP (stdio | sse | streamable-http)
   v
server/xprof_mcp_server.py      <- FastMCP registration, ~17 tools
   |
   +-- internal/xprof_client.py  <- HTTP client + disk access + GCS via tf.io.gfile
   +-- internal/xprof_data.py    <- JSON DataTable parsing, op_profile tree walk
   +-- internal/hlo_tools.py     <- graph_viewer long_txt + regex BFS
   +-- internal/xplane_tools.py  <- .xplane.pb proto parse (needs TF)
   +-- internal/hlo_dump_tools.py<- XLA dump dir scan, JAX + PyTorch/XLA formats
   +-- tools/*.py                <- thin wrappers over internal/* (list_runs, get_overview, get_top_hlo_ops, get_memory_profile)
         |
         v
   HTTP to xprof server (http://localhost:8791/data/plugin/profile/...)
   OR direct read of <logdir>/plugins/profile/<run>/*.xplane.pb
   OR direct read of <XLA_HLO_DUMP_DIR>/module_*.hlo[.pb|.txt]
```

Three data paths, picked per tool:
1. **HTTP JSON** — endpoints like `overview_page`, `hlo_stats`, `op_profile`, `memory_profile`, `graph_viewer?type=long_txt`, `roofline_model`. Returned as Google DataTable JSON and flattened by `_parse_datatable` ([internal/xprof_data.py](../../raw/code/xprof-mcp/internal/xprof_data.py) lines 18-42). No protobuf or TF needed.
2. **Direct `.xplane.pb` read** — requires `tensorflow-cpu` for `xplane_pb2`, requires `XPROF_LOGDIR` (or auto-detected logdir) to locate files. Three import fallbacks (`tensorflow.python.profiler.trace`, `tensorflow.tsl.profiler.protobuf`, `tensorboard_plugin_profile.protobuf`) at [internal/xplane_tools.py](../../raw/code/xprof-mcp/internal/xplane_tools.py) lines 32-47 — guards against TF version drift.
3. **XLA dump text scan** — filename regex parse, no server, no TF. Supports JAX (`.hlo`) and PyTorch/XLA (`.txt` with 4-digit sequence number) naming ([internal/hlo_dump_tools.py](../../raw/code/xprof-mcp/internal/hlo_dump_tools.py) lines 46-67). Complement to the HTTP path: gives access to every compilation stage including pre-optimization HLO, which the xprof server cannot show.

GCS-aware: if `XPROF_LOGDIR` is a `gs://` path, `.xplane.pb` reads go through `tf.io.gfile` ([internal/xprof_client.py](../../raw/code/xprof-mcp/internal/xprof_client.py) lines 25-55). Local-disk path is a plain `open()`.

## Key abstractions

- **`OSSXprofClient`** ([internal/xprof_client.py](../../raw/code/xprof-mcp/internal/xprof_client.py) lines 136-308) — module-level singleton via `get_client()`. Holds `XPROF_URL`, `XPROF_LOGDIR`, a `requests.Session`, and the `fetch(tool, run, host, **kwargs)` method that hits `/data/plugin/profile/data?tag=<tool>&run=<run>`.
- **Run / host / module naming** — a `run` is the directory name under `plugins/profile/`; a `host` is the `<host>.xplane.pb` basename; a `module` is the compiled HLO program name (listed via `/data/plugin/profile/module_list`).
- **DataTable flatten** — xprof returns `[{cols, rows, p}]` sections; `_parse_datatable` turns them into row dicts, `_extract_p_dict` merges the scalar `p` properties. Used by every non-proto tool.
- **HLO regex graph** — `get_hlo_neighborhood` and `get_hlo_dump_neighborhood` build an adjacency map from HLO text using `^\s*%name\s*=` and `%operand` regexes, then BFS. No real HLO parser ([internal/hlo_tools.py](../../raw/code/xprof-mcp/internal/hlo_tools.py) lines 190-204).
- **Dump stage aliases** — for PyTorch/XLA dumps with many numbered passes, `before_optimizations` == first sequence, `after_optimizations` == last, intermediate == `pass_NNNN` ([internal/hlo_dump_tools.py](../../raw/code/xprof-mcp/internal/hlo_dump_tools.py) lines 16-24). This lets the agent write tool calls that are format-agnostic across JAX and PyTorch/XLA.
- **Tool return shape** — every tool returns a string (JSON text or formatted text), never a Python object or binary. Errors are caught and returned as error-shaped JSON rather than raised, so a single bad call does not break the agent's conversation state.
- **No caching layer** — every call re-hits the xprof HTTP server or re-reads the file from disk. The server is `stateless_http=True` ([server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) line 68), so concurrent agent calls are safe but not deduplicated.

## Entry points

- `python -m xprof_mcp` → [__main__.py](../../raw/code/xprof-mcp/__main__.py) → `server.xprof_mcp_server:main`.
- `xprof-mcp` console script, declared in [pyproject.toml](../../raw/code/xprof-mcp/pyproject.toml) line 22.
- `main()` at [server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) lines 197-215 parses `--transport {stdio,sse,http}` and `--port`, maps `http` → `streamable-http`.
- All tool registration: [server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) lines 65-107.
- `@mcp.prompt() discovery_flow` at [server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) lines 113-194 — the built-in agent-facing workflow hint, which includes TPU gotchas (roofline thresholds, MXU tile alignment, KV cache, TP ≤ 8).

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `mcp>=1.0.0` | yes | FastMCP server, transports |
| `requests>=2.28.0` | yes | HTTP client to xprof server |
| `protobuf>=3.20.0` | yes | declared but only touched via TF |
| `xprof` (pip) | runtime | the external HTTP server this proxies |
| `tensorflow-cpu>=2.13.0` | optional (`xplane` extra) | parses `.xplane.pb`, enables timeline tools, and `tf.io.gfile` for GCS logdirs |
| Python | `>=3.11` | from [pyproject.toml](../../raw/code/xprof-mcp/pyproject.toml) |

## Notable files

- [README.md](../../raw/code/xprof-mcp/README.md) — install modes, the authoritative tool table (lines 179-201), the recommended analysis workflow (lines 310-319), the XLA dump workflow (lines 231-306).
- [docs/TPU_OPTIMIZATION.md](../../raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md) — **crown-jewel TPU-perf practical guide** (roofline, dimension alignment, dtype, fusion, remat, KV cache, decision trees). Ingested separately as a `source` page and spawns multiple `concept` pages; not summarized here.
- [server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) — the single registration point for all tools and the discovery prompt.
- [internal/xprof_client.py](../../raw/code/xprof-mcp/internal/xprof_client.py) — HTTP + disk + GCS abstraction; the `/proc`-based logdir auto-detection lives at lines 61-133.
- [internal/hlo_dump_tools.py](../../raw/code/xprof-mcp/internal/hlo_dump_tools.py) — dual-format (JAX vs PyTorch/XLA) dump parser, stage diff.

## Performance-relevant surfaces

The tools below are the interface an optimization agent uses to turn a captured profile into observations. File paths are relative to [raw/code/xprof-mcp/](../../raw/code/xprof-mcp/).

| MCP tool | Defined in | Line | Data path | 1-line description |
|---|---|---|---|---|
| `list_runs` | `tools/list_runs_tool.py` | 9 | HTTP `runs` | Lists profiling run names on the xprof server. Start here. |
| `get_hosts` | `internal/xprof_data.py` | 316 | HTTP `hosts` | Returns hosts (JSON) for a run. |
| `get_overview` | `tools/get_overview_tool.py` | 50 | HTTP `overview_page` | Step time, MXU util, HBM BW util, idle %, device type — the first thing to read on a run. |
| `get_memory_profile` | `tools/get_memory_profile_tool.py` | 10 | HTTP `memory_profile` | Peak HBM, heap/stack breakdown. |
| `get_top_hlo_ops` | `tools/get_top_hlo_ops_tool.py` | 18 | HTTP `hlo_stats` (+ `op_profile` fallback) | Top ops by time, FLOPs, bytes accessed (three ranked lists). |
| `get_op_profile` | `internal/xprof_data.py` | 196 | HTTP `op_profile` | Hierarchical per-program breakdown with idle time. Used when `hlo_stats` is empty (common for inference). |
| `get_profile_summary` | `internal/xprof_data.py` | 62 | HTTP `hlo_stats` (+ `overview_page` fallback) | Text summary of top ops by self-time. |
| `get_device_information` | `internal/xprof_data.py` | 334 | HTTP `roofline_model` | Accelerator specs: peak FLOP, peak HBM BW, ridge points. Needed to reason about compute vs memory bound. |
| `list_hlo_modules` | `internal/hlo_tools.py` | 65 | HTTP `module_list` | Lists compiled HLO program names in the run. |
| `get_hlo_module_content` | `internal/hlo_tools.py` | 92 | HTTP `graph_viewer?type=long_txt` | Full HLO text for a module (default max 2000 lines). |
| `get_hlo_neighborhood` | `internal/hlo_tools.py` | 148 | HTTP `graph_viewer` + regex BFS | Producers + consumers of a named instruction (radius 2). Root-cause a slow fusion. |
| `list_xplane_events` | `internal/xplane_tools.py` | 143 | disk `.xplane.pb` (TF) | Regex-filtered timeline events (plane + event name). Use to find individual kernel instances or sync waits. |
| `aggregate_xplane_events` | `internal/xplane_tools.py` | 246 | disk `.xplane.pb` (TF) | Count / total / avg / min / max / stddev per event type. Use to judge systemic slowness and variance. |
| `get_xspace_proto` | `internal/xplane_tools.py` | 87 | disk `.xplane.pb` (TF) | Raw XSpace bytes or text — escape hatch when the filtered tools miss something. |
| `list_hlo_dump_modules` | `internal/hlo_dump_tools.py` | 189 | XLA dump dir | Discovers modules and their available compilation stages in `XLA_HLO_DUMP_DIR`. No xprof server needed. |
| `get_hlo_dump` | `internal/hlo_dump_tools.py` | 268 | XLA dump dir | HLO text at a specific stage (`before_optimizations`, `after_optimizations`, `after_pass_<X>`, `pass_NNNN`). |
| `diff_hlo_stages` | `internal/hlo_dump_tools.py` | 371 | XLA dump dir | Unified diff between two stages — answers "what did this pass change". |
| `get_hlo_dump_neighborhood` | `internal/hlo_dump_tools.py` | 471 | XLA dump dir | Same BFS as `get_hlo_neighborhood`, but over a dumped stage file. |

Additional surfaces the agent will touch when interpreting results:
- **Overview keys** that matter for TPU triage are enumerated at [tools/get_overview_tool.py](../../raw/code/xprof-mcp/tools/get_overview_tool.py) lines 11-34: `steptime_ms_average`, `mxu_utilization_percent`, `flop_rate_utilization_relative_to_roofline`, `device_duty_cycle_percent`, `memory_bw_utilization_relative_to_hw_limit`, `hbm_bw_utilization_percent`, `device_idle_time_percent`, `host_idle_time_percent`, plus the `sc_*` keys for SparseCore runs.
- **Op-profile tree metrics** parsed at [internal/xprof_data.py](../../raw/code/xprof-mcp/internal/xprof_data.py) lines 162-193: `rawTime`, `rawFlops`, `rawBytesAccessedArray[0]`, `bandwidthUtils[0]` (HBM BW util) — these are what feeds time/FLOP/byte tables.
- **Timeline regex pairs** the discovery prompt calls out: `plane_regex='/device:TPU:0'` for TPU device timelines; `plane_regex='host.*', event_regex='.*Wait.*'` for host syncs.
- **XLA dump scan limits** — `_scan_dump_dir` at [internal/hlo_dump_tools.py](../../raw/code/xprof-mcp/internal/hlo_dump_tools.py) line 118 enumerates every file once per call; large dump dirs (per-pass dumps with `--xla_dump_hlo_pass_re=.*`) can produce thousands of files. Per-call `dump_dir` override is supported on all four dump tools.

## Connections

- Any `experiment` page's Profile section points at a `raw/profiles/<YYYY-MM-DD>-<slug>/` directory that these tools read.
- `observation` pages typically cite "How to see it" as a specific `get_top_hlo_ops` / `aggregate_xplane_events` / `diff_hlo_stages` invocation plus a run name — this codebase page is the reference for what those tool names mean.
- The discovery prompt at [server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py) lines 150-190 encodes a short list of TPU gotchas (roofline crossover, MXU tile 128/256, wrong-dtype inference, materialized broadcasts, TP > 8 across DCN, KV-cache decode patterns). These will be fleshed out as individual `concept` pages; for now, the prompt itself is an agent-visible signal.

## See also

- [../index.md](../index.md)

## Sources

- [raw/code/xprof-mcp/README.md](../../raw/code/xprof-mcp/README.md)
- [raw/code/xprof-mcp/pyproject.toml](../../raw/code/xprof-mcp/pyproject.toml)
- [raw/code/xprof-mcp/server/xprof_mcp_server.py](../../raw/code/xprof-mcp/server/xprof_mcp_server.py)
- [raw/code/xprof-mcp/internal/xprof_client.py](../../raw/code/xprof-mcp/internal/xprof_client.py)
- [raw/code/xprof-mcp/internal/xprof_data.py](../../raw/code/xprof-mcp/internal/xprof_data.py)
- [raw/code/xprof-mcp/internal/hlo_tools.py](../../raw/code/xprof-mcp/internal/hlo_tools.py)
- [raw/code/xprof-mcp/internal/xplane_tools.py](../../raw/code/xprof-mcp/internal/xplane_tools.py)
- [raw/code/xprof-mcp/internal/hlo_dump_tools.py](../../raw/code/xprof-mcp/internal/hlo_dump_tools.py)
- [raw/code/xprof-mcp/tools/list_runs_tool.py](../../raw/code/xprof-mcp/tools/list_runs_tool.py)
- [raw/code/xprof-mcp/tools/get_overview_tool.py](../../raw/code/xprof-mcp/tools/get_overview_tool.py)
- [raw/code/xprof-mcp/tools/get_top_hlo_ops_tool.py](../../raw/code/xprof-mcp/tools/get_top_hlo_ops_tool.py)
- [raw/code/xprof-mcp/tools/get_memory_profile_tool.py](../../raw/code/xprof-mcp/tools/get_memory_profile_tool.py)
- [raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md](../../raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md) (will be ingested as its own source page)
