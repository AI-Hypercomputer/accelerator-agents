---
title: "Profile-Analyzer Index — xprof + HLO + Pallas + TPU device reference"
type: index
tags: [profile-analyzer, xprof, hlo, stablehlo, pallas, tpu-device, mcp]
created: 2026-05-30
updated: 2026-05-30
---

This page is the **reference catalog** that the [`profile-analyzer`](../.claude/agents/profile-analyzer.md) agent reads in full at Step 0 of every analysis. It contains ONLY documentation and tool reference for **xprof**, **HLO / StableHLO**, **Pallas / custom_call**, the **TPU device model**, and the **mcp__xprof__** server. It carries no hypothesis-generation content and no per-experiment narrative — those live elsewhere (see [Cardinal rule](#cardinal-rule)).

## Cardinal rule

**ONLY information that helps an agent INTERPRET a captured xprof trace or HLO dump.** Concretely, content earns a place on this page only if it helps the agent answer:

- (a) **WHICH tool** to call to get a specific piece of data out of a trace or HLO dump.
- (b) **WHAT a metric / bucket / counter MEANS** when it comes back from the tool.
- (c) **HOW the device or compiler structurally produces what we observe** (HLO op semantics, fusion patterns, collective forms, sharding annotations, custom_call shapes).
- (d) **WHICH GENERATION-SPECIFIC CONSTANTS** to apply when interpreting numbers (MXU shape, VMEM cap, HBM BW, ICI BW) so the agent doesn't have to look them up mid-analysis.
- (e) **HOW to format an observation page** so findings extracted from a profile follow the wiki schema.

Content that **does NOT belong**:
- Hypothesis logic ("if you see X, try Y") — that's the agent's Phase 3 job, driven by the **experiment page's hypothesis context**, not by this index.
- Per-experiment narratives, per-model history, refuted-pattern lists with v-IDs — those live in `wiki/experiments/`, `wiki/models/`, per-model `refuted-patterns.md`.
- Optimization recommendations of any kind — `wiki/model-optimization-index.md` owns that surface.
- Tutorials. Link out; don't duplicate.

If a candidate addition doesn't fit (a)–(e), it goes in a concept page / source page / model-optimization-index — not here.

---

## TLDR — analysis flow (generic, hypothesis-agnostic)

The profile-analyzer agent typically walks a fixed path through the data before reaching Phase 3 (hypothesis-firing audit). This TLDR gives the canonical order; per-tool detail is in [§ mcp__xprof__ tool catalog](#mcp__xprof__-tool-catalog).

| Step | Tool | Question answered |
|------|------|-------------------|
| 1 | [`list_runs`](#tool-list_runs) | Which runs are loaded on the xprof server? Pick the experiment's slug. |
| 2 | [`get_overview`](#tool-get_overview) | Step time, MXU util %, HBM BW util %, device idle %, host idle %. The triage screen. |
| 3 | [`get_device_information`](#tool-get_device_information) | TPU generation, peak FLOPS, peak HBM BW, ridge points. Needed to reason roofline. |
| 4 | [`get_top_hlo_ops`](#tool-get_top_hlo_ops) | Top ops by time / FLOPs / bytes. The compute attribution leaderboard. |
| 5 | [`list_hlo_modules`](#tool-list_hlo_modules) | How many compiled programs? Eager → many; compiled → few; scan → very few. See [HLO module count fingerprint](#hlo-module-count-fingerprint). |
| 6 | [`get_memory_profile`](#tool-get_memory_profile) | HBM peak, heap/stack breakdown. Triage OOM and transient-memory issues. |
| 7 | [`list_xplane_events`](#tool-list_xplane_events) / [`aggregate_xplane_events`](#tool-aggregate_xplane_events) | Timeline events on a plane (e.g. `/device:TPU:0`) — find specific kernels, sync waits, collective spans. |
| 8 | [`list_hlo_dump_modules`](#tool-list_hlo_dump_modules) + [`get_hlo_dump`](#tool-get_hlo_dump) | Read the actual HLO text at a compile stage. Use `after_optimizations` for steady-state; `before_optimizations` to see the lowering source. |
| 9 | [`get_hlo_neighborhood`](#tool-get_hlo_neighborhood) / [`get_hlo_dump_neighborhood`](#tool-get_hlo_dump_neighborhood) | Producers + consumers of a named op (radius 2). Root-cause a slow fusion. |
| 10 | [`diff_hlo_stages`](#tool-diff_hlo_stages) | What did a specific pass change? Answers questions like "did fusion run?", "what survived layout?". |

For drilldowns (Mode B / `focus` parameter), the agent typically iterates between [`aggregate_xplane_events`](#tool-aggregate_xplane_events) (to quantify what it sees in the timeline) and [`get_hlo_neighborhood`](#tool-get_hlo_neighborhood) / [`diff_hlo_stages`](#tool-diff_hlo_stages) (to explain it structurally).

---

## mcp__xprof__ tool catalog

The MCP server is documented in full at [`wiki/codebases/xprof-mcp.md`](codebases/xprof-mcp.md). What follows is the **per-tool quick reference** for the agent — name, where it's defined (so the agent can grep for arg signatures), data path, and a 1-line description.

Three data paths exist; understanding which path a tool uses tells you what infrastructure has to be running:

1. **HTTP JSON** — calls a local `xprof --logdir=... --port=8791` server. Needs the server up; works against GCS logdirs via the server's own `tf.io.gfile` access.
2. **`.xplane.pb` direct read** — reads the trace proto from disk (`<logdir>/plugins/profile/<run>/host*.xplane.pb`). Needs `tensorflow-cpu`; needs `XPROF_LOGDIR` env set or auto-detected.
3. **XLA dump dir scan** — reads files written by `XLA_FLAGS=--xla_dump_to=<dir>`. No xprof server needed; format-aware for both JAX (`.hlo`) and PyTorch/XLA (`.txt` with 4-digit sequence numbers).

| MCP tool | Defined in | Data path | 1-line use |
|---|---|---|---|
| <a id="tool-list_runs"></a>`list_runs` | [tools/list_runs_tool.py:9](../raw/code/xprof-mcp/tools/list_runs_tool.py) | HTTP `runs` | Lists profiling run names on the xprof server. **Start here.** |
| <a id="tool-get_hosts"></a>`get_hosts` | [internal/xprof_data.py:316](../raw/code/xprof-mcp/internal/xprof_data.py) | HTTP `hosts` | Returns hosts (JSON) for a run. Needed to scope per-host queries on multi-host runs. |
| <a id="tool-get_overview"></a>`get_overview` | [tools/get_overview_tool.py:50](../raw/code/xprof-mcp/tools/get_overview_tool.py) | HTTP `overview_page` | **Triage screen**: step time, MXU util %, HBM BW util %, idle %, device type. First call after `list_runs`. Keys at [tools/get_overview_tool.py:11-34](../raw/code/xprof-mcp/tools/get_overview_tool.py). |
| <a id="tool-get_memory_profile"></a>`get_memory_profile` | [tools/get_memory_profile_tool.py:10](../raw/code/xprof-mcp/tools/get_memory_profile_tool.py) | HTTP `memory_profile` | Peak HBM, heap/stack breakdown. Triage OOM and transient memory. |
| <a id="tool-get_top_hlo_ops"></a>`get_top_hlo_ops` | [tools/get_top_hlo_ops_tool.py:18](../raw/code/xprof-mcp/tools/get_top_hlo_ops_tool.py) | HTTP `hlo_stats` (+ `op_profile` fallback) | **Top ops by time, FLOPs, bytes** — three ranked lists. The compute attribution leaderboard. |
| <a id="tool-get_op_profile"></a>`get_op_profile` | [internal/xprof_data.py:196](../raw/code/xprof-mcp/internal/xprof_data.py) | HTTP `op_profile` | Hierarchical per-program breakdown with idle time. Use when `hlo_stats` is empty (common for inference). |
| <a id="tool-get_profile_summary"></a>`get_profile_summary` | [internal/xprof_data.py:62](../raw/code/xprof-mcp/internal/xprof_data.py) | HTTP `hlo_stats` (+ `overview_page` fallback) | Text summary of top ops by self-time. Quick narrative summary. |
| <a id="tool-get_device_information"></a>`get_device_information` | [internal/xprof_data.py:334](../raw/code/xprof-mcp/internal/xprof_data.py) | HTTP `roofline_model` | Accelerator specs: peak FLOP, peak HBM BW, ridge points. **Needed for any roofline reasoning.** |
| <a id="tool-list_hlo_modules"></a>`list_hlo_modules` | [internal/hlo_tools.py:65](../raw/code/xprof-mcp/internal/hlo_tools.py) | HTTP `module_list` | Lists compiled HLO program names in the run. Module count is a key fingerprint — see [HLO module count fingerprint](#hlo-module-count-fingerprint). |
| <a id="tool-get_hlo_module_content"></a>`get_hlo_module_content` | [internal/hlo_tools.py:92](../raw/code/xprof-mcp/internal/hlo_tools.py) | HTTP `graph_viewer?type=long_txt` | Full HLO text for a module (default max 2000 lines). |
| <a id="tool-get_hlo_neighborhood"></a>`get_hlo_neighborhood` | [internal/hlo_tools.py:148](../raw/code/xprof-mcp/internal/hlo_tools.py) | HTTP `graph_viewer` + regex BFS | **Producers + consumers of a named instruction** (radius 2). Root-cause a slow fusion. |
| <a id="tool-list_xplane_events"></a>`list_xplane_events` | [internal/xplane_tools.py:143](../raw/code/xprof-mcp/internal/xplane_tools.py) | disk `.xplane.pb` (TF) | Regex-filtered timeline events (plane + event name). Use to find specific kernel instances or sync waits. |
| <a id="tool-aggregate_xplane_events"></a>`aggregate_xplane_events` | [internal/xplane_tools.py:246](../raw/code/xprof-mcp/internal/xplane_tools.py) | disk `.xplane.pb` (TF) | Count / total / avg / min / max / stddev per event type. Use to **quantify** what `list_xplane_events` shows you (systemic slowness, variance). |
| <a id="tool-get_xspace_proto"></a>`get_xspace_proto` | [internal/xplane_tools.py:87](../raw/code/xprof-mcp/internal/xplane_tools.py) | disk `.xplane.pb` (TF) | Raw XSpace bytes or text — escape hatch when filtered tools miss something. |
| <a id="tool-list_hlo_dump_modules"></a>`list_hlo_dump_modules` | [internal/hlo_dump_tools.py:189](../raw/code/xprof-mcp/internal/hlo_dump_tools.py) | XLA dump dir | Discovers modules and their available compilation stages in `XLA_HLO_DUMP_DIR`. **No xprof server needed.** |
| <a id="tool-get_hlo_dump"></a>`get_hlo_dump` | [internal/hlo_dump_tools.py:268](../raw/code/xprof-mcp/internal/hlo_dump_tools.py) | XLA dump dir | HLO text at a specific stage: `before_optimizations`, `after_optimizations`, `after_pass_<X>`, `pass_NNNN`. |
| <a id="tool-diff_hlo_stages"></a>`diff_hlo_stages` | [internal/hlo_dump_tools.py:371](../raw/code/xprof-mcp/internal/hlo_dump_tools.py) | XLA dump dir | **Unified diff between two stages** — answers "what did this pass change". |
| <a id="tool-get_hlo_dump_neighborhood"></a>`get_hlo_dump_neighborhood` | [internal/hlo_dump_tools.py:471](../raw/code/xprof-mcp/internal/hlo_dump_tools.py) | XLA dump dir | Same BFS as `get_hlo_neighborhood`, but over a dumped stage file. |

**Tool-call ergonomics:**
- Every tool returns a **string** (JSON text or formatted text). Never a Python object or binary.
- Errors return error-shaped JSON, not exceptions. A bad call doesn't break the agent's conversation state.
- No caching — every call hits the server or re-reads the file. Server is `stateless_http=True`; concurrent calls are safe but not deduplicated.
- Filename-format awareness: dump tools handle **JAX `.hlo`** and **PyTorch/XLA `.txt` (4-digit seq)** transparently. Dump-stage aliases (`before_optimizations` / `after_optimizations` / `pass_NNNN`) defined at [internal/hlo_dump_tools.py:16-24](../raw/code/xprof-mcp/internal/hlo_dump_tools.py).
- For multi-host runs, prefer `host0` queries first; only iterate hosts when you suspect skew.

---

## xprof bucket taxonomy

What each metric in `get_overview` (and adjacent endpoints) actually measures. Source: [TensorBoard Profiler overview page docs](sources/2026-xprof-overview-page.md), [xprof terminology](sources/2026-xprof-terminology.md), [perf counters](sources/2026-xprof-perf-counters.md), [Scaling Book ch9](sources/2025-scaling-book-ch9-profiling.md), and the project's [trace event categories concept page](concepts/trace-event-categories.md).

### Time decomposition

A captured **step** has wall time partitioned roughly into:

```
step_time = compute + collectives + idle + overhead
```

Where:
- **Compute** = time the TPU device cores were executing HLO (MXU, VPU, scalar). Subdivides into MXU-busy, VPU-busy, scalar-busy, "TC active but not utilized" (kernels running, not saturating).
- **Collectives** = time spent in `all-gather`, `reduce-scatter`, `all-reduce`, `all-to-all`, `send`/`recv` ops. May overlap compute (good) or block (bad).
- **Idle (device)** = TC stalls. Common causes: HBM-bound op waiting on memory, awaiting collective completion, awaiting host I/O.
- **Idle (host)** = host CPU stall waiting on device — typically means infeed/outfeed exhaustion or device step too short.
- **Overhead** = compile, dispatch, eager-mode kernel launch. On eager-mode lanes, large.

### Headline metrics (from `get_overview`)

The keys the agent should always look at first, with how to interpret them:

| Key | Meaning | Healthy band (training) |
|-----|---------|-------------------------|
| `steptime_ms_average` | Wall-time per step. | Compare to baseline; +/-5% within noise. |
| `mxu_utilization_percent` | Fraction of MXU cycles doing FMA. The headline "are we compute-bound" number. | 30-70% is typical; <20% suggests HBM or collective bound; >85% near optimal. |
| `flop_rate_utilization_relative_to_roofline` | Observed FLOP/s / roofline FLOP/s. | Read alongside `mxu_utilization_percent` — the two answer different questions; see [roofline-model](concepts/roofline-model.md). |
| `device_duty_cycle_percent` | Fraction of step the device was doing something (compute OR collective OR memory). | <80% → device-idle bound; root-cause via timeline. |
| `memory_bw_utilization_relative_to_hw_limit` | HBM bandwidth observed / peak HBM BW. | >70% = HBM-bound. <30% = compute-bound (good). |
| `hbm_bw_utilization_percent` | Raw HBM BW fraction. | Same as above, different normalization. |
| `device_idle_time_percent` | TC stall time. | High → root-cause via timeline ([`list_xplane_events`](#tool-list_xplane_events) with `event_regex='.*Wait.*'`). |
| `host_idle_time_percent` | Host CPU stall. | High → infeed pipeline starved. |
| `sc_*` keys | SparseCore equivalents (only present on SparseCore runs). | Use the same shape of reasoning. |

(Full key list at [xprof-mcp tools/get_overview_tool.py:11-34](../raw/code/xprof-mcp/tools/get_overview_tool.py). The agent should read those lines to see every key the server emits.)

### Op-profile metrics (from `get_top_hlo_ops` / `get_op_profile`)

Per-op fields parsed at [xprof-mcp internal/xprof_data.py:162-193](../raw/code/xprof-mcp/internal/xprof_data.py):
- **`rawTime`** — total time the op consumed (sum across instances).
- **`rawFlops`** — total FLOP count (analytic, not measured).
- **`rawBytesAccessedArray[0]`** — total bytes read/written.
- **`bandwidthUtils[0]`** — HBM BW utilization during this op.

Derived ratios the agent should compute:
- **Arithmetic intensity** = `rawFlops / rawBytesAccessedArray[0]` (FLOP/byte). Compare to the chip's **critical arithmetic intensity** (FLOP-peak / HBM-BW-peak) from [`get_device_information`](#tool-get_device_information) and [tpu-hardware-generations](concepts/tpu-hardware-generations.md). Below the ridge ⇒ memory-bound; above ⇒ compute-bound.
- **Achieved FLOP/s** = `rawFlops / rawTime`. Compare to chip peak.

### Roofline interpretation

Reference: [roofline-model](concepts/roofline-model.md), [ridge-point](concepts/ridge-point.md), [ici-roofline](concepts/ici-roofline.md), [scaling-book ch1](sources/2025-scaling-book-ch1-roofline.md), [xprof roofline-model source](sources/2026-xprof-roofline-model.md).

- The **HBM ridge point** is FLOPs-per-byte where memory bandwidth and compute bandwidth break even. Below ⇒ HBM-bound (e.g. elementwise, RMSNorm, gather/scatter). Above ⇒ compute-bound (matmul-dominated).
- The **ICI ridge point** is the equivalent for cross-chip collectives. Collectives below it should be DCN-aware (i.e. don't expect linear scaling).
- The roofline view in xprof is per-op; for whole-step verdicts, read `flop_rate_utilization_relative_to_roofline` and `memory_bw_utilization_relative_to_hw_limit` from `get_overview`.

### Timeline regexes

The agent will use `list_xplane_events` / `aggregate_xplane_events` with these standard regex patterns (from the xprof-mcp discovery prompt):
- `plane_regex='/device:TPU:0'` — TPU device timeline.
- `plane_regex='/device:TPU.*'` — all TPU devices (multi-chip per host).
- `plane_regex='host.*'`, `event_regex='.*Wait.*'` — host sync stalls.
- `event_regex='.*[Cc]ollective.*'` — collective spans on the device.
- `event_regex='.*Pallas.*'` or `event_regex='.*custom_call.*'` — custom kernel invocations.

---

## TPU device model

Per-generation constants the agent must apply when interpreting numbers. Source: [tpu-hardware-generations](concepts/tpu-hardware-generations.md), the cloud TPU docs ([v4](sources/2026-cloud-tpu-v4-docs.md), [v5e](sources/2026-cloud-tpu-v5e-docs.md), [v5p](sources/2026-cloud-tpu-v5p-docs.md), [v6e](sources/2026-cloud-tpu-v6e-docs.md), [v7x](sources/2026-cloud-tpu-v7x-ironwood-docs.md)), and [scaling-book ch2](sources/2025-scaling-book-ch2-tpus.md).

### Per-generation envelope

(Authoritative numbers live in [model-optimization-index § TPU hardware envelope](model-optimization-index.md). Cited here as the constants the agent should apply when interpreting profile numbers. If they differ, the model-optimization-index is canonical.)

| Generation | bf16 peak (TC) | HBM | HBM BW | VMEM/TC | MXU shape | ICI BW | Critical FLOP/byte (bf16) |
|---|---|---|---|---|---|---|---|
| v4 | 137 TF | 32 GB | 1.2 TB/s | 32 MiB | 128×128 | 50 GB/s/link | ~114 |
| v5e | 197 TF | 16 GB | 0.82 TB/s | 32 MiB | 128×128 | 80 GB/s/link | ~240 |
| v5p | 459 TF | 96 GB | 2.77 TB/s | 96 MiB | 128×128 | 100 GB/s/link | ~165 |
| v6e (Trillium) | 918 TF | 32 GB | 1.64 TB/s | 64 MiB | 256×256 | 100 GB/s/link | ~560 |
| v7x (Ironwood) | 4614 TF (FP8 9.2 PF) | 192 GB | 7.4 TB/s | 128 MiB | 256×256 | 800 GB/s/link | ~624 |
| v8 (preview) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### XPK / chip-count naming caveats

- **v5p**: `--tpu-type=v5p-N` counts **TensorCores**, not chips. Each v5p chip has 2 TCs. So `v5p-8` = 4 chips, `v5p-16` = 8 chips, `v5p-32` = 16 chips. See [feedback memory v5p naming convention](../memory/feedback_v5p_naming_convention.md).
- **v6e / v7x**: `--tpu-type=v6e-N` counts **chips** (1 TC per chip on these gens).

### Practical VMEM caps observed

- v5p-16 / v6e-8 / v6e-16 in libtpu 0.0.41: **65472 KiB** maximum for `xla_tpu_scoped_vmem_limit_kib`. Higher values silently fall back. See [feedback memory v5p16_vmem_cap_universal](../memory/project_v5p16_vmem_cap_universal.md) for the universal-cap observation.
- Reference: [vmem](concepts/vmem.md), [vmem-budget](concepts/vmem-budget.md), [libtpu-flags](concepts/libtpu-flags.md).

### Memory hierarchy

Five tiers, fastest to slowest, with relative bandwidth:

| Tier | Capacity (typ) | BW (relative) | Concept page |
|------|---------------|---------------|--------------|
| Register file | tiny | 1× | (in [scaling-book ch2](sources/2025-scaling-book-ch2-tpus.md)) |
| VMEM / SMEM (per TC scratch) | 32-128 MiB | ~10-50× HBM | [vmem](concepts/vmem.md), [vmem-budget](concepts/vmem-budget.md) |
| HBM | 16-192 GB | 1× | [hbm](concepts/hbm.md) |
| ICI (intra-pod) | unbounded | 0.05-0.4× HBM | [collective-communication](concepts/collective-communication.md) |
| DCN (inter-pod) | unbounded | <<0.01× HBM | (in tpu-hardware-generations) |

The agent uses this hierarchy to reason about where a slow op's bytes are coming from.

### Compute units

- **MXU** (Matrix Multiply Unit) — the systolic array. Shape is `128×128` on v4/v5*, `256×256` on v6e+. Misaligned dims to MXU shape leave units idle. See [mxu](concepts/mxu.md), [mxu-utilization](concepts/mxu-utilization.md).
- **VPU** (Vector Processing Unit) — elementwise, reduction, etc. Used for activations, normalization, softmax non-matmul parts.
- **Scalar core** — control, addressing, low-throughput math.
- **SparseCore** (on v5p+) — embedding lookups, sparse ops. Has its own `sc_*` overview keys.

---

## HLO IR reference

What the HLO text actually means when the agent reads it. Source: [hlo](concepts/hlo.md), [hlo-op](concepts/hlo-op.md), [custom-call](concepts/custom-call.md), [xla-fusion](concepts/xla-fusion.md), and the StableHLO codebase page (which doubles as the HLO spec reference).

### Module structure

An HLO **module** is one compiled program. A run can have 1 module (fully-compiled `jit` step) up to hundreds (eager dispatch). See [HLO module count fingerprint](#hlo-module-count-fingerprint) below.

Each module is one or more **computations**. The `ENTRY` computation is the top-level program. Other computations are typically called from `while`, `conditional`, `scan`, `fusion`, or `call`.

Each computation is a sequence of **instructions** (ops). Each instruction has:
- A name (e.g. `%fusion.42`).
- An opcode (e.g. `dot`, `fusion`, `all-gather`, `custom-call`).
- A type (e.g. `f32[1024,8192]{1,0}`) — element type + shape + **layout** (`{1,0}` = dim 1 is minor, dim 0 is major).
- Operands (referenced by `%name`).
- Optional attributes: `sharding={...}`, `metadata={op_name=..., source_file=..., source_line=...}`, `frontend_attributes={...}`, `backend_config={...}`.

### Op categories the agent cares about

| Op | What it does | Read its attrs |
|---|---|---|
| `dot(a, b)` | Matmul. Carries `dot_dimension_numbers` (contracting + batch dims). | Check contracting dims align with MXU shape. |
| `fusion(...) calls=%fusion_comp` | A fused computation. Body is in `%fusion_comp`. **Fusion kind**: `kInput`, `kOutput`, `kLoop`, `kCustom`. | Read body to see what's inside. See [xla-fusion](concepts/xla-fusion.md). |
| `custom-call(...)` | Calls a backend-specific kernel (Pallas, splash, tokamax, cuDNN-style). Has `custom_call_target="<name>"`. | The `custom_call_target` string is the **kernel name** — see [Pallas custom_call signatures](#pallas-custom_call-signatures). |
| `all-gather(...)` | Gathers shards from all replicas along an axis. Has `replica_groups`, `dimensions`, `use_global_device_ids`. | See [all-gather](concepts/all-gather.md). |
| `reduce-scatter(...)` | All-reduce + shard along an axis. | See [reduce-scatter](concepts/reduce-scatter.md). |
| `all-reduce(...)` | Reduce across replicas. | See [all-reduce](concepts/all-reduce.md). |
| `all-to-all(...)` | Reshuffle along two axes; common in expert-parallel MoE. | (No dedicated page yet — see collective-communication.) |
| `collective-permute(...)` | Point-to-point shuffle (ring patterns). | (Ring-attention reference: [ringattention codebase](codebases/ringattention.md).) |
| `send` / `recv` / `send-done` / `recv-done` | Cross-replica P2P. | See [send-recv-done](concepts/send-recv-done.md). |
| `convert(x)` | Dtype cast. | Watch for unintended fp32 conversions on bf16 paths. |
| `transpose(x)` | Permute dims. | Often a sign of layout mismatch; expensive on TPU. |
| `gather(operand, indices)` / `dynamic-slice` | Indexed read. | Scalar-index `gather` can lower to a degenerate per-element form on some eager backends; prefer a bulk row-gather (`index_select`). |
| `scatter` | Indexed write. | Often shows up in MoE dispatch. |
| `while(cond=..., body=...)` | Loop. **Scan-over-layers lowers to `while`.** | See [scan-over-layers](concepts/scan-over-layers.md). |
| `conditional(...)` | If/else. | |
| `bitcast` / `bitcast-convert` | Reinterpret memory. Free at runtime. | |
| `broadcast(x, dims=...)` | Replicate. May or may not materialize. Check the consumer's fusion. | |

### Sharding annotations

`sharding={...}` comes in several forms:
- `sharding={replicated}` — all replicas hold the same data.
- `sharding={maximal device=0}` — single device.
- `sharding={devices=[A,B,...]<=[mesh]}` — tile along mesh axes. The numbers `[A,B,...]` are tile counts per HLO dim; `[mesh]` is the device-ordering.
- `sharding={manual}` — `shard_map` produced this; sharding is asserted by the user, compiler trusts it.
- `sharding={unknown}` — propagated through but not resolved.

The agent reading sharding asks: "is this op replicated where I expected sharded?" — a common cause of unexpected `all-gather` materialization.

### Fusion patterns

Reference: [xla-fusion](concepts/xla-fusion.md), [xla-cost-model](concepts/xla-cost-model.md).

- **Input fusion** — multiple producers fold into one consumer (e.g. epilogue ops fold into a matmul).
- **Output fusion** — one producer feeds multiple consumers and is replicated rather than materialized.
- **Loop fusion** — multiple elementwise ops fold into one loop.
- **Custom fusion** — backend-specific (Pallas etc.).

**Failure modes the agent reads in HLO**:
- A `convert → broadcast → multiply` that didn't fuse — extra HBM traffic.
- A `transpose` outside a `dot` — layout mismatch; should have been a `dot` with different `dot_dimension_numbers`.
- A `gather` outside a `fusion` — indexed read materialized; often a candidate for kernel work.
- Two adjacent `dot`s with the same LHS that didn't fuse — `dot` chaining missed.

### Collective forms — what to expect when

| Pattern | HLO ops emitted | Reference |
|---------|----------------|-----------|
| FSDP forward param gather | `all-gather` per layer (or batched). | [all-gather](concepts/all-gather.md), [collective-bucketing-prefetch](concepts/collective-bucketing-prefetch.md) |
| FSDP backward grad reduce | `reduce-scatter` per layer (or batched). | [reduce-scatter](concepts/reduce-scatter.md) |
| Data-parallel grad reduce | `all-reduce`. | [all-reduce](concepts/all-reduce.md) |
| Tensor-parallel matmul | `all-reduce` on output, OR `reduce-scatter` + `all-gather` for sequence-parallel-style. | (in collective-communication) |
| Expert-parallel MoE dispatch | `all-to-all` (forward) + `all-to-all` (backward). | (cite ringattention, tpu-inference codebases for examples) |
| Ring attention | `collective-permute` in a `while` loop. | [ringattention](codebases/ringattention.md) |
| Async overlap | `*-start` / `*-done` pairs (e.g. `all-gather-start` / `all-gather-done`). | [async-collectives](concepts/async-collectives.md) |

### Special op metadata to look at

- `metadata.op_name` — the framework-level name (e.g. `model/layer_0/attention/qkv`). Lets the agent jump from HLO back to the source code.
- `metadata.source_file` + `metadata.source_line` — exact source location.
- `frontend_attributes` — passed-through hints (e.g. `enable_async_collective=true`).
- `backend_config` — backend-specific (e.g. for `custom-call`, contains the kernel's serialized config).

---

## StableHLO reference

StableHLO is the **stable, version-controlled MLIR dialect** that JAX and PyTorch/XLA lower to before the XLA compiler takes over. It's a superset of HLO with a formal spec. Source repo: [`raw/code/stablehlo/`](../raw/code/stablehlo/). Wiki page: [`stablehlo` codebase page](codebases/stablehlo.md).

### Why the agent reads StableHLO

When analyzing a profile, the agent typically reads **HLO** (the XLA-internal form after StableHLO → HLO conversion). But two scenarios require touching StableHLO:

1. **Diagnosing illegal ops** — error messages like `mhlo.ragged_dot illegal` come from StableHLO legality checks. The fix usually requires inspecting whether the op is in spec at the StableHLO level.
2. **Comparing JAX-lowered IR across versions** — JAX's lowering target is StableHLO; behavioral changes between JAX versions sometimes surface as different StableHLO output for the same source. Useful when diffing a JAX-lane regression.

### Spec + passes

- Spec: [`raw/code/stablehlo/docs/spec.md`](../raw/code/stablehlo/docs/spec.md) — op semantics, type system, attribute grammar.
- Compatibility: [`raw/code/stablehlo/docs/compatibility.md`](../raw/code/stablehlo/docs/compatibility.md) — version guarantees (5-year forward / 1-month backward).
- Optimization passes: [`raw/code/stablehlo/stablehlo/transforms/`](../raw/code/stablehlo/stablehlo/transforms/) — the passes that run before HLO lowering. Notable: `stablehlo-aggressive-simplification`, `stablehlo-aggressive-folder`, `stablehlo-canonicalize-dynamism`.
- StableHLO → HLO bridge: [`raw/code/stablehlo/stablehlo/transforms/StablehloLegalizeToHlo.cpp`](../raw/code/stablehlo/stablehlo/transforms/StablehloLegalizeToHlo.cpp).

### Practical tooling

- `stablehlo-opt --stablehlo-XYZ` — run a single pass. Useful when reproducing a compile error locally.
- `stablehlo-translate --interpret` — execute StableHLO directly. Useful for verifying op semantics on tiny inputs.

The agent rarely runs these — but they're the escape hatch when an HLO question can't be answered from the dump alone.

---

## Pallas + custom_call signatures

When a hypothesis touches a kernel (Pallas / mosaic-tpu / tokamax / splash / ejkernel), the **`custom_call_target` string** in HLO is the structural signature that says "did the kernel actually fire?".

Reference: [custom-call](concepts/custom-call.md), [mosaic-kernel](concepts/mosaic-kernel.md), [manual-mlir-dialect-pallas](concepts/manual-mlir-dialect-pallas.md), [pallas-forge codebase](codebases/pallas-forge.md), [tokamax codebase](codebases/tokamax.md), [ejkernel codebase](codebases/ejkernel.md), and the Pallas kernel catalog at [`analyses/2026-04-23-pallas-kernel-directory.md`](analyses/2026-04-23-pallas-kernel-directory.md).

### Custom_call target naming

`custom-call(...)` carries `custom_call_target="<name>"`. Common targets the agent will see on TPU:

| Target | Kernel | Source |
|--------|--------|--------|
| `Mosaic` or `mosaic-tpu` (varies) | **Any Pallas-on-TPU kernel** (the generic Mosaic backend). Generic — the kernel name is buried in `backend_config`. | [pallas-forge](codebases/pallas-forge.md), [mosaic-kernel](concepts/mosaic-kernel.md) |
| `__gpu$xla.gpu.triton` | Triton-on-GPU. (Not TPU; you'd see this only on a GPU run.) | n/a |
| `splash_attention*` (or wrapped under Mosaic with `backend_config` indicating splash) | Splash multi-head / GQA attention kernel. | (in pallas-forge / jax.experimental.pallas.ops) |
| `tokamax_*` (often Mosaic-wrapped) | tokamax kernel (e.g. CE loss). | [tokamax](codebases/tokamax.md) |
| `ejkernel_*` | ejkernel suite (paged attention, NSA, FA3 ports). | [ejkernel](codebases/ejkernel.md) |
| `mhlo.scatter` / `mhlo.gather` etc. | Not custom — these are standard HLO. | (no kernel; reads `backend_config`) |

### Reading whether a Pallas kernel fired

The agent's Phase 3 check for "kernel-port" hypotheses is structural — grep the HLO for the expected `custom_call_target`:

```bash
# In get_hlo_dump output:
grep -c 'custom-call' module_after_optimizations.hlo
grep 'custom_call_target=' module_after_optimizations.hlo | sort -u
```

If the expected target isn't present, the kernel is **silent-noop** — the hypothesis didn't fire, regardless of measured MFU change. See [v675b session in memory](../memory/project_v675b_torch_compile_fires_first_time.md) for the canonical case where MFU change was real but the mechanism was different from what the hypothesis claimed.

### Backend_config inspection

`backend_config="..."` on a `custom-call` carries a serialized config (often base64-encoded protobuf or JSON). The agent should:
- Note the **target name** first.
- Note the **operand shapes** (sizes the kernel was invoked with).
- Only decode `backend_config` if the question is "what params was the kernel called with" — typically not needed for Phase 3 audit.

### Pallas kernel surfaces in raw/code

- `raw/code/jax/jax/experimental/pallas/` — Pallas frontend.
- `raw/code/jax/jax/experimental/mosaic/` — Mosaic backend (TPU).
- `raw/code/pallas-forge/` — community Pallas kernels.
- `raw/code/tokamax/` — kernel autotuner.
- `raw/code/ejkernel/` — attention kernel collection.

Catalog of available kernels with quick-pick by function: [`analyses/2026-04-23-pallas-kernel-directory.md`](analyses/2026-04-23-pallas-kernel-directory.md).

---

## HLO dumping + diffing

### Dump dir layout

Set with `XLA_FLAGS="--xla_dump_to=/path/to/dump_dir"`. Optional refinements:
- `--xla_dump_hlo_as_text` — text dumps (default).
- `--xla_dump_hlo_as_proto` — proto dumps.
- `--xla_dump_hlo_pass_re=.*` — dump after every pass (verbose; per-pass files).
- `--xla_dump_hlo_module_re=.*` — filter modules.

Filename conventions (handled transparently by [`get_hlo_dump`](#tool-get_hlo_dump)):
- **JAX**: `module_<id>.<name>.before_optimizations.hlo`, `module_<id>.<name>.after_optimizations.hlo`, `module_<id>.<name>.<pass_name>.hlo`.
- **PyTorch/XLA**: 4-digit sequence numbers per pass (`module_0000.<name>.before_optimizations.txt`, `module_0001...`).

Stage aliases the agent uses (defined at [internal/hlo_dump_tools.py:16-24](../raw/code/xprof-mcp/internal/hlo_dump_tools.py)):
- `before_optimizations` → first stage in the sequence.
- `after_optimizations` → last stage.
- `after_pass_<X>` → after a named pass.
- `pass_NNNN` → specific numbered pass.

### Diff workflow

[`diff_hlo_stages`](#tool-diff_hlo_stages) returns a unified diff between two stages of the same module. Typical questions answered:

- "Did fusion run?" → diff `before_optimizations` vs `after_optimizations`; count `fusion` lines.
- "What did pass X do?" → diff stages flanking X.
- "Why are my expected ops missing?" → diff to find when they were eliminated.

Reference: [hlo-dumping-and-diffing](concepts/hlo-dumping-and-diffing.md).

---

## Reading recipes (how-to patterns)

These are reusable analytical techniques the agent applies. Each is hypothesis-agnostic — they tell you HOW to extract a kind of signal from the profile/HLO, not WHAT to look for.

### HLO module count fingerprint

`list_hlo_modules` returns a module list. The **count + naming pattern** is a fingerprint of the execution mode:

| Pattern | Means |
|---------|-------|
| ~1-3 modules, source-named (e.g. `train_step`) | Fully `jit`-compiled / `torch.compile`-d with one entry. |
| ~10-50 modules, source-named per region | `torch.compile` with multiple compile regions (e.g. per-layer compiled). |
| 100+ modules, generic names (`module_<id>`) | Eager dispatch (each Python-level op compiled separately). |
| 1 module containing a `while` loop body | **Scan-over-layers** active — confirms the scan compile target. See [scan-over-layers](concepts/scan-over-layers.md), [v379 session in memory](../memory/project_v379_scan_over_layers_breakthrough.md). |

### Detecting eager vs compiled execution

- Many modules + per-op generic names → eager.
- Few modules + source-named regions → compiled.
- HLO ops that reference Python frame info via `metadata.source_file` → both can do this; check the **module count** as the disambiguator.

Specific failure mode: a torch.compile flag set, but module count says eager. This is the v675b structural failure — the wrapping broke the compile-region match. See [v675b memory](../memory/project_v675b_torch_compile_fires_first_time.md).

### Detecting recompilation

- xprof overview will show high host-idle %.
- Multiple identically-named modules with different IDs in `list_hlo_modules`.
- `list_xplane_events` with `event_regex='.*[Cc]ompile.*'` on `host.*` plane will show compile events deep into the run.

References: [jaxpr-hash-cache-keys](concepts/jaxpr-hash-cache-keys.md), [torch-tpu-compilation-cache](concepts/torch-tpu-compilation-cache.md).

### Finding the collectives gap

```
# 1. Get the headline gap:
get_overview → device_idle_time_percent  (high → there's a gap)

# 2. Enumerate collective events:
list_xplane_events(plane_regex='/device:TPU:0', event_regex='.*[Cc]ollective.*')

# 3. Quantify:
aggregate_xplane_events(plane_regex='/device:TPU:0', event_regex='.*[Cc]ollective.*')
# → per-collective count + total time

# 4. Find what blocks the device while collectives run:
list_xplane_events(plane_regex='/device:TPU:0', event_regex='.*Wait.*')
```

If `wait time on collective` / `collective time` is high (>40%), the collective is **not overlapping compute**. Reference: [async-collectives](concepts/async-collectives.md), [collective-bucketing-prefetch](concepts/collective-bucketing-prefetch.md).

### Finding a specific op in HLO

```
# 1. Search by name in the HLO text:
get_hlo_module_content(module_name) → grep for the op name

# 2. Expand neighborhood:
get_hlo_neighborhood(instruction_name=<name>, radius=2)
# → producers + consumers
```

For dump-based analysis, swap to [`get_hlo_dump`](#tool-get_hlo_dump) + [`get_hlo_dump_neighborhood`](#tool-get_hlo_dump_neighborhood).

### Reading the memory profile

`get_memory_profile` returns peak HBM + allocation timeline. The agent should look for:
- **Peak HBM** vs chip HBM cap (per the per-gen table above). Within 90% of cap ⇒ OOM-risk.
- **Stack/heap balance** — large stack ⇒ transient allocations (often during backward or splash attention).
- **Allocation spikes** — sudden growth at specific op IDs often points to a non-fused intermediate.

### Detecting a misaligned MXU shape

In HLO, look at the `dot` op's `dot_dimension_numbers`. The **contracting dim** size after lowering should be a multiple of the MXU shape (128 on v4/v5*, 256 on v6e+). If not, padding leaves MXU lanes idle.

```
# In HLO: %dot.42 = f32[4096,2048]{1,0} dot(...) lhs_contracting_dims={1}, rhs_contracting_dims={0}
# Contracting dim sizes need MXU-shape alignment after sharding.
```

References: [mxu](concepts/mxu.md), [mxu-utilization](concepts/mxu-utilization.md), [xla-cost-model](concepts/xla-cost-model.md).

### Reading SparseCore stats

If the run is on v5p+ with SparseCore enabled, `get_overview` will have `sc_*` keys mirroring the TC keys. Same reasoning shape, applied to SparseCore.

---

## Generation-specific gotchas

When analyzing a profile, these per-gen behaviors affect **interpretation** of what the trace shows. This section is descriptive (what to expect when reading a profile of gen X); prescriptive concerns (what flag to set when running on gen X) live in [`model-optimization-index.md`](model-optimization-index.md).

### v5p
- **Expect VMEM bucket to top out near 65472 KiB** under libtpu 0.0.41 (universal cap; higher-requested values silently fall back). If the profile shows `xla_tpu_scoped_vmem_limit_kib` requested above this, the actual VMEM used will be capped — don't read the *requested* number as the *operative* one.
- **Chip count is half the XPK name**: `v5p-N` counts TCs (2 TCs per chip). When you see "v5p-16" in run metadata, divide by 2 for chip count when reasoning about per-chip metrics. See [feedback memory v5p naming convention](../memory/feedback_v5p_naming_convention.md).
- **Multi-host hang signature**: absence of `--xla_tpu_use_enhanced_launch_barrier=false` on multi-host runs (≥v5p-16) often correlates with FSDP all-gather deadlocks, which appear in xprof as long `Wait` events on `host.*` planes synchronized across workers — the workload looks "running" but step time grows unboundedly. See [feedback memory libtpu enhanced launch barrier](../memory/feedback_libtpu_enhanced_launch_barrier.md).

### v6e (Trillium)
- **MXU shape is 256×256** (vs 128×128 on v4/v5*). When checking [misaligned MXU shape](#detecting-a-misaligned-mxu-shape), use 256 as the alignment unit. A `dot` aligned for 128 leaves half the MXU idle on v6e — the symptom is MXU utilization stuck below 50% even on compute-bound ops.
- **HBM cap 32 GB is tight**: memory profile peaks frequently sit at 80-95% of cap on larger models. Expect more HBM-bound ops than on v5p.
- **Critical FLOP/byte ridge ~560** (vs ~165 on v5p). Ops that read as compute-bound on a v5p trace may read as memory-bound on the same op-point on v6e — the same arithmetic intensity straddles the ridge differently.

### v7x (Ironwood)
- **Massive bf16 peak (4614 TF) + native FP8 path (9.2 PF)**: when interpreting MXU utilization on v7x, check whether the dominant `dot` ops are in bf16 or FP8 — FP8 ops report against the higher peak. A naive "MXU is low" reading may be wrong if FP8 isn't accounted for.
- **HBM 192 GB + 7.4 TB/s BW**: entire memory regime is different. Recipes that were HBM-bound on v5p/v6e may have lots of headroom here. Expect lower HBM utilization % at the same op-point.
- **ICI BW jumped 8×**: collective overhead drops dramatically. A profile that would show 40% collective gap on v5p may show <10% on v7x — re-evaluate "collective-bound" verdicts when crossing generations.

### v8 (preview)
- Numbers TBD. When announced, the [per-generation envelope](#per-generation-envelope) and [`model-optimization-index.md`](model-optimization-index.md) hardware envelope are canonical.

---

## Observation page template

Findings extracted from a profile are filed as `wiki/observations/<slug>.md`. Per SCHEMA:

```yaml
---
title: "<short one-line title>"
type: observation
tags: [...]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

Lead paragraph (1-3 sentences) defining the observation.

## What was observed
<the finding, in concrete terms — numbers, op names, HLO patterns>

## How to see it
<exact tool call(s) that reproduce it>:
- Profile path: `raw/profiles/<YYYY-MM-DD>-<exp-slug>/`
- Run name: `<run>`
- Tool calls:
  - `get_top_hlo_ops(run='<run>', host='<host>')` → top op is `<name>` consuming `<time>`
  - `get_hlo_neighborhood(run='<run>', instruction_name='<name>')` → shows `<pattern>`
- HLO dump dir (if applicable): `raw/profiles/<...>/hlo/`
  - `diff_hlo_stages(dump_dir='<dir>', module='<m>', stage_a='before_optimizations', stage_b='after_optimizations')` → `<what differs>`

## Implications
<what this means — be careful: implications are hypothesis-adjacent. Keep this section
descriptive ("X happens because Y in the compiler") not prescriptive ("we should
try Z"). Prescriptive content belongs in hypothesis pages, not observations.>

## Seen in
- [experiments/.../<exp>.md] — original observation
- [experiments/.../<other-exp>.md] — same pattern repeated

## See also
## Sources
```

### Cardinal rule reminder for observations

An observation describes **what the profile / HLO shows**. It does NOT propose what to try next. If you have a "we should try X" thought, file a **hypothesis** page in `wiki/hypotheses/`, not an observation. The observation can be cited from the hypothesis page as its `origin:`.

---

## See also

- [`SCHEMA.md`](../SCHEMA.md) — wiki structure, experiment page rules, the `profile-analyzer` agent's role in RUN-EXPERIMENT.
- [`.claude/agents/profile-analyzer.md`](../.claude/agents/profile-analyzer.md) — the consumer of this index. Phase 0 reads this page in full.
- [`wiki/model-optimization-index.md`](model-optimization-index.md) — the sibling index for **hypothesis-generation** material (cardinal rule: only optimization recommendations). Strict separation.
- [`wiki/codebases/xprof.md`](codebases/xprof.md) — xprof server internals.
- [`wiki/codebases/xprof-mcp.md`](codebases/xprof-mcp.md) — the MCP server source-of-truth (full tool catalog + tool signatures).
- [`wiki/codebases/stablehlo.md`](codebases/stablehlo.md) — StableHLO spec + passes.
- [`analyses/2026-04-23-pallas-kernel-directory.md`](analyses/2026-04-23-pallas-kernel-directory.md) — Pallas kernel catalog.
- [`profile-analyzer-index-regenerate-prompt.md`](profile-analyzer-index-regenerate-prompt.md) — how to rebuild this page.

## Sources

The substrate sources this index rests on:

- [`sources/2026-xprof-overview-page.md`](sources/2026-xprof-overview-page.md), [`sources/2026-xprof-terminology.md`](sources/2026-xprof-terminology.md) — metric definitions.
- [`sources/2026-xprof-hlo-op-profile.md`](sources/2026-xprof-hlo-op-profile.md), [`sources/2026-xprof-hlo-op-stats.md`](sources/2026-xprof-hlo-op-stats.md) — HLO-op profiler views.
- [`sources/2026-xprof-memory-profile.md`](sources/2026-xprof-memory-profile.md), [`sources/2026-xprof-memory-viewer.md`](sources/2026-xprof-memory-viewer.md) — memory analysis.
- [`sources/2026-xprof-trace-viewer.md`](sources/2026-xprof-trace-viewer.md), [`sources/2026-xprof-roofline-model.md`](sources/2026-xprof-roofline-model.md), [`sources/2026-xprof-perf-counters.md`](sources/2026-xprof-perf-counters.md) — trace + roofline + counters.
- [`sources/2026-xprof-custom-call-profiling.md`](sources/2026-xprof-custom-call-profiling.md), [`sources/2026-xprof-mcp-tpu-optimization.md`](sources/2026-xprof-mcp-tpu-optimization.md) — custom_call + MCP triage flow.
- [`sources/2026-xprof-capturing-profiles.md`](sources/2026-xprof-capturing-profiles.md), [`sources/2026-xprof-jax-profiling.md`](sources/2026-xprof-jax-profiling.md), [`sources/2026-xprof-pytorch-xla-profiling.md`](sources/2026-xprof-pytorch-xla-profiling.md) — capture how-tos.
- [`sources/2026-xprof-megascale-stats.md`](sources/2026-xprof-megascale-stats.md), [`sources/2026-xprof-megascale-viewer.md`](sources/2026-xprof-megascale-viewer.md), [`sources/2026-xprof-megascale-viewer-sql.md`](sources/2026-xprof-megascale-viewer-sql.md) — multi-host views.
- [`sources/2026-cloud-tpu-v4-docs.md`](sources/2026-cloud-tpu-v4-docs.md), [`sources/2026-cloud-tpu-v5e-docs.md`](sources/2026-cloud-tpu-v5e-docs.md), [`sources/2026-cloud-tpu-v5p-docs.md`](sources/2026-cloud-tpu-v5p-docs.md), [`sources/2026-cloud-tpu-v6e-docs.md`](sources/2026-cloud-tpu-v6e-docs.md), [`sources/2026-cloud-tpu-v7x-ironwood-docs.md`](sources/2026-cloud-tpu-v7x-ironwood-docs.md) — per-gen TPU device docs.
- [`sources/2025-scaling-book-ch1-roofline.md`](sources/2025-scaling-book-ch1-roofline.md), [`sources/2025-scaling-book-ch2-tpus.md`](sources/2025-scaling-book-ch2-tpus.md), [`sources/2025-scaling-book-ch9-profiling.md`](sources/2025-scaling-book-ch9-profiling.md) — Scaling Book foundations.
- [`sources/2025-ultrascale-playbook.md`](sources/2025-ultrascale-playbook.md) — profiling playbook.
- [`raw/code/stablehlo/docs/spec.md`](../raw/code/stablehlo/docs/spec.md) — StableHLO spec (in repo).

## How to maintain this file

- New xprof source / concept ingested? Cite it under the appropriate § (tool catalog / bucket taxonomy / device model / HLO ref / Pallas / dumping / recipes).
- New TPU generation? Update the [per-generation envelope](#per-generation-envelope) and add a [generation-specific gotchas](#generation-specific-gotchas) subsection.
- New mcp__xprof__ tool? Add a row to the [tool catalog](#mcp__xprof__-tool-catalog) with file path + line + 1-line use.
- New custom_call_target observed? Add a row to [Pallas custom_call signatures](#pallas-custom_call-signatures).
- Every line passes the **cardinal rule test**: it earns its place by helping the agent INTERPRET a profile or HLO dump. If it's prescriptive ("try X"), it goes in model-optimization-index, not here.

Periodic full regeneration via [profile-analyzer-index-regenerate-prompt.md](profile-analyzer-index-regenerate-prompt.md) when 10+ new sources have landed, a new TPU gen ships, or the structure itself needs rethinking.
