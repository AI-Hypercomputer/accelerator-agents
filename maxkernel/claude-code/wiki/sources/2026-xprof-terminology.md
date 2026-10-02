---
title: "XProf Terminology (docs)"
type: source
tags: [docs, profiler, xprof, terminology]
created: 2026-04-22
updated: 2026-04-22
---

Short glossary of the core terms XProf uses to talk about profiling: profile, session, run, host, device, step. Anchor definitions for the rest of the XProf source pages.

## Overview

The terminology page defines six base concepts used across all XProf docs. Each term is a precise, narrow meaning that other tools assume without restating — in particular, `Step` is the unit of measurement for performance reports, and the `host` / `device` split underlies every timeline, memory view, and utilization chart.

## Key claims

- **Profile**: the data captured about a program's execution performance (memory, op durations, transfer sizes, etc.). The noun for the entire captured record.
- **Session**: one specific data-capture instance, with a unique name. Each subdirectory under `plugins/profile/` represents one session.
- **Run**: one training job or workflow end-to-end; synonymous with an "experiment" in XProf's vocabulary. A run may contain multiple sessions.
- **Host**: the system CPU — controls program flow and data transfer. "Host memory" = system RAM.
- **Device**: the accelerator (GPU or TPU) — executes the actual computations. "Device memory" = the accelerator's HBM.
- **Step**: one iteration of the training loop. `Step time` is the per-iteration wall-clock duration, and it is **the unit of measurement XProf uses to report performance**.
- XLA-specific terminology is deferred to the OpenXLA terminology page (linked out of the doc), not defined here.

## Key data points

### Term table

| Term | Definition | Notes |
|---|---|---|
| Profile | Captured performance data | Umbrella noun |
| Session | One capture instance | Subdirectory under `plugins/profile/` |
| Run | One training job / workflow | == experiment |
| Host | System CPU | Host memory = RAM |
| Device | Accelerator (GPU/TPU) | Device memory = HBM |
| Step | One iteration of training loop | Step time is XProf's headline metric |

### Directory convention

- `plugins/profile/<session-name>/` — one directory per profiling session.

### Relationships

- A run contains one or more sessions; each session produces one profile.
- The host/device split is used by Trace Viewer (sections), Memory Profile (host vs. device memory), Memory Viewer (memory types), and the Overview Page (step-time host-vs-device categorization).

## Techniques referenced

- Normalizing performance reporting to `step time` as the top-line metric.
- Distinguishing capture instance (session) from workflow (run) from raw data (profile).
- Explicit host-vs-device semantic separation in memory and compute terminology.

## Gaps & caveats

- The page is intentionally minimal; XLA-specific terms (HLO, fusion, etc.) are off-loaded to the OpenXLA terminology doc and not duplicated.
- "Session" directory convention (`plugins/profile/`) is stated as the canonical layout — not enumerated for edge cases (multi-host, captures without the plugin path).
- "Run == experiment" is a local convention; this wiki distinguishes "experiment" pages (hypothesis-bound runs) from arbitrary runs, so the terms should not be conflated cross-wiki.
- `Step` is defined for training; inference has "sessions" (per the Overview Page) that serve a similar role but are not covered here.

## Connections

Concept slugs this source informs:

- `profile` — the captured record.
- `profiling-session` — single capture with directory semantics.
- `run` — one training workflow.
- `host` — CPU side of the system.
- `device` — accelerator side.
- `step` — iteration unit.
- `step-time` — per-iteration wall-clock, XProf's headline metric.
- `host-device-split` — foundational dichotomy across every XProf view.

## See also

- [xprof](../codebases/xprof.md)
- [xprof overview page](2026-xprof-overview-page.md)
- [xprof trace viewer](2026-xprof-trace-viewer.md)
- [xprof memory profile](2026-xprof-memory-profile.md)
- [xprof memory viewer](2026-xprof-memory-viewer.md)
- [xprof graph viewer](2026-xprof-graph-viewer.md)
- [xprof utilization viewer](2026-xprof-utilization-viewer.md)

## Sources

- `raw/code/xprof/docs/terminology.md`
