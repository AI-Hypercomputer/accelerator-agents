---
title: "XProf Megascale Viewer"
type: source
tags: [docs, profiler, megascale, dcn, collective-communication, perfetto, multi-slice, action-graph]
created: 2026-04-22
updated: 2026-04-22
---

The Megascale Viewer is xprof's replacement for the older Megascale Stats tool. It opens a profile inside Perfetto UI and exposes, per host, aggregated network counters plus per-TPU timelines that include a **Megascale** track whose children are the action graphs of individual Megascale collectives. This lets the user connect a slow TPU op (e.g. a long `recv-done`) to the concrete host-side action (NetworkSend / NetworkReceive / D2H / H2D) that unblocked it, and read network latency, transfer size, and peer device IDs directly from event arguments.

## Overview

Access: xprof → Tools dropdown → **Megascale Viewer** → pick a host → **Open in Perfetto**.

Trace structure:

- **Global Counters** (host-level): e.g. network utilization. Aggregated across all megascale collectives on all profiled TPUs of the selected host.
- **TPUs** section, one group per TPU. Expanding a TPU reveals:
  - **Steps** — training steps as tracked by xprof.
  - **XLA Modules** — all XLA program runs during the profile.
  - **XLA Ops** — all XLA ops and custom kernels on the TPU.
  - **XLA TraceMe** — synchronization events (e.g. `barrier-cores`).
  - **Megascale** — parent track; one child per unique Megascale collective. Each child's events form that collective's action graph.

Child track names encode `collective_name [megascale_device_id]`, where `megascale_device_id = (slice_id, logical_tpu_id)`.

Connecting TPU ops to the megascale action graph:

- Click a `recv-done` on the TPU → in the bottom panel, click its **`recv-done END`** flow event → an arrow jumps to the megascale graph execution that unblocked it.
- Click **`DeviceToHost START`** at the beginning of an action graph → arrow jumps to the corresponding `send` op on the TPU.
- The `recv-done END` / `NetworkReceive END` events are **synthetic** post-processing markers that exist only to make Perfetto flow arrows work — their on-screen duration is not meaningful.

Action collapsing:

- The Megascale child track renders a single slice at any given time even though multiple actions are in flight. The displayed slice is the one that **finishes first**. Real per-action durations and latencies live in event arguments in the bottom panel — notably `action_duration_ns` and `network_transport_latency_us`. The doc gives a concrete example where the rendered slice reads 13.5 ms but the true `action_duration_ns` is 20.6 ms.

Statistics via PerfettoSQL:

- Events are queryable through the **Query (SQL)** page in Perfetto. A macro'd UI for these is promised; for now users paste queries directly. Switching to the Timeline tab shows query results inline; clicking a slice id jumps to that event.

## Key claims

- **Successor to Megascale Stats.** This doc explicitly replaces `megascale_stats.md`; the old tool will be deprecated.
- **Per-host granularity.** You select one host at a time; global counters are host-aggregated.
- **Per-collective action graphs.** Each unique Megascale collective gets its own child track with its full action graph (D2H, NetworkSend, NetworkReceive, H2D events).
- **Synthetic END events carry flow links.** `recv-done END` and `NetworkReceive END` are not real TPU events — they exist solely so Perfetto flow arrows can connect TPU ops to host/network actions.
- **Rendered slice duration on Megascale tracks is a "first-to-finish" artifact.** Ground truth for a given action lives in `action_duration_ns`. Reading the timeline slice width as duration will mislead.
- **`network_transport_latency_us` is exposed per action.** Network time is a first-class argument on NetworkReceive/NetworkSend events.
- **Outliers are found via PerfettoSQL stats.** The canonical workflow is "recv-done stats" grouped by op and device, sorting by `dur_ns_sum` to rank collectives by total time cost, and `p99_over_mean` to flag heavy-tail variance.
- **Plotting duration over time (counter debug track).** Using Perfetto's "Add debug track" with type "counter" against the `recv-done ops` query produces a line chart; the doc observes that recv-done duration spikes tend to cluster at step boundaries.
- **Single slow instance is reachable.** The `id` column from SQL links directly into the timeline; pressing **F** zooms to the event.

## Key data points

### Event arguments that matter on Megascale tracks

| Argument | Meaning |
|---|---|
| `action_duration_ns` | Real duration of the action (ignore rendered slice width) |
| `network_transport_latency_us` | On-wire latency for a Network* action |
| `buffer_sizes` | Bytes transferred |
| `run_id` | XLA run identifier — joins to other tracks |
| Peer device IDs | Available in arguments; action target |

### Megascale child-track naming

| Part | Meaning |
|---|---|
| Collective name | From XLA (same as in Trace Viewer) |
| `[device_id]` | `(slice_id, logical_tpu_id)` — the local TPU's megascale ID |

### User journey presented in the doc

| Step | Action |
|---|---|
| 1 | Run `recv-done stats`, sort by `duration_ns_sum` — finds the dominant recv-done (example: recv-done.28 ≈ 2.2 s of 14 s) |
| 2 | Note high p99 vs. median/mean — indicates heavy-tail variance, i.e. headroom |
| 3 | (Optional) Add counter debug track for `recv-done ops` to see temporal pattern — spikes tended to hit at step boundaries |
| 4 | Re-run `recv-done ops` filtered to `recv-done.28`, sort descending, click `id` → jump to slowest instance |
| 5 | Hit **F** to zoom fully into the event |
| 6 | Click `recv-done END` flow → jump to the action graph execution that unblocked it |
| 7 | Inspect NetworkReceive entries; the doc's example shows a 27 ms transfer for 3.5 MiB with `action_duration_ns` of 47.2 ms |

### Synthetic vs. real events

| Event | Synthetic? | Notes |
|---|---|---|
| `recv-done END` | Yes | Exists only for flow arrow to megascale graph |
| `NetworkReceive END` | Yes | Same; its slice width is not the action duration |
| `DeviceToHost START` | No (visible as instant) | Click to jump back to the `send` on the TPU |

## Techniques referenced

- **Perfetto UI as the trace browser** — xprof embeds Perfetto for the viewer.
- **Action graph of a Megascale collective** — D2H → NetworkSend → NetworkReceive → H2D as a connected sequence of host/network events linked to TPU send/recv-done.
- **Flow events** — Perfetto's mechanism for arrow-linking events across tracks; used here to tie TPU ops to host actions via synthetic END events.
- **Debug counter tracks** — transforming a SQL query result into a line-chart counter on the timeline for temporal pattern detection.
- **p99-over-mean as a heavy-tail indicator** — a specific signal highlighted by the doc for finding optimization headroom.
- **Pinning tracks** — UI trick to keep distant tracks adjacent on screen.

## Gaps & caveats

- **Single host at a time.** Multi-host/multi-slice analyses must be done per host, then reasoned about manually.
- **Rendered durations on Megascale tracks lie.** Any hypothesis derived from the timeline's visual widths without reading `action_duration_ns` will be wrong.
- **Collapsed track shows "first to finish".** If you want to understand an in-flight action that is ultimately long-running, it may not be the visible slice — only the arguments/SQL reveal it.
- **UI macros for common queries don't exist yet.** The doc flags this as future work; for now stats require hand-pasted SQL.
- **No bandwidth roof.** Unlike Megascale Stats, this doc does not surface a "required bandwidth vs. host DCN" comparison; that math is on the user.
- **Step-boundary spike observation is anecdotal.** The doc's example shows recv-done duration spiking at step boundaries; there is no framework claim that this is general.
- **ICI collectives not in scope.** Megascale = DCN/cross-slice; intra-slice ICI work is not covered here.
- **Doc lacks a comprehensive event-type list.** Only NetworkSend, NetworkReceive, DeviceToHost, and H2D are explicitly named; others in the action graph are not enumerated.

## Connections

- `megascale`
- `megascale-viewer`
- `megascale-action-graph`
- `dcn`
- `dcn-collective`
- `collective-communication`
- `send-recv-done`
- `network-transport-latency`
- `perfetto`
- `perfetto-flow-events`
- `perfetto-debug-counter`
- `multi-slice`
- `step-boundary-collective-spikes`

## See also

- [xprof](../codebases/xprof.md)
- [XProf Megascale Viewer SQL](2026-xprof-megascale-viewer-sql.md) — sample queries referenced here
- [XProf Megascale Stats Tool](2026-xprof-megascale-stats.md) — predecessor
- [XProf Roofline Model Tool](2026-xprof-roofline-model.md)

## Sources

- `raw/code/xprof/docs/megascale_viewer.md`
