---
title: "XProf Megascale Viewer — PerfettoSQL Query Cookbook"
type: source
tags: [docs, profiler, megascale, dcn, perfetto, perfettosql, sql, collective-communication]
created: 2026-04-22
updated: 2026-04-22
---

A short cookbook of three PerfettoSQL queries to run inside the Megascale Viewer in Perfetto UI: list all recv-done ops, compute per-op duration statistics (p50/p90/p99/mean/sum/count, plus p99/mean ratio), and list all `NetworkReceive END` actions with network latency and transfer size extracted from event arguments. These are the queries the Megascale Viewer doc uses for its outlier-finding user journey.

## Overview

The queries exploit Perfetto's `slice` / `track` tables plus `EXTRACT_ARG(arg_set_id, 'debug.<key>')` for pulling typed arguments out of events. Two patterns recur:

1. Join `slice → track → track` twice — once on `track_id` to get the containing track (e.g. `XLA Ops`), and once on that track's `parent_id` to get the grandparent (the TPU device track). The device name is read off the grandparent.
2. Match event names via `REGEXP 'recv-done.\d+$'` to pick up all HLO-numbered variants of a given op.

All three queries target slices whose containing track has `name GLOB '*XLA Ops*'` (case-insensitive) for TPU ops, or `name = 'Megascale'` for action-graph events.

## Key claims

- **Three canonical queries are provided:** `recv-done ops`, `recv-done stats`, `NetworkReceive Actions`.
- **`recv-done ops` returns raw slices** with `id`, `name`, `ts`, `dur`, `run_id`, `device`, `track_id`, `slice_id` — enough to both power a debug counter track and to click-through from a result row into the timeline.
- **`recv-done stats` aggregates per `(name, device)`** producing p50, p90, p99, mean, count, sum, and **`p99_over_mean`** — the last is explicitly called out in the viewer doc as the heavy-tail/headroom indicator.
- **`NetworkReceive Actions` pulls arguments** from the megascale action graph: `network_transport_latency_us`, `action_duration_ns`, `buffer_sizes`, `run_id`. This is how the user journey reads the "real" duration rather than the misleading rendered slice width.
- **`run_id` is exposed on both sides** — `debug.run_id` appears as an extractable argument on both XLA Ops `recv-done` slices and Megascale `NetworkReceive END` slices, enabling joins.
- **Device identity is read from the grandparent track.** `ppt.name AS device` in the joined hierarchy names the TPU.

## Key data points

### Query 1 — recv-done ops

| Column | Source | Use |
|---|---|---|
| `s.id` | slice id | Click-through to timeline |
| `s.name` | e.g. `recv-done.28` | Distinguish collectives |
| `s.ts`, `s.dur` | timestamp, duration | Temporal filtering, debug counter |
| `run_id` | `EXTRACT_ARG(arg_set_id, 'debug.run_id')` | Join key |
| `device` | grandparent track name | Per-TPU grouping |
| `s.track_id`, `s.slice_id` | identifiers | Reference |

Filter: `LOWER(pt.name) GLOB LOWER('*XLA Ops*')` AND `s.name REGEXP 'recv-done.\d+$'`.

### Query 2 — recv-done stats

| Column | Definition |
|---|---|
| `name` | recv-done op name |
| `device` | TPU (parent track name in this query — note: `pt.name` is used as `device` here, not `ppt.name`) |
| `dur_ns_p50` | `ROUND(PERCENTILE(dur, 50), 2)` |
| `dur_ns_p90` | p90 |
| `dur_ns_p99` | p99 |
| `dur_ns_mean` | `ROUND(AVG(dur), 2)` |
| `count` | `COUNT(*)` |
| `dur_ns_sum` | `SUM(dur)` — total time cost; primary ranking |
| `p99_over_mean` | `PERCENTILE(dur, 99) / AVG(dur)` — heavy-tail ratio |

Grouped by `(name, device)`.

### Query 3 — NetworkReceive Actions

| Column | Argument key |
|---|---|
| `network_latency_us` | `debug.network_transport_latency_us` |
| `action_duration_ns` | `debug.action_duration_ns` (real duration, not rendered width) |
| `buffer_sizes` | `debug.buffer_sizes` (transfer size) |
| `run_id` | `debug.run_id` |
| `device` | grandparent track name |

Filter: `pt.name = 'Megascale'` AND `s.name = 'NetworkReceive END'`.

### Track hierarchy used

```
device-track  (TPU name — ppt)
 └── container-track  (e.g. "XLA Ops" or "Megascale" — pt)
      └── slice-track  (where s lives; or slices may live directly on container)
```

The two-join pattern (`slice → pt ON track_id` then `pt → ppt ON parent_id`) is how the queries name the device.

## Techniques referenced

- **PerfettoSQL** over the `slice` and `track` tables.
- **`EXTRACT_ARG(arg_set_id, 'debug.<key>')`** for reading typed event arguments into SQL columns.
- **`PERCENTILE(col, p)`** aggregate for p50/p90/p99.
- **Regex op-name matching** (`REGEXP 'recv-done.\d+$'`) to group HLO-numbered variants.
- **`p99_over_mean` as a heavy-tail metric** — a simple ratio to flag distributions worth chasing.
- **Track hierarchy joins** to surface device identity from the grandparent track.
- **Slice-id click-through** — `s.id` in the result lets you jump to the event in timeline view.

## Gaps & caveats

- **Only three queries.** Nothing here for send-done, per-collective bandwidth, step-boundary correlation, or cross-host aggregation — those are exercises for the user.
- **Stats query uses `pt.name` for device, but `recv-done ops` uses `ppt.name`.** This is a minor inconsistency in the doc; the two queries will label "device" from different levels of the track hierarchy. If tracks are organized so `XLA Ops` sits directly on the device track, `pt.name` in the stats query will return `'XLA Ops'`, not the TPU name. Hypotheses that group by `device` off the stats query should verify they are grouping by what they think they are.
- **Queries do not join XLA Ops recv-done slices to their NetworkReceive counterparts.** `run_id` is exposed on both sides to make this possible, but the join is not written down.
- **No collective-name translation.** `recv-done.28` vs. the XLA-assigned DCN collective name in the Megascale Stats table — the mapping is not shown in SQL form.
- **UI-macro roadmap.** The viewer doc says these queries will become UI macros; paths/names will change.
- **`dur` units.** Called `_ns` in the stats query aliases — so Perfetto slice durations are in nanoseconds for this workload; the `NetworkReceive` query exposes `_us` for network latency. Mixed units within one query result: read the suffixes.
- **No covering index guarantees.** On very large traces these queries scan `slice`; the doc does not advise on trace size limits.

## Connections

- `perfetto`
- `perfettosql`
- `megascale-viewer`
- `recv-done-stats`
- `recv-done-ops`
- `networkreceive-action`
- `network-transport-latency`
- `p99-over-mean`
- `heavy-tail-latency`
- `dcn-collective`
- `collective-communication`
- `extract-arg`

## See also

- [xprof](../codebases/xprof.md)
- [XProf Megascale Viewer](2026-xprof-megascale-viewer.md) — parent doc; user journey that invokes these queries
- [XProf Megascale Stats Tool](2026-xprof-megascale-stats.md)
- [XProf Roofline Model Tool](2026-xprof-roofline-model.md)

## Sources

- `raw/code/xprof/docs/megascale_viewer_sql.md`
