---
title: "XProf Megascale Stats Tool"
type: source
tags: [docs, profiler, megascale, dcn, collective-communication, multi-slice, tpu-stall]
created: 2026-04-22
updated: 2026-04-22
---

The Megascale Stats tool analyzes inter-slice collective communication performance for workloads that span multiple TPU slices over the Data Center Network (DCN). It produces a per-collective, per-TPU table that quantifies how much time each collective spends stalling the TPU vs. transmitting data, and what bandwidth would be required to fit the transfer inside the available slack. **Note:** this tool is slated for deprecation and is being replaced by the Megascale Viewer.

## Overview

In multi-slice training, each inter-slice collective is expressed on the TPU as a four-op chain: `send` → (transfer) → `send-done`, and on the receiving side `recv` → (transfer) → `recv-done`. A collective is the span from a `send` to its matching `recv-done`. The non-transfer ops (`send`, `send-done`, `recv`, `recv-done`) block the TPU — their aggregate time is the collective's **stall**. The transfer itself happens asynchronously in between; the time budget available for it without stalling the TPU is the **slack**.

The tool presents one row per profiled collective with durations, stall, slack, bytes, and the implied required bandwidth (bytes / slack). The intended workflow is to sort by aggregated total stall, identify the dominant offender, check whether total required bandwidth (summed across cores on a host) exceeds the host's DCN bandwidth (network congestion), and then drop to Trace Viewer / HLO dump to diagnose whether the root cause is scheduling, sharding, or hardware.

TPU-only. Collectives here are DCN collectives (cross-slice); ICI collectives within a slice are out of scope.

## Key claims

- **Collective = send → recv-done.** A collective is initiated by `send` and completed by the matching `recv-done`. Transfer happens after `send` completes and before `recv-done`; `send-done` fires after data is on the wire, `recv-done` after it has been received.
- **Stall is the TPU-blocking time.** Stall duration = t_send + t_send-done + t_recv + t_recv-done, i.e. only the four op endpoints, excluding transfer time.
- **Slack is the network-independent budget.** Slack time is the gap(s) between the four ops when the TPU is doing other work; increasing slack reduces the chance that the collective stalls the TPU.
- **Required bandwidth = bytes / slack.** This is the bandwidth needed to fit the collective's transfer entirely within its slack. If `required_bw × cores_per_host` exceeds the host's DCN bandwidth, multiple collectives are competing for the wire — likely network congestion.
- **Aggregated total stall is the primary ranking metric.** It equals stall_duration × occurrences; sorting by it surfaces the collective that costs the most TPU time over the profile.
- **Fan-out of `send`/`recv-done` can reduce stall.** The analysis workflow suggests that the compiler scheduling more HLO ops between fanned-out sends and recv-dones increases overlap and reduces stall.
- **Diagnostic split.** If `recv-done` duration is high vs. slack → bandwidth bottleneck. If `recv-done` duration is low vs. slack → possible hardware issue.

## Key data points

### Columns in the per-collective table

| Column | Definition |
|---|---|
| DCN collective name | Assigned by XLA |
| Recv op name | The TPU `recv-done` op name (searchable in Trace Viewer) |
| Send op name | The TPU `send` op name |
| Slack time | Gaps between the four endpoint ops — time available for transfer without stalling |
| Observed duration | t_send + t_1 + t_send-done + t_2 + t_recv + t_3 + t_recv-done (full span) |
| Stall duration | t_send + t_send-done + t_recv + t_recv-done (endpoint ops only) |
| Occurrences | Number of completed send → recv-done pairs in the profile |
| Aggregated total stall | stall_duration × occurrences |
| Data transmitted size | From XLA op shape |
| Required bandwidth | data_transmitted_size / slack_time |

### Analysis procedure (from the doc)

| Step | Action | What it tells you |
|---|---|---|
| 1 | Sort by aggregated total stall, descending | Finds the dominant collective |
| 2 | Inspect top offender | If it dwarfs the rest, it's the bottleneck |
| 3 | Compute required_bw × cores_per_host (e.g. 8 on v4) | Compare to host DCN bandwidth — over ⇒ congestion |
| 4 | Mitigate via sharding changes | Reduces per-collective bytes / required bandwidth |
| 5 | Generate HLO dump, look for fan-out of send/recv-done | Enables more overlap scheduling |
| 6 | Open Trace Viewer, inspect recv-done duration | High transfer time ⇒ bandwidth-bound |
| 7 | If recv-done duration is low vs. slack | Suspect hardware issue |

## Techniques referenced

- **DCN collectives** — cross-slice send/recv over the data center network.
- **Send/recv-done op quartet** — XLA's representation of an asynchronous collective as four discrete TPU ops.
- **Slack-based overlap** — scheduling non-collective work between send and recv-done so the transfer hides behind compute.
- **Sharding changes** — altering data parallel / model parallel partitioning to reduce per-collective bytes.
- **HLO-level fan-out of send/recv-done** — compiler scheduling trick to surface more independent work between endpoints.
- **Per-host bandwidth sanity check** — multiplying required_bw by cores per host to detect network oversubscription (e.g. 8 cores per v4 TPU host).

## Gaps & caveats

- **Deprecation.** The doc states this tool will be deprecated in favor of the Megascale Viewer; long-lived hypotheses should reference the viewer doc as the authoritative source when the two differ.
- **DCN only.** ICI (intra-slice) collectives are not analyzed by this tool.
- **No per-slice DCN bandwidth number.** The doc instructs the user to compare against "the maximum network bandwidth of the TPU" but does not provide values per generation.
- **"Cores per host" example is v4-specific.** 8 per host is given; v5/v6 generations differ and are not specified.
- **`Required bandwidth` assumes the whole transfer must fit in slack.** If the schedule already hides the transfer partially, the metric can be pessimistic — a collective above the per-host DCN bandwidth threshold is not automatically broken.
- **Aggregated total stall ignores order / overlap between distinct collectives.** Two collectives each with 10ms stall can (in principle) overlap — the metric counts 20ms either way.
- **HLO-fan-out remedy is hand-wavy.** The doc points at fanning out send/recv-done but does not prescribe flags or passes.
- **Hardware-issue diagnosis is a residual.** "recv-done not excessively high vs. slack ⇒ maybe hardware" is a weak signal; no threshold given.

## Connections

- `megascale`
- `dcn`
- `dcn-collective`
- `collective-communication`
- `send-recv-done`
- `slack-time`
- `stall-duration`
- `aggregated-total-stall`
- `required-bandwidth`
- `sharding`
- `hlo-dump`
- `multi-slice`
- `megascale-viewer` (successor tool)

## See also

- [xprof](../codebases/xprof.md)
- [XProf Megascale Viewer](2026-xprof-megascale-viewer.md) — successor tool
- [XProf Megascale Viewer SQL](2026-xprof-megascale-viewer-sql.md)
- [XProf Roofline Model Tool](2026-xprof-roofline-model.md)

## Sources

- `raw/code/xprof/docs/megascale_stats.md`
