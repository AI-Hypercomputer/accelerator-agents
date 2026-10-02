---
title: "Megascale viewer"
type: source
provenance: docs
source_doc: docs/megascale_viewer.md
updated: 2026-07-03
status: fresh
---
# Megascale viewer

The Megascale Viewer visualizes the Megascale action graph, connecting TPU ops to collective actions to diagnose latency (e.g. slow recv-done instances).

## Key points
- Connecting TPU ops to the Megascale action graph and the trace layout ([connecting-tpu-ops-to-megascale-action-graph](src:docs/megascale_viewer.md#connecting-tpu-ops-to-megascale-action-graph))
- Example user journey: find slow instances of a recv-done and understand latency ([example-user-journey](src:docs/megascale_viewer.md#example-user-journey))

## Feeds topics
- [Multi-slice collectives](../topics/multislice-collectives.md)

## Sources
- raw/code/xprof/docs/megascale_viewer.md
