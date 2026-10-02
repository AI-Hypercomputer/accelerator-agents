---
title: "Source: microbenchmarks/ironwood/automation/README.md"
type: source-page
doc: microbenchmarks/ironwood/automation/README.md
updated: 2026-07-03
---
# microbenchmarks/ironwood/automation/README.md

An automation harness that batches the manual single-job Ironwood microbenchmark workflow into a
launch → monitor/retry → aggregate pipeline, using Kueue for job queuing across many topology
configurations (e.g. `2x2x1`, `2x2x2`) at once
([overview](src:microbenchmarks/ironwood/automation/README.md#overview),
[ironwood-benchmark-automation](src:microbenchmarks/ironwood/automation/README.md#ironwood-benchmark-automation)).
Failed jobs auto-retry up to 3 times; once all jobs succeed, an aggregator job pulls every job's
intermediate results from a GCS bucket and consolidates them into summary TSV files
([directory-structure](src:microbenchmarks/ironwood/automation/README.md#directory-structure),
[output](src:microbenchmarks/ironwood/automation/README.md#output)). It requires a pre-existing GKE
cluster with the target node pools already provisioned
([prerequisites](src:microbenchmarks/ironwood/automation/README.md#prerequisites)) — a static
counterpart to the autoscaling/CCC variant.

## Feeds
- [TPU microbenchmarks](../topics/tpu-microbenchmarks.md) — the fleet-scale automation layer this
  topic's benchmark-running section describes.
